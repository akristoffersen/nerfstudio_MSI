# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
TensorRF implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.kplanes_field import KPlanesField, KPlanesDensityField
from nerfstudio.model_components.losses import MSELoss, distortion_loss, interlevel_loss
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler
from nerfstudio.model_components.renderers import (
    AccumulationRenderer,
    DepthRenderer,
    RGBRenderer,
)
from nerfstudio.model_components.scene_colliders import AABBBoxCollider, NearFarCollider
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, colors, misc


@dataclass
class KPlanesModelConfig(ModelConfig):
    """K-Planes model config"""

    _target: Type = field(default_factory=lambda: KPlanesModel)
    """target class to instantiate"""
    near_plane: float = 0.05
    """How far along the ray to start sampling."""
    far_plane: float = 1000.0
    """How far along the ray to stop sampling."""
    grid_config: List[Dict] = field(
        default_factory=lambda: [
            {
                "input_coordinate_dim": 4,
                "output_coordinate_dim": 16,
                "resolution": [64, 64, 64, 150],
            },
        ]
    )
    multiscale_res: Sequence[int] = (1, 2, 4, 8)

    concat_features_across_scales: bool = True
    linear_decoder: bool = False
    linear_decoder_layers: Optional[int] = 1
    # Spatial distortion
    global_translation: Optional[torch.Tensor] = None
    global_scale: Optional[torch.Tensor] = None
    # proposal-sampling arguments
    num_proposal_iterations: int = 2
    use_same_proposal_network: bool = False
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"num_input_coords": 4, "num_output_coords": 8, "resolution": [128, 128, 128, 150]},
            {"num_input_coords": 4, "num_output_coords": 8, "resolution": [256, 256, 256, 150]},
        ]
    )
    num_proposal_samples: Optional[Tuple[int, int]] = (256, 128)
    # num_samples: Optional[int] = None
    single_jitter: bool = False
    proposal_warmup: int = 5000
    proposal_update_every: int = 5
    use_proposal_weight_anneal: bool = True
    proposal_weights_anneal_max_num_iters: int = 1000
    proposal_weights_anneal_slope: float = 10.0
    # appearance embedding (phototourism)
    use_appearance_embedding: bool = False
    appearance_embedding_dim: int = 0
    disable_viewing_dependent: bool = False 
    """If true, color is independent of viewing direction. (Neural Decoder Only)"""
    num_images: Optional[int] = None
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    """Number of samples per ray for each proposal network."""
    num_nerf_samples_per_ray: int = 48
    """Number of samples per ray for the nerf network."""
    loss_coefficients: Dict[str, float] = to_immutable_dict({"rgb_loss": 1.0})
    """Loss specific weights."""
    interlevel_loss_mult: float = 1.0
    """Proposal loss multiplier."""
    distortion_loss_mult: float = 0.002
    """Distortion loss multiplier."""


class KPlanesModel(Model):
    """K-Planes Model

    Args:
        config: K-Planes configuration to instantiate model
    """

    config: KPlanesModelConfig

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        if isinstance(self.config.grid_config, str):
            self.grid_config: Sequence[Dict] = eval(self.config.grid_config)
        else:
            self.grid_config: Sequence[Dict] = self.config.grid_config

        self.concat_features_across_scales = self.config.concat_features_across_scales
        linear_decoder = self.config.linear_decoder
        self.linear_decoder_layers = self.config.linear_decoder_layers

        scene_contraction = SceneContraction(order=float("inf"))

        self.field = KPlanesField(
            self.scene_box.aabb,
            grid_config=self.grid_config,
            concat_features_across_scales=self.concat_features_across_scales,
            multiscale_res=self.config.multiscale_res,
            use_appearance_embedding=self.config.use_appearance_embedding,
            appearance_dim=self.config.appearance_embedding_dim,
            spatial_distortion=scene_contraction,
            linear_decoder=linear_decoder,
            linear_decoder_layers=self.linear_decoder_layers,
            num_images=self.num_train_data,
            disable_viewing_dependent=self.config.disable_viewing_dependent,
        )

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = KPlanesDensityField(
                self.scene_box.aabb, spatial_distortion=scene_contraction,
                linear_decoder=linear_decoder, **prop_net_args)
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = KPlanesDensityField(
                    self.scene_box.aabb, spatial_distortion=scene_contraction,
                    linear_decoder=linear_decoder, **prop_net_args)
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # Samplers
        update_schedule = lambda step: np.clip(
            np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
            1,
            self.proposal_update_every,
        )

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.single_jitter,
            update_sched=update_schedule,
        )

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=colors.BLACK)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {
            "proposal_networks": list(self.proposal_networks.parameters()),
            "fields": list(self.field.parameters())
        }
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)
                bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        return callbacks

    def get_outputs(self, ray_bundle: RayBundle):
        # uniform sampling

        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
            ray_bundle, density_fns=self.density_fns
        )
        field_out = self.field(ray_samples)
        rgb, density = field_out[FieldHeadNames.RGB], field_out[FieldHeadNames.DENSITY]

        weights = ray_samples.get_weights(density)
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=rgb, weights=weights)
        accumulation = self.renderer_accumulation(weights)
        depth = self.renderer_depth(weights, ray_samples)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }

        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        for i in range(self.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(
                weights=weights_list[i], ray_samples=ray_samples_list[i]
            )

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        device = outputs["rgb"].device
        image = batch["image"].to(device)

        rgb_loss = self.rgb_loss(image, outputs["rgb"])
        loss_dict = {"rgb_loss": rgb_loss}
        loss_coef = self.config.loss_coefficients

        if self.training:
            if "distortion_loss" in loss_coef:
                loss_dict["distortion_loss"] = distortion_loss(
                    outputs["weights_list"], outputs["ray_samples_list"]
                )
            if "interlevel_loss" in loss_coef:
                loss_dict["interlevel_loss"] = interlevel_loss(
                    outputs["weights_list"], outputs["ray_samples_list"]
                )

        loss_dict = misc.scale_dict(loss_dict, loss_coef)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(outputs["rgb"].device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
            near_plane=self.config.collider_params["near_plane"],
            far_plane=self.config.collider_params["far_plane"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        metrics_dict = {
            "psnr": float(psnr.item()),
            "ssim": float(ssim.item()),
            "lpips": float(lpips.item()),
        }
        images_dict = {"img": combined_rgb, "accumulation": acc, "depth": depth}
        return metrics_dict, images_dict
