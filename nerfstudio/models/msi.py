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
Implementation of vanilla nerf.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchtyping import TensorType

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.model_components.losses import MSELoss, total_variation
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import misc


@dataclass
class MSIModelConfig(ModelConfig):
    """MSI model config"""

    _target: Type = field(default_factory=lambda: MSIModel)
    """target class to instantiate"""
    h: int = 960
    w: int = 1920
    num_msis: int = 2
    nlayers: int = 16
    nsublayers: int = 2
    dmin: float = 2.0
    dmax: float = 20.0
    poses_src: torch.Tensor = torch.eye(4).unsqueeze(0)
    sigmoid_offset: float = 5.0

    loss_coefficients: Dict[str, float] = to_immutable_dict({"rgb_loss": 1.0, "tv_loss": 0.05})


class MSI_field(nn.Module):
    def __init__(self, nlayers, nsublayers, dmin, dmax, pose, H, W, sigmoid_offset):
        super().__init__()
        self.nlayers = nlayers
        self.nsublayers = nsublayers
        self.dmin = dmin
        self.dmax = dmax
        self.pose = pose

        self.H, self.W = H, W

        self.n_total_layers = self.nlayers * self.nsublayers

        self.planes_init = 1.0 / torch.linspace(1.0 / self.dmin, 1.0 / self.dmax, self.n_total_layers).cuda()

        deltas = torch.diff(self.planes_init)
        self.layer_deltas = Parameter(torch.log(deltas).cuda(), requires_grad=False)

        self.sigmoid_offset = sigmoid_offset

        self.alpha = Parameter(torch.zeros(self.n_total_layers, 1, H, W).uniform_(-1, 1).cuda(), requires_grad=True)
        self.rgb = Parameter(torch.zeros(self.nlayers, 3, H, W).uniform_(-1, 1).cuda(), requires_grad=True)

    def calculate_radii(self):

        radii = torch.zeros((self.n_total_layers,)).cuda()
        radii[0] = self.dmin

        radii[1:] = torch.exp(self.layer_deltas)

        return torch.cumsum(radii, dim=0)

    def forward(self, ray_bundle: RayBundle):

        center_src = self.pose[:3, 3]

        radiis = self.calculate_radii()

        # get the intersection (world coords) of each ray with each of the concentric spheres
        intersections, mask = MSIModel.intersect_rays_with_spheres(ray_bundle, center_src, radiis)

        # make them in MSI space
        xyzs = intersections - center_src  # (N, R, 3)
        # normalize by radius
        xyzs_normalized = xyzs / radiis.reshape(1, -1, 1)  # (N, R, 3)

        # convert these into uv coordinates (equirectangular projection)
        uvs = torch.stack(
            [
                xyzs_normalized[..., 2],
                torch.atan2(xyzs_normalized[..., 1], -xyzs_normalized[..., 0]) / (torch.pi),
            ],
            dim=2,
        )  # (N, R, 2)

        # sample the alphas
        uvs = uvs.permute(1, 0, 2).unsqueeze(1)  # (R, 1, N, 2)
        alphas = F.grid_sample(self.alpha, uvs, align_corners=True, padding_mode="reflection")  # (R, 1, 1, N)
        alphas_sig = torch.sigmoid(alphas - self.sigmoid_offset)  # (R, 1, 1, N)
        alphas_sig = alphas_sig.permute(0, 1, 3, 2)  # (R, 1, N, 1)

        # adding mask for when intersections are bad (alpha -> 0 if so) (N, R)
        alphas_sig_clone = alphas_sig.clone()  # clone done to allow gradients to flow
        mask = mask.permute(1, 0).unsqueeze(1).unsqueeze(-1)
        alphas_sig_clone[~mask] = 0
        alphas_sig = alphas_sig_clone

        # sample the RGBs
        rgbs = F.grid_sample(
            self.rgb,
            uvs[:: self.nsublayers],
            align_corners=True,
            padding_mode="reflection",
        )  # (L // sublayers, 3, 1, N)
        rgbs = torch.sigmoid(rgbs)
        rgbs = rgbs.permute(0, 1, 3, 2)  # (L // sublayers, 3, N, 1)

        # since RGBs are samples at a smaller rate, we will repeat interleave to get the
        # same output shape as the alphas
        rgbs = rgbs.repeat_interleave(self.nsublayers, dim=0)  # (R, 3, N, 1)

        # accrue alphas over the rays
        weight = misc.cumprod(1 - alphas_sig, exclusive=True) * alphas_sig

        # weighted sum
        output_vals = torch.sum(weight * rgbs, dim=0, keepdim=True)  # [1, 3, N, 1]

        return output_vals

    def calculate_tv_loss(self):
        tv_loss = total_variation(torch.sigmoid(self.rgb)) + total_variation(
            torch.sigmoid(self.alpha - self.sigmoid_offset)
        )
        return tv_loss


class MSIModel(Model):
    """MSI model

    CUDA_VISIBLE_DEVICES=1 python3 scripts/train.py nerfacto-msi --data /home/akristoffersen/data/bww/ --vis viewer --viewer.websocket-port=7008

    Args:
        config: Basic NeRF configuration to instantiate model
    """

    config: MSIModelConfig

    def __init__(
        self,
        config: MSIModelConfig,
        **kwargs,
    ) -> None:

        input_centers = kwargs["data_poses"]  # (K, 3)
        kmeans = KMeans(n_clusters=config.num_msis).fit(input_centers)
        msi_centers = torch.from_numpy(kmeans.cluster_centers_)

        self.poses_src = torch.eye(4).repeat(config.num_msis, 1, 1)
        self.poses_src[:, :3, 3] = msi_centers

        self.poses_src = self.poses_src.cuda()

        print(msi_centers)

        super().__init__(config=config, **kwargs)

        # H = self.config.h
        # W = self.config.w

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        self.msi_fields = nn.ModuleList(
            [
                MSI_field(
                    self.config.nlayers,
                    self.config.nsublayers,
                    self.config.dmin,
                    self.config.dmax,
                    self.poses_src[i],
                    self.config.h,
                    self.config.w,
                    self.config.sigmoid_offset,
                )
                for i in range(self.config.num_msis)
            ]
        )

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity()
        self.tv_loss = total_variation
        self.rgb_loss = MSELoss()

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}

        param_groups["planes"] = list(itertools.chain(*[msi_field.parameters() for msi_field in self.msi_fields]))
        print([param.shape for param in param_groups["planes"]])

        return param_groups

    @classmethod
    def intersect_rays_with_spheres(
        cls, rays: RayBundle, center=torch.zeros(3), radii=torch.Tensor([1.0])
    ) -> Tuple[TensorType["num_rays", "num_layers", 3], TensorType["num_rays", "num_layers"]]:
        """Intersect provided rays with multiple spheres

        :param rays: RayBundle
        :param center: (1, 3)
        :param radii: (L,)
        """
        R = radii.shape[-1]
        N = rays.shape[0]
        O, D = rays.origins, rays.directions

        O = O.unsqueeze(1).tile(1, R, 1)
        D = D.unsqueeze(1).tile(1, R, 1)

        # print("O", O.min(), O.max())

        # compute quadratic form coefficients
        ray = O - center
        a = (D**2.0).sum(dim=-1).view(N, R)
        b = 2 * (D * ray).sum(dim=-1).view(N, R)
        c = (ray**2.0).sum(dim=-1).view(N, R) - radii[None] ** 2.0

        # solve for ray intersection with sphere
        discriminant = b**2.0 - 4 * a * c  # (N, R)
        t0 = (-b + discriminant.sqrt()) / (2.0 * a)  # (N, R)
        t1 = (-b - discriminant.sqrt()) / (2.0 * a)  # (N, R)

        # ignore rays that miss (both neg) or intersect from outside (both pos)
        mask = t0 * t1 < 0  # (N, R) - if sign differs, then ray intersects from the inside
        intersection = O + D * t0[:, :, None]  # (N, R, 3)
        return intersection, mask  # (N, R, 3)

    def get_outputs(self, ray_bundle: RayBundle, inference=False):
        # https://diegoinacio.github.io/computer-vision-notebooks-page/pages/ray-intersection_sphere.html

        outputs = {}
        ray_bundle_shape = ray_bundle.shape

        ray_bundle = ray_bundle.flatten()

        if self.config.num_msis == 1:
            output_vals = self.msi_fields[0](ray_bundle).permute(0, 2, 1, 3).squeeze(0).squeeze(-1)
        else:
            distances = torch.cdist(ray_bundle.origins, self.poses_src[:, :3, 3])  # (N, K)

            if inference:
                # this means inference
                indices = torch.argmin(distances, dim=1)
                if distances.isnan().any():
                    print("uh oh")
            else:
                probabilities = 1 - F.softmax(distances, dim=1)
                top_k_probs, top_k_indices = torch.topk(probabilities, 2, dim=1)

                indices = top_k_indices[:, 1]

                probability_mask = torch.rand((ray_bundle.size,), device="cuda") < top_k_probs[:, 0]
                indices[probability_mask] = top_k_indices[probability_mask][:, 0]

            output_vals = torch.zeros((ray_bundle.size, 3)).cuda()
            for i in range(self.config.num_msis):
                # create the mask
                mask = indices == i
                if mask.any():
                    # evaluate
                    index_outputs = self.msi_fields[i](ray_bundle[mask])
                    # write to outputs
                    output_vals[mask] = index_outputs.permute(0, 2, 1, 3).squeeze(0).squeeze(-1)

        output_vals = output_vals.reshape(*ray_bundle_shape, 3)

        outputs["rgb"] = output_vals

        return outputs

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in camera parameters and computes the output of the model.

        Args:
            camera_ray_bundle: ray bundle to calculate outputs over
        """
        num_rays_per_chunk = self.config.eval_num_rays_per_chunk
        image_height, image_width = camera_ray_bundle.origins.shape[:2]
        num_rays = len(camera_ray_bundle)
        outputs_lists = defaultdict(list)
        for i in range(0, num_rays, num_rays_per_chunk):
            start_idx = i
            end_idx = i + num_rays_per_chunk
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
            outputs = self.forward(ray_bundle=ray_bundle, inference=True)
            for output_name, output in outputs.items():  # type: ignore
                outputs_lists[output_name].append(output)
        outputs = {}
        for output_name, outputs_list in outputs_lists.items():
            if not torch.is_tensor(outputs_list[0]):
                # TODO: handle lists of tensors as well
                continue
            outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
        return outputs

    def forward(self, ray_bundle: RayBundle, inference=False) -> Dict[str, torch.Tensor]:
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """

        if self.collider is not None:
            ray_bundle = self.collider(ray_bundle)

        return self.get_outputs(ray_bundle, inference=inference)

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        # Scaling metrics by coefficients to create the losses.
        device = outputs["rgb"].device
        image = batch["image"].to(device)

        # image = torch.zeros_like(image).to(device)
        # image[:, 0] = 1.0

        rgb_loss = self.rgb_loss(image, outputs["rgb"])  # (N, 3)
        # print(image.shape, outputs["rgb"].shape)
        tv_loss = sum(msi_field.calculate_tv_loss() for msi_field in self.msi_fields)

        loss_dict = {"rgb_loss": rgb_loss, "tv_loss": tv_loss}

        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(outputs["rgb"].device)
        rgb = outputs["rgb"]

        combined_rgb = torch.cat([image, rgb], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        print("test 1")
        ssim = self.ssim(image, rgb)
        print("test 2")
        # lpips = self.lpips(image, rgb)
        # print("test 3")

        metrics_dict = {
            "psnr": float(psnr.item()),
            "ssim": float(ssim),  # type: ignore
            # "lpips": float(lpips),
        }
        images_dict = {"img": combined_rgb}
        return metrics_dict, images_dict
