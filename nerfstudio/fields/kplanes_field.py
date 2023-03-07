# Copyright 2022 The Plenoptix Team. All rights reserved.
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

"""KPlanes Field"""


import itertools
from typing import Collection, Dict, Iterable, List, Optional, Sequence, Union

import tinycudann as tcnn
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torchtyping import TensorType

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field
from nerfstudio.utils.interpolation import grid_sample_wrapper


def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]
    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


def init_grid_param(in_dim: int, out_dim: int, reso: Sequence[int], a: float = 0.1, b: float = 0.5):
    """TODO"""
    assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
    has_time_planes = in_dim == 4
    coo_combs = list(itertools.combinations(range(in_dim), 2))
    grid_coefs = nn.ParameterList()
    for ci, coo_comb in enumerate(coo_combs):
        new_grid_coef = nn.Parameter(torch.empty(
            [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
        ))
        if has_time_planes and 3 in coo_comb:  # Initialize time planes to 1
            nn.init.ones_(new_grid_coef)
        else:  # Initialize spatial planes as uniform[a, b]
            nn.init.uniform_(new_grid_coef, a=a, b=b)
        grid_coefs.append(new_grid_coef)

    return grid_coefs


def interpolate_kplanes(
    pts: torch.Tensor,
    ms_grids: Collection[Iterable[nn.Module]],
    concat_features: bool,
) -> torch.Tensor:
    """K-Planes: query multi-scale planes at given points

    Args:
        pts: 3D or 4D points at which the planes are queries
        ms_grids: Multi-scale k-plane grids
        concat_features: If true, the features from each scale are concatenated.
            Otherwise they are summed together.
    """
    coo_combs = list(itertools.combinations(range(pts.shape[-1]), 2))
    multi_scale_interp = [] if concat_features else 0.0
    grid: nn.ParameterList
    for scale_id, grid in enumerate(ms_grids):  # type: ignore
        interp_space = 1.0
        for ci, coo_comb in enumerate(coo_combs):
            # interpolate in plane
            feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
            interp_out_plane = grid_sample_wrapper(grid[ci], pts[..., coo_comb]).view(-1, feature_dim)
            # compute product over planes
            interp_space = interp_space * interp_out_plane

        # combine over scales
        if concat_features:
            multi_scale_interp.append(interp_space)  # type: ignore
        else:
            multi_scale_interp = multi_scale_interp + interp_space  # type: ignore

    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)  # type: ignore
    return multi_scale_interp  # type: ignore


class KPlanesField(Field):
    """TensoRF Field"""

    def __init__(
        self,
        aabb,
        # the aabb bounding box of the dataset
        appearance_dim: int = 27,
        # the number of dimensions for the appearance embedding
        spatial_distortion: Optional[SpatialDistortion] = None,
        grid_config: Union[str, List[Dict]] = "",
        num_images: int = 0,
        multiscale_res: Optional[Sequence[int]] = None,
        concat_features_across_scales: bool = False,
        linear_decoder: bool = True,
        linear_decoder_layers: Optional[int] = None,
        use_appearance_embedding: bool = False,
        disable_viewing_dependent: bool = False,
    ) -> None:
        super().__init__()
        self.aabb = Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion
        self.grid_config = grid_config
        self.num_images = num_images

        self.multiscale_res_multipliers: List[int] = multiscale_res or [1]  # type: ignore

        self.concat_features_across_scales = concat_features_across_scales
        self.linear_decoder = linear_decoder
        # 1. Init planes
        self.grids = nn.ModuleList()
        self.feature_dim = 0

        for res in self.multiscale_res_multipliers:
            config = self.grid_config[0].copy()  # type: ignore
            # Resolution fix: multi-res only on spatial planes
            config["resolution"] = [r * res for r in config["resolution"][:3]] + config["resolution"][3:]
            gp = init_grid_param(
                in_dim=config["input_coordinate_dim"],
                out_dim=config["output_coordinate_dim"],
                reso=config["resolution"],
            )
            # shape[1] is out-dim - Concatenate over feature len for each scale
            if self.concat_features_across_scales:
                self.feature_dim += gp[-1].shape[1]
            else:
                self.feature_dim = gp[-1].shape[1]
            self.grids.append(gp)

        # Initialize appearance code-related parameters
        self.use_appearance_embedding = use_appearance_embedding
        self.appearance_embedding = None
        self.appearance_embedding_dim = 0
        if use_appearance_embedding:
            assert self.num_images is not None
            self.appearance_embedding_dim = appearance_dim
            # this will initialize as normal_(0.0, 1.0)
            self.appearance_embedding = Embedding(self.num_images, self.appearance_embedding_dim)

        # Initialize direction encoder
        self.disable_viewing_dependent = disable_viewing_dependent
        if not self.disable_viewing_dependent:
            self.direction_encoder = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": 4,
                },
            )

        # Initialize decoder network
        if self.linear_decoder:
            assert linear_decoder_layers is not None
            # The NN learns a basis that is used instead of spherical harmonics
            # Input is an encoded view direction, output is weights for
            # combining the color features into RGB. This architecture is based on instant-NGP
            self.color_basis = tcnn.Network(
                n_input_dims=3 + self.appearance_embedding_dim,
                n_output_dims=3 * self.feature_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 128,
                    "n_hidden_layers": linear_decoder_layers,
                },
            )
            # sigma_net just does a linear transformation on the features to get density
            self.sigma_net = tcnn.Network(
                n_input_dims=self.feature_dim,
                n_output_dims=1,
                network_config={
                    "otype": "CutlassMLP",
                    "activation": "None",
                    "output_activation": "None",
                    "n_neurons": 128,
                    "n_hidden_layers": 0,
                },
            )
        else:
            self.geo_feat_dim = 15
            self.sigma_net = tcnn.Network(
                n_input_dims=self.feature_dim,
                n_output_dims=self.geo_feat_dim + 1,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                },
            )
            self.in_dim_color = self.geo_feat_dim + self.appearance_embedding_dim
            if not disable_viewing_dependent:
                self.in_dim_color += self.direction_encoder.n_output_dims
            self.color_net = tcnn.Network(
                n_input_dims=self.in_dim_color,
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "Sigmoid",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                },
            )

    def get_density(self, ray_samples: RaySamples):
        """Computes and returns the densities."""
        positions = ray_samples.frustums.get_positions()
        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0  # from [-2, 2] to [-1, 1]
        else:
            positions = SceneBox.get_normalized_positions(positions, self.aabb)
        n_rays, n_samples = positions.shape[:2]

        timestamps = ray_samples.times
        if timestamps is not None:
            # Normalize timestamps from [0, 1] to [-1, 1]
            timestamps = (timestamps * 2) - 1
            positions = torch.cat((positions, timestamps), dim=-1)  # [n_rays, n_samples, 4]

        positions = positions.reshape(-1, positions.shape[-1])

        features = interpolate_kplanes(
            positions,
            ms_grids=self.grids,  # type: ignore
            concat_features=self.concat_features_across_scales,
        )

        if len(features) < 1:
            features = torch.zeros((0, 1)).to(features.device)
        if self.linear_decoder:
            density_before_activation = self.sigma_net(features)  # [batch, 1]
        else:
            features = self.sigma_net(features)
            features, density_before_activation = torch.split(features, [self.geo_feat_dim, 1], dim=-1)

        density = trunc_exp(density_before_activation.to(pts)).view(n_rays, n_samples, 1)  # type: ignore
        return density, features

    def get_outputs(self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None) -> TensorType:
        assert density_embedding is not None
        n_rays, n_samples = ray_samples.frustums.shape

        directions = ray_samples.frustums.directions.reshape(-1, 3)
        if self.use_linear_decoder or self.disable_viewing_dependent:
            color_features = [density_embedding]
        else:
            directions = get_normalized_directions(directions)
            encoded_directions = self.direction_encoder(directions)
            color_features = [encoded_directions, density_embedding]

        if self.use_appearance_embedding:
            assert ray_samples.camera_indices is not None
            camera_indices = ray_samples.camera_indices.squeeze()
            if self.training:
                embedded_appearance = self.appearance_embedding(camera_indices)
            else:
                # Average of appearance embeddings for test data
                embedded_appearance = torch.ones(
                    (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                ) * self.appearance_embedding.mean(dim=0)

            # expand embedded_appearance from n_rays, dim to n_rays*n_samples, dim
            ea_dim = embedded_appearance.shape[-1]
            embedded_appearance = (
                embedded_appearance.view(-1, 1, ea_dim)
                                   .expand(n_rays, n_samples, -1)
                                   .reshape(-1, ea_dim)
            )
            if self.use_linear_decoder:
                directions = torch.cat((directions, embedded_appearance), dim=-1)
            else:
                color_features.append(embedded_appearance)

        color_features = torch.cat(color_features, dim=-1)

        if self.linear_decoder:
            basis_values = self.color_basis(directions)  # [batch, color_feature_len * 3]
            basis_values = basis_values.view(color_features.shape[0], 3, -1)  # [batch, 3, color_feature_len]
            rgb = torch.sum(color_features[:, None, :] * basis_values, dim=-1)  # [batch, 3]
            rgb = rgb.to(directions)
            rgb = torch.sigmoid(rgb).view(n_rays, n_samples, 3)
        else:
            rgb = self.color_net(color_features).to(directions).view(n_rays, n_samples, 3)

        return rgb

    def forward(
        self,
        ray_samples: RaySamples,
        compute_normals: bool = False,
        mask: Optional[TensorType] = None,
        bg_color: Optional[TensorType] = None,
    ):
        density, density_features = self.get_density(ray_samples)
        rgb = self.get_outputs(ray_samples, density_features)  # type: ignore

        return {FieldHeadNames.DENSITY: density, FieldHeadNames.RGB: rgb}
