# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.init import trunc_normal_

from ..options import Options


class BaseActivationHead(nn.Module, ABC):
    """
    Base class for converting tokens to Gaussians.

    This is a minimal interface - subclasses can have completely different
    internal structures and initialization logic.
    """

    def __init__(self, opt: Options):
        super().__init__()
        self.opt = opt

        self.num_gaussians_per_token = self.opt.deconv_patch_size**2

        scale_cap = self.opt.gaussian_scale_cap
        if self.opt.scale_shift == "default":
            self.scale_shift = 1 - math.log(scale_cap)
        else:
            self.scale_shift = self.opt.scale_shift
        self.scale_cap = scale_cap

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        rays_os: torch.Tensor | None = None,
        rays_ds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Convert tokens to Gaussians.

        Args:
            x: Input tokens [B, N, C] where N is number of tokens
            rays_os: Optional ray origins [B, V, 3, H, W] for depth prediction
            rays_ds: Optional ray directions [B, V, 3, H, W] for depth prediction

        Returns:
            Gaussians [B, N * num_gaussians_per_token, 14]
        """
        pass


class ClipActivationHead(BaseActivationHead):
    """
    Clip-based activation head that uses exponential activations with hard clipping.
    Uses a single deconv (linear) layer for all Gaussian components.
    Supports both xyz and depth prediction modes.
    """

    def __init__(self, opt: Options):
        super().__init__(opt)

        # Set output dimensions based on prediction mode
        self.output_dims = 3 + 1 + 3 + 4 + 3  # x, y, z, opacity, scale, rotation, rgb

        # Single linear layer (deconv) for all components
        self.deconv = nn.Linear(
            self.opt.enc_embed_dim,
            self.output_dims * self.opt.deconv_patch_size * self.opt.deconv_patch_size,
            bias=False,
        )

        if self.opt.init_deconv:
            self._init_weights(self.deconv)

        # For depth prediction
        self.dnear = self.opt.dnear
        self.dfar = self.opt.dfar

        # Mark parameters to exclude from weight decay
        for param in self.parameters():
            param._no_weight_decay = True

    def _init_weights(self, m):
        """Initialize deconv weights"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.002)
            if m.bias is not None:
                m.bias.data.zero_()

    def pos_act(self, x: torch.Tensor) -> torch.Tensor:
        """Position activation without scaling factor (for xyz prediction)"""
        pos = torch.sign(x) * (torch.expm1(torch.abs(x)))  # inverse log transform
        if self.opt.safe_mode:
            pos = torch.clamp(pos, min=-self.opt.zfar, max=self.opt.zfar)
        return pos

    def scale_act(self, x: torch.Tensor) -> torch.Tensor:
        """Scale activation with exponential and hard clipping"""
        x = x - self.scale_shift
        if self.opt.safe_mode:
            x = torch.clamp(x, min=-10, max=10)

        return torch.minimum(
            torch.exp(x),
            torch.tensor(self.scale_cap, device=x.device, dtype=x.dtype),
        ).clamp(min=self.opt.scale_min)

    def opacity_act(self, x: torch.Tensor) -> torch.Tensor:
        """Opacity activation with hard clipping"""
        return torch.sigmoid(x - 2.0).clamp(min=self.opt.opacity_min)

    def rot_act(self, x: torch.Tensor) -> torch.Tensor:
        """Rotation normalization"""
        return F.normalize(x, dim=-1)

    def rgb_act(self, x: torch.Tensor) -> torch.Tensor:
        """RGB activation (always tanh for clip-based head)"""
        return 0.5 * torch.tanh(x) + 0.5

    def forward(
        self,
        x: torch.Tensor,
        rays_os: torch.Tensor | None = None,
        rays_ds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Convert tokens to Gaussians using a single deconv layer.
        Supports both xyz and depth prediction modes.

        Args:
            x: Input tokens [B, N, C]
            rays_os: Ray origins [B, V, 3, H, W] (required for depth prediction)
            rays_ds: Ray directions [B, V, 3, H, W] (required for depth prediction)

        Returns:
            Gaussians [B, N * num_gaussians_per_token, 14]
        """
        B = x.shape[0]

        # Process through deconv
        x = x.permute(0, 2, 1).unsqueeze(-1)  # B, C, N, 1

        x = x.squeeze(-1).permute(0, 2, 1)  # B, N, C
        x = self.deconv(x)  # [B, N, output_dims * P * P]
        x = rearrange(x, "b n (p c) -> b c (n p)", p=self.opt.deconv_patch_size**2)

        x = x.reshape(B, self.output_dims, -1)  # B, output_dims, N * P * P
        x = x.permute(0, 2, 1).contiguous()  # B, N * P * P, output_dims

        # Apply activations based on prediction mode
        pos, rgb, scaling, rotation, opacity = x.split([3, 3, 3, 4, 1], dim=-1)
        pos = self.pos_act(pos)

        opacity = self.opacity_act(opacity)
        scale = self.scale_act(scaling)
        rotation = self.rot_act(rotation)
        rgbs = self.rgb_act(rgb)

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1)  # [B, N, 14]

        return gaussians


class DirectClipHead(ClipActivationHead):
    """
    Uses the same activation functions as ClipActivationHead, but does not use a deconv layer.
    Additionally supports (pseudo)-inverting the activation functions to get the original values back.

    Meant for test-time training of Gaussian parameters: we first get the output of a model (gaussian parameters),
    then invert the activation functions to get back to unconstrained, real-valued parameters and then optimize in that space.
    """

    def __init__(self, opt: Options):
        super().__init__(opt)

    def pos_act_inverse(self, pos: torch.Tensor) -> torch.Tensor:
        sign = torch.sign(pos)
        log_x = torch.log(torch.abs(pos) + 1)
        return sign * log_x

    def scale_act_inverse(self, scale: torch.Tensor) -> torch.Tensor:
        return torch.log(scale) + self.scale_shift

    def opacity_act_inverse(self, opacity: torch.Tensor) -> torch.Tensor:
        # inverse of sigmoid(x - 2.0).clamp(min=opacity_min)
        # Note: clamp makes this a pseudo-inverse
        opacity_unclamped = torch.clamp(opacity, min=self.opt.opacity_min + 1e-8, max=1.0 - 1e-8)
        return torch.logit(opacity_unclamped) + 2.0

    def rot_act_inverse(self, rotation: torch.Tensor) -> torch.Tensor:
        return rotation  # normalized rotation is a valid quaternion

    def rgb_act_inverse(self, rgb: torch.Tensor) -> torch.Tensor:
        # inverse of 0.5 * tanh(x) + 0.5
        tanh = (rgb - 0.5) * 2
        # Clamp to avoid numerical issues at boundaries (-1, 1)
        tanh = torch.clamp(tanh, min=-1.0 + 1e-7, max=1.0 - 1e-7)
        return torch.atanh(tanh)

    def forward_inverse(self, gaussians: torch.Tensor) -> torch.Tensor:
        pos, opacity, scale, rotation, rgb = gaussians.split([3, 1, 3, 4, 3], dim=-1)
        return torch.cat(
            [
                self.pos_act_inverse(pos),
                self.rgb_act_inverse(rgb),
                self.scale_act_inverse(scale),
                self.rot_act_inverse(rotation),
                self.opacity_act_inverse(opacity),
            ],
            dim=-1,
        )

    def forward(
        self, x: torch.Tensor, rays_os: torch.Tensor | None = None, rays_ds: torch.Tensor | None = None
    ) -> torch.Tensor:
        assert rays_os is None and rays_ds is None, "DirectClipHead does not support rays_os and rays_ds"
        assert x.shape[-1] == 14

        pos, rgb, scaling, rotation, opacity = x.split([3, 3, 3, 4, 1], dim=-1)
        return torch.cat(
            [
                self.pos_act(pos),
                self.opacity_act(opacity),
                self.scale_act(scaling),
                self.rot_act(rotation),
                self.rgb_act(rgb),
            ],
            dim=-1,
        )


class ObjaverseActivationHead(BaseActivationHead):
    """
    Activation head for Objaverse dataset.
    """

    def __init__(self, opt: Options):
        super().__init__(opt)

        self.output_dims = 3 + 1 + 3 + 4 + 3  # x, y, z, opacity, scale, rotation, rgb

        # Single linear layer (deconv) for all components
        self.deconv = nn.Linear(
            self.opt.enc_embed_dim,
            self.output_dims * self.opt.deconv_patch_size * self.opt.deconv_patch_size,
            bias=True,  # bias is needed for objaverse dataset for legacy compatibility
        )

        if self.opt.init_deconv:
            self._init_weights(self.deconv)

        self.pos_act = lambda x: x.clamp(-1, 1)
        self.scale_act = lambda x: 0.1 * F.softplus(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = lambda x: F.normalize(x, dim=-1)
        self.rgb_act = lambda x: 0.5 * torch.tanh(x) + 0.5

    def _flatten_gaussians(self, x: torch.Tensor) -> torch.Tensor:
        """
        Unpacks Gaussian parameters from (batch, num_gs_tokens, num_gaussians_per_token * output_dims)
        to (batch, num_gs_tokens * num_gaussians_per_token, output_dims).
        """
        return rearrange(x, "b n (g o) -> b (n g) o", g=self.num_gaussians_per_token)

    def _init_weights(self, m):
        """Initialize deconv weights"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.002)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        rays_os: torch.Tensor | None = None,
        rays_ds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Convert tokens to Gaussians using a single deconv layer.
        Supports both xyz and depth prediction modes.

        Args:
            x: Input tokens [B, N, C]
            rays_os: Ray origins [B, V, 3, H, W] (required for depth prediction)
            rays_ds: Ray directions [B, V, 3, H, W] (required for depth prediction)

        Returns:
            Gaussians [B, N * num_gaussians_per_token, 14]
        """
        x = self.deconv(x)  # [B, N, output_dims * P * P]
        x = rearrange(x, "b n (p c) -> b (n p) c", p=self.opt.deconv_patch_size**2)

        # Apply activations based on prediction mode
        pos, rgb, scaling, rotation, opacity = x.split([3, 3, 3, 4, 1], dim=-1)
        pos = self.pos_act(pos)

        opacity = self.opacity_act(opacity)
        scale = self.scale_act(scaling)
        rotation = self.rot_act(rotation)
        rgbs = self.rgb_act(rgb)

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1)  # [B, N, 14]

        return gaussians
