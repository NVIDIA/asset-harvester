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


"""
Attribute estimation model for camera/object attribute prediction.

Provides SharedRegressionHead and AttributeModel, ported from
ah-camera-estimator/ahc/models/. The backbone is injected rather than
constructed internally, allowing the caller to reuse an already-loaded
C-RADIO model shared with the diffusion pipeline.

Output layout: [distance(1) | lwh(3) | fov(1) | cam_pose(3)] = 8 dims.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

OUTPUT_DIMS = (1, 3, 1, 3)
OUTPUT_KEYS = ("distance", "lwh", "fov", "cam_pose")
TOTAL_OUT_DIM = sum(OUTPUT_DIMS)


class SharedRegressionHead(nn.Module):
    """MLP that outputs all attributes concatenated, to be chunked by the caller."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int = TOTAL_OUT_DIM,
        hidden_dim: int = 0,
        num_layers: int = 1,
    ):
        super().__init__()
        if num_layers <= 1 or hidden_dim <= 0:
            self.net = nn.Linear(in_dim, out_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True)]
            for _ in range(num_layers - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
            layers.append(nn.Linear(hidden_dim, out_dim))
            self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CRadioBackboneAdapter(nn.Module):
    """
    Thin adapter that wraps an already-loaded C-RADIO (model, processor) pair.

    Converts (B, 3, H, W) float tensors in [0, 1] to the global summary
    feature vector (B, C) expected by the regression head. The model and
    processor are shared with the diffusion pipeline to avoid loading the
    ~multi-GB backbone twice.
    """

    def __init__(self, cradio_model: nn.Module, cradio_image_processor):
        super().__init__()
        self.encoder = cradio_model
        self.image_processor = cradio_image_processor
        self._feat_dim: int | None = None

    @property
    def feat_dim(self) -> int:
        if self._feat_dim is not None:
            return self._feat_dim
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            pv = self._to_pixel_values(dummy)
            device = next(self.encoder.parameters()).device
            pv = pv.to(device)
            summary, _ = self.encoder(pv)
            self._feat_dim = int(summary.shape[1])
        return self._feat_dim

    def _to_pixel_values(self, images: torch.Tensor) -> torch.Tensor:
        """Convert (B, 3, H, W) in [0, 1] to processor pixel_values."""
        pil_images = [to_pil_image(images[i].cpu()) for i in range(images.shape[0])]
        out = self.image_processor(images=pil_images, return_tensors="pt", do_resize=True)
        return out.pixel_values

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pixel_values = self._to_pixel_values(images).to(images.device)
        summary, _ = self.encoder(pixel_values)
        return summary


class AttributeModel(nn.Module):
    """
    Backbone adapter + shared regression head.

    Predicts distance, lwh (length/width/height), fov, and cam_pose from
    a single masked image.

    Args:
        backbone: A nn.Module with forward(B, 3, H, W) -> (B, feat_dim).
                  Typically a CRadioBackboneAdapter.
        feat_dim: Feature dimension output by backbone.
        head_hidden_dim: Hidden dim for the MLP head (0 = single linear layer).
        head_num_layers: Number of layers in the MLP head.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feat_dim: int,
        head_hidden_dim: int = 256,
        head_num_layers: int = 1,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = SharedRegressionHead(feat_dim, TOTAL_OUT_DIM, head_hidden_dim, head_num_layers)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            image: (B, 3, H, W) float tensor in [0, 1].
        Returns:
            Dict with keys distance (B,), lwh (B, 3), fov (B,), cam_pose (B, 3).
        """
        feat = self.backbone(image)
        out = self.head(feat)
        start = 0
        result = {}
        for key, dim in zip(OUTPUT_KEYS, OUTPUT_DIMS):
            chunk = out[:, start : start + dim]
            if key == "distance":
                chunk = torch.exp(chunk)
            elif key == "cam_pose":
                chunk = F.normalize(chunk, dim=-1)
            result[key] = chunk.squeeze(-1) if dim == 1 else chunk
            start += dim
        return result
