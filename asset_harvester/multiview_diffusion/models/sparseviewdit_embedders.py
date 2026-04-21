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
Embedding modules for SparseViewDiT models.
"""

import torch
import torch.nn as nn


class BasicCameraEmbedder(nn.Module):
    """
    Embeds camera-related vectors (camera matrices, FOV, augmentation params, etc.)
    into vector representations.

    Args:
        hidden_size (int): The output embedding dimension.
        camera_emb_size (int, optional): The input embedding size. Defaults to 17
            (16 for flattened 4x4 camera matrix + 1 for FOV).
    """

    def __init__(self, hidden_size: int, camera_emb_size: int = 17):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(camera_emb_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.camera_emb_size = camera_emb_size

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Camera embedding tensor of shape [batch_size, camera_emb_size]

        Returns:
            Embedded tensor of shape [batch_size, hidden_size]
        """
        t_emb = self.mlp(t.to(self.dtype))
        return t_emb

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype of the module parameters."""
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32
