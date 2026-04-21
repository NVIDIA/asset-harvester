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

import imageio
import matplotlib.cm as cm
import numpy as np
import torch
from einops import rearrange


def image_batch_to_grid(images: torch.Tensor) -> torch.Tensor:
    """
    Convert a batch of images to a grid of images.
    Each batch item is a row, with views shown as columns.
    """
    assert images.ndim == 5
    assert images.shape[2] == 3

    return rearrange(images, "b v c h w -> (b h) (v w) c")


def save_image_grid(images: torch.Tensor, output_path: str):
    images = (images.clamp(0, 1) * 255).to(torch.uint8)
    images_grid = image_batch_to_grid(images)
    imageio.imwrite(output_path, images_grid.cpu().numpy())


def image_views_to_square_grid(images: torch.Tensor) -> torch.Tensor:
    """
    Arrange multiple views of a single scene in a square-like grid.

    Args:
        images: Tensor of shape (v, c, h, w) where v is number of views

    Returns:
        Grid image arranged in a roughly square layout
    """
    assert images.ndim == 4, f"Expected 4D tensor (v, c, h, w), got {images.ndim}D"
    assert images.shape[1] == 3, f"Expected 3 channels, got {images.shape[1]}"

    n_views = images.shape[0]

    # Find the best grid dimensions (close to square)
    # Try to make n_rows <= n_cols for a horizontal-ish rectangle
    n_cols = math.ceil(math.sqrt(n_views))
    n_rows = math.ceil(n_views / n_cols)

    # Pad if necessary to fill the grid
    n_needed = n_rows * n_cols
    if n_needed > n_views:
        # Pad with black images
        padding = torch.zeros(
            n_needed - n_views,
            images.shape[1],
            images.shape[2],
            images.shape[3],
            dtype=images.dtype,
            device=images.device,
        )
        images = torch.cat([images, padding], dim=0)

    # Reshape to grid: (n_rows, n_cols, c, h, w)
    images = images.reshape(n_rows, n_cols, images.shape[1], images.shape[2], images.shape[3])

    # Rearrange to final image grid
    grid = rearrange(images, "r c ch h w -> (r h) (c w) ch")

    return grid


def save_image_grid_square(images: torch.Tensor, output_path: str):
    """
    Save views of a single scene arranged in a square-like grid.

    Args:
        images: Tensor of shape (v, c, h, w) where v is number of views
        output_path: Path to save the image
    """
    images = (images.clamp(0, 1) * 255).to(torch.uint8)
    images_grid = image_views_to_square_grid(images)
    imageio.imwrite(output_path, images_grid.cpu().numpy())


def visualize_depth(depth: torch.Tensor, vmin: float = None, vmax: float = None) -> torch.Tensor:
    """
    Convert depth maps to RGB visualization using a colormap.

    Args:
        depth: Tensor of shape (v, 1, h, w) or (v, h, w) where v is number of views
        vmin: Minimum depth value for normalization (None = auto)
        vmax: Maximum depth value for normalization (None = auto)

    Returns:
        RGB tensor of shape (v, 3, h, w) with values in [0, 1]
    """
    # Handle both (v, 1, h, w) and (v, h, w) formats
    if depth.ndim == 4 and depth.shape[1] == 1:
        depth = depth.squeeze(1)  # (v, h, w)

    assert depth.ndim == 3, f"Expected 3D tensor (v, h, w), got {depth.ndim}D"

    # Normalize depth to [0, 1]
    if vmin is None:
        vmin = depth.min()
    if vmax is None:
        vmax = depth.max()

    depth_normalized = (depth - vmin) / (vmax - vmin + 1e-8)
    depth_normalized = depth_normalized.clamp(0, 1)

    # Apply turbo colormap
    # Convert to numpy for colormap application
    depth_np = depth_normalized.cpu().numpy()  # (v, h, w)

    # Use turbo colormap (good for depth visualization)
    cmap = cm.get_cmap("turbo")

    # Apply colormap to each view
    colored_depth = []
    for i in range(depth_np.shape[0]):
        depth_colored = cmap(depth_np[i])  # (h, w, 4) - RGBA
        depth_colored = depth_colored[..., :3]  # (h, w, 3) - RGB only
        colored_depth.append(depth_colored)

    colored_depth = torch.from_numpy(np.stack(colored_depth)).float()  # (v, h, w, 3)
    colored_depth = colored_depth.permute(0, 3, 1, 2)  # (v, 3, h, w)

    return colored_depth.to(depth.device)
