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

import logging
import math

import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def custom_meshgrid(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """
    Creates meshgrid with 'ij' indexing from input tensors.

    Parameters:
    - *args: Variable number of 1D tensors.

    Returns:
    - Tuple of meshgrid tensors.
    """
    return torch.meshgrid(*args, indexing="ij")


def get_grid_uvs(
    batch_shape: tuple[int, int],
    H: int,
    W: int,
    device: str | torch.device,
    dtype: torch.dtype | None = None,
    flip_flag: torch.Tensor | None = None,
    nh: int | None = None,
    nw: int | None = None,
    margin: float = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates UV coordinate grids for ray generation.

    Parameters:
    - batch_shape: Tuple of (batch_size, num_views).
    - H: Image height.
    - W: Image width.
    - device: Torch device for tensor allocation.
    - dtype: Data type for tensors.
    - flip_flag: Optional tensor indicating which views to flip.
    - nh: Number of height samples.
    - nw: Number of width samples.
    - margin: Margin to apply to UV coordinates.

    Returns:
    - Tuple of (u coordinates, v coordinates).
    """
    if dtype is None:
        dtype = torch.float32
    if nh is None:
        nh = H
    if nw is None:
        nw = W
    # c2w: B, V, 4, 4
    # K: B, V, 4
    # c2w @ dirctions
    B, V = batch_shape

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, nh, device=device, dtype=dtype),
        torch.linspace(0, W - 1, nw, device=device, dtype=dtype),
    )
    i = i.reshape([1, 1, nh * nw]).expand([B, V, nh * nw]) + 0.5  # [B, V, HxW]
    j = j.reshape([1, 1, nh * nw]).expand([B, V, nh * nw]) + 0.5  # [B, V, HxW]

    if margin != 0:
        marginw = 1 - 2 * margin
        i = marginw * i + margin * W
        j = marginw * j + margin * H

    n_flip = torch.sum(flip_flag).item() if flip_flag is not None else 0
    if n_flip > 0 and flip_flag is not None:
        j_flip, i_flip = custom_meshgrid(
            torch.linspace(0, H - 1, nh, device=device, dtype=dtype),
            torch.linspace(W - 1, 0, nw, device=device, dtype=dtype),
        )
        i_flip = i_flip.reshape([1, 1, nh * nw]).expand(B, 1, nh * nw) + 0.5
        j_flip = j_flip.reshape([1, 1, nh * nw]).expand(B, 1, nh * nw) + 0.5
        i[:, flip_flag, ...] = i_flip
        j[:, flip_flag, ...] = j_flip
    return i, j


def get_rays_from_uvs(
    i: torch.Tensor, j: torch.Tensor, K: torch.Tensor, c2w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes ray origins and directions from UV coordinates and camera parameters.

    Parameters:
    - i: U coordinates tensor.
    - j: V coordinates tensor.
    - K: Camera intrinsics (fx, fy, cx, cy).
    - c2w: Camera-to-world transformation matrices.

    Returns:
    - Tuple of (ray origins, ray directions).
    """
    fx, fy, cx, cy = K.chunk(4, dim=-1)  # B,V, 1

    zs = torch.ones_like(i)  # [B, V, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)  # B, V, HW, 3

    # printarr(directions, c2w)
    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, HW, 3
    rays_o = c2w[..., :3, 3]  # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, HW, 3
    return rays_o, rays_d


def get_rays(
    K: torch.Tensor,
    c2w: torch.Tensor,
    H: int,
    W: int,
    device: str | torch.device,
    flip_flag: torch.Tensor | None = None,
    nh: int | None = None,
    nw: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates camera rays for given intrinsics and poses.

    Parameters:
    - K: Camera intrinsics tensor.
    - c2w: Camera-to-world transformation matrices.
    - H: Image height.
    - W: Image width.
    - device: Torch device.
    - flip_flag: Optional flip indicators.
    - nh: Number of height samples.
    - nw: Number of width samples.

    Returns:
    - Tuple of (ray origins, ray directions).
    """
    batch_shape = (K.shape[0], K.shape[1])
    i, j = get_grid_uvs(batch_shape, H=H, W=W, dtype=K.dtype, device=device, flip_flag=flip_flag, nh=nh, nw=nw)
    # printarr(i,j,K, c2w)
    return get_rays_from_uvs(i, j, K, c2w)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        device=timesteps.device
    )
    args = timesteps[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    return embedding


def get_time_embeddings(image_shape: tuple[int, int, int, int, int]):
    """
    Generates duration time embeddings for given duration.
    returns:
    - time_embeddings: [B, V, 2, H, W]
    - target_time_embeddings: [B, 1, 2, H, W]
    """
    B, V, _, H, W = image_shape
    timesteps = torch.linspace(0, 1, V)
    time_embeddings = timestep_embedding(timesteps, 2)
    time_embeddings = time_embeddings[None, :, :, None, None].repeat(B, 1, 1, H, W)
    target_time_embeddings = time_embeddings[:, :1]  # dummy target time embeddings at frame 0
    return time_embeddings.cuda(), target_time_embeddings.cuda()


def calculate_view_indices(num_input_views: int, num_generated_views: int) -> list[int]:
    """
    Calculate which views to select from generated views for input.
    Applies a 90-degree rotation hack for 32 views.

    Parameters:
    - num_input_views: Number of views to select.
    - num_generated_views: Total number of generated views.

    Returns:
    - List of indices to select from generated views.
    """
    return [
        (int(x / num_input_views * num_generated_views) - num_generated_views // 4) % num_generated_views
        for x in range(num_input_views)
    ]
