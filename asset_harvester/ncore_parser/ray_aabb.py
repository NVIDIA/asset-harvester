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

"""PyTorch implementation of ray/AABB intersection."""

from __future__ import annotations

import torch


def _validate_inputs(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    aabbs_min: torch.Tensor,
    aabbs_max: torch.Tensor,
) -> tuple[int, int]:
    if rays_o.ndim != 2 or rays_o.shape[1] != 3:
        raise ValueError(f"rays_o must have shape (N, 3), got {tuple(rays_o.shape)}")
    if rays_d.ndim != 2 or rays_d.shape[1] != 3:
        raise ValueError(f"rays_d must have shape (N, 3), got {tuple(rays_d.shape)}")
    if aabbs_min.ndim != 2 or aabbs_min.shape[1] != 3:
        raise ValueError(f"aabbs_min must have shape (M, 3), got {tuple(aabbs_min.shape)}")
    if aabbs_max.ndim != 2 or aabbs_max.shape[1] != 3:
        raise ValueError(f"aabbs_max must have shape (M, 3), got {tuple(aabbs_max.shape)}")
    if rays_o.shape[0] != rays_d.shape[0]:
        raise ValueError("rays_o and rays_d must have the same number of rays")
    if aabbs_min.shape[0] != aabbs_max.shape[0]:
        raise ValueError("aabbs_min and aabbs_max must have the same number of boxes")
    if rays_o.dtype != rays_d.dtype or rays_o.dtype != aabbs_min.dtype or rays_o.dtype != aabbs_max.dtype:
        raise TypeError("All inputs must have the same dtype")
    if not rays_o.is_floating_point():
        raise TypeError("Inputs must be floating point tensors")
    if rays_o.device != rays_d.device or rays_o.device != aabbs_min.device or rays_o.device != aabbs_max.device:
        raise ValueError("All inputs must be on the same device")
    return rays_o.shape[0], aabbs_min.shape[0]


def ray_aabb_intersect(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    aabbs_min: torch.Tensor,
    aabbs_max: torch.Tensor,
    compute_hits_flag: bool = True,
    compute_hits_t: bool = True,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Compute ray/AABB intersections.

    Args:
        rays_o: (N_rays, 3) ray origins.
        rays_d: (N_rays, 3) ray directions (typically normalized).
        aabbs_min: (N_aabbs, 3) min corners of AABBs.
        aabbs_max: (N_aabbs, 3) max corners of AABBs.
        compute_hits_flag: If True, return per-ray/AABB hit flags.
        compute_hits_t: If True, return per-ray/AABB entry/exit t values.

    Returns:
        (hits_flag, hits_t):
            hits_flag: (N_rays, N_aabbs) bool tensor or None.
            hits_t: (N_rays, N_aabbs, 2) tensor of t1/t2 or None.
    """
    n_rays, n_aabbs = _validate_inputs(rays_o, rays_d, aabbs_min, aabbs_max)

    if n_rays == 0 or n_aabbs == 0:
        hits_flag = None
        hits_t = None
        if compute_hits_flag:
            hits_flag = torch.zeros((n_rays, n_aabbs), dtype=torch.bool, device=rays_o.device)
        if compute_hits_t:
            hits_t = torch.full(
                (n_rays, n_aabbs, 2),
                -1.0,
                dtype=rays_o.dtype,
                device=rays_o.device,
            )
        return hits_flag, hits_t

    rays_o_e = rays_o[:, None, :]
    rays_d_e = rays_d[:, None, :]
    aabbs_min_e = aabbs_min[None, :, :]
    aabbs_max_e = aabbs_max[None, :, :]

    inv_d = torch.reciprocal(rays_d_e)
    t_min = (aabbs_min_e - rays_o_e) * inv_d
    t_max = (aabbs_max_e - rays_o_e) * inv_d

    t1 = torch.max(torch.minimum(t_min, t_max), dim=-1).values
    t2 = torch.min(torch.maximum(t_min, t_max), dim=-1).values

    hits_mask = (t1 <= t2) & (t2 > 0)

    hits_flag = hits_mask if compute_hits_flag else None
    hits_t = None
    if compute_hits_t:
        neg_one = torch.full_like(t1, -1.0)
        t1_out = torch.where(hits_mask, t1, neg_one)
        t2_out = torch.where(hits_mask, t2, neg_one)
        hits_t = torch.stack((t1_out, t2_out), dim=-1)

    return hits_flag, hits_t
