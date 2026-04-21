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

import torch


def reflect_extrinsics(
    extrinsics: torch.Tensor,
) -> torch.Tensor:
    reflect = torch.eye(4, dtype=torch.float32, device=extrinsics.device)
    reflect[0, 0] = -1
    return reflect @ extrinsics @ reflect


def random_reflect(
    rgbs: torch.Tensor,
    c2ws: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly augment the training images."""
    # Do not augment with 50% chance.
    if torch.rand(tuple(), generator=generator) < 0.5:
        return rgbs, c2ws

    return rgbs.flip(-1), reflect_extrinsics(c2ws)


def augment_camera_uniform(
    rng: torch.Generator,
    c2w: torch.Tensor,
    intrinsics: torch.Tensor,
    rot_deg_range: tuple[float, float] = (0, 3),
    trans_range: tuple[float, float] = (0, 0.01),
    intrin_range: tuple[float, float] = (0, 0.08),
    probability: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        c2w: [V, 4, 4] tensor of camera-to-world matrices
        intrinsics: [V, 4] tensor of [fx, fy, cx, cy]
        rot_deg_range: rotation angle range in degrees (min, max)
        trans_range: translation distance range (min, max)
        intrin_range: multiplicative noise range on intrinsics (min, max)
        probability: probability of applying the augmentation
    Returns:
        c2w_aug: [V, 4, 4] augmented camera-to-world matrices
        intrin_aug: [V, 4] augmented intrinsics

    rot_deg_range set to (0, 3): https://github.com/facebookresearch/vggt/issues/254
    trans_range set to (0, 0.01)
    intrin_range set to (0, 0.05): https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136610247.pdf
    """
    if torch.rand(tuple(), generator=rng) >= probability:
        return c2w, intrinsics

    V = c2w.shape[0]
    device = c2w.device

    # Generate small random rotations using axis-angle
    axis = torch.randn(V, 3, device=device, generator=rng)
    rot_rad_range = (rot_deg_range[0] * math.pi / 180.0, rot_deg_range[1] * math.pi / 180.0)
    random_sign = torch.randint(0, 2, (V, 1), device=device, generator=rng) * 2.0 - 1.0
    angle = random_sign * (
        torch.rand(V, 1, device=device, generator=rng) * (rot_rad_range[1] - rot_rad_range[0]) + rot_rad_range[0]
    )
    axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)
    theta = angle * axis  # [V, 3]

    def axis_angle_to_matrix(theta):
        """Convert axis-angle to rotation matrix."""
        angle = theta.norm(dim=-1, keepdim=True) + 1e-8
        axis = theta / angle
        x, y, z = axis.unbind(dim=-1)
        ca = torch.cos(angle).squeeze(-1)
        sa = torch.sin(angle).squeeze(-1)
        C = 1 - ca
        rot = torch.stack(
            [
                ca + x * x * C,
                x * y * C - z * sa,
                x * z * C + y * sa,
                y * x * C + z * sa,
                ca + y * y * C,
                y * z * C - x * sa,
                z * x * C - y * sa,
                z * y * C + x * sa,
                ca + z * z * C,
            ],
            dim=-1,
        ).reshape(V, 3, 3)
        return rot

    R_delta = axis_angle_to_matrix(theta)  # [V, 3, 3]
    trans_sign = torch.randint(0, 2, (V, 3), device=device, generator=rng) * 2.0 - 1.0
    t_delta = trans_sign * (
        torch.rand(V, 3, device=device, generator=rng) * (trans_range[1] - trans_range[0]) + trans_range[0]
    )  # [V, 3]

    R_orig = c2w[:, :3, :3]  # [V, 3, 3]
    t_orig = c2w[:, :3, 3]  # [V, 3]

    R_new = torch.bmm(R_delta, R_orig)
    t_new = t_orig + t_delta

    c2w_aug = c2w.clone()
    # not augment the rotation of the first camera
    c2w_aug[1:, :3, :3] = R_new[1:, :3, :3]
    # not augment the position of the first camera
    c2w_aug[1:, :3, 3] = t_new[1:, :3]

    # Intrinsic augmentation: multiplicative noise, only focal, tie fx and ft, tie all frames
    intrin_sign = torch.randint(0, 2, (1,), device=intrinsics.device, generator=rng) * 2.0 - 1.0
    intrin_noise = 1.0 + intrin_sign * (
        torch.rand(1, generator=rng, device=intrinsics.device) * (intrin_range[1] - intrin_range[0]) + intrin_range[0]
    )
    intrin_aug = intrinsics.clone()
    intrin_aug[..., :2] = intrinsics[..., :2] * intrin_noise

    return c2w_aug, intrin_aug
