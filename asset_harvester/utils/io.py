# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""I/O helpers for saving multiview diffusion and lifting outputs."""

from __future__ import annotations

import os

import imageio
import numpy as np
from PIL import Image


def save_mvd_outputs(
    images: list[Image.Image],
    fov: float,
    dist: float,
    lwh: list[float] | np.ndarray | None,
    output_dir: str,
) -> list[np.ndarray]:
    """Save multiview diffusion outputs (individual PNGs, video, camera metadata).

    Args:
        images: List of PIL images from pipeline output["images"].
        fov: Field of view (scalar).
        dist: Camera distance (scalar).
        lwh: Object [length, width, height] or None.
        output_dir: Root output directory for this sample.

    Returns:
        images_np: List of numpy uint8 arrays [H, W, 3] (for downstream lifting).
    """
    os.makedirs(output_dir, exist_ok=True)
    recon_dir = os.path.join(output_dir, "multiview")
    os.makedirs(recon_dir, exist_ok=True)

    images_np = [np.array(img) for img in images]

    imageio.v2.mimwrite(os.path.join(output_dir, "multiview.mp4"), images_np, fps=5, macro_block_size=1)

    for i, img in enumerate(images):
        img.save(os.path.join(recon_dir, f"{i}.png"))

    with open(os.path.join(recon_dir, "fov.txt"), "w") as f:
        f.write(str(fov))
    with open(os.path.join(recon_dir, "dist.txt"), "w") as f:
        f.write(str(dist))
    if lwh is not None:
        with open(os.path.join(recon_dir, "lwh.txt"), "w") as f:
            f.write(f"{lwh[0]} {lwh[1]} {lwh[2]}")

    return images_np


def save_input_views(
    cond_images: list[Image.Image],
    mask_images: list[Image.Image],
    output_dir: str,
) -> None:
    """Save conditioning input views and masks.

    Args:
        cond_images: List of PIL RGB conditioning frame images.
        mask_images: List of PIL mask images.
        output_dir: Root output directory for this sample.
    """
    cond_view_dir = os.path.join(output_dir, "input")
    os.makedirs(cond_view_dir, exist_ok=True)

    for i, (cond_image, msk_image) in enumerate(zip(cond_images, mask_images)):
        cond_image.save(os.path.join(cond_view_dir, f"frame_{i}.jpeg"))
        msk_image.save(os.path.join(cond_view_dir, f"mask_{i}.png"))


def save_lifting_outputs(
    rendered_images: list[np.ndarray],
    ply_path: str,
    gaussians,
    lifting_runner,
    output_dir: str,
) -> None:
    """Save Gaussian lifting outputs (rendered views, video, PLY).

    Args:
        rendered_images: List of numpy uint8 arrays [H, W, 3] (orbit renders).
        ply_path: Path to write the Gaussian PLY file.
        gaussians: Gaussian tensor [1, N, 14] from run_lifting.
        lifting_runner: TokengsLiftingRunner instance (for save_ply).
        output_dir: Root output directory for this sample.
    """

    imageio.v2.mimwrite(os.path.join(output_dir, "3d_lifted.mp4"), rendered_images, fps=25, macro_block_size=1)

    lifting_runner.save_ply(gaussians, ply_path)
