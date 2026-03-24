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

from __future__ import annotations

import copy
import json
import os
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw

from .models import (
    ModelInput,
    ModelInputDecoder,
    ModelInputEncoder,
    ModelSupervision,
    model_registry,
)
from .options import config_defaults
from .utils import orbit_camera
from .utils.data import ray_condition

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_INPUT_VIEWS = 16


def _intrinsics_from_fov(fov_deg: float, H: int, W: int) -> np.ndarray:
    """Return [fx, fy, cx, cy] for a pinhole camera given FOV (degrees) and resolution."""
    fov_rad = np.deg2rad(float(fov_deg))
    half_tan = np.tan(fov_rad / 2.0)
    fx = (W / 2.0) / half_tan
    fy = (H / 2.0) / half_tan
    cx = W / 2.0
    cy = H / 2.0
    return np.array([fx, fy, cx, cy], dtype=np.float32)


def get_orbit_c2w_relatives(
    dist: float,
    num_views: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Return c2w matrices [num_views, 4, 4] for orbit cameras (objaverse-style).
    Used when loading from disk to build a minimal data_dict for TTT.
    """
    azimuths = [i * 360.0 / num_views for i in range(num_views)]
    elevation = 0.0
    c2w_list = []
    for azi in azimuths:
        cam = orbit_camera(elevation, azi, radius=dist, opengl=True)
        cam = torch.from_numpy(cam).float()
        cam[:3, 1:3] *= -1  # OpenGL to colmap for gs renderer
        c2w_list.append(cam)
    c2w = torch.stack(c2w_list, dim=0)  # [V, 4, 4]
    ref = torch.tensor(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, -dist],
            [0, 0, 0, 1],
        ],
        dtype=c2w.dtype,
        device=c2w.device,
    )
    ref = ref @ torch.inverse(c2w[0]).unsqueeze(0)
    c2w = ref @ c2w  # [V, 4, 4]
    return c2w.to(device)


def _load_camera_metadata(input_dir: str) -> dict:
    """Load camera metadata from input_dir/camera.json."""
    camera_path = os.path.join(input_dir, "camera.json")
    if not os.path.isfile(camera_path):
        raise FileNotFoundError(f"Missing camera.json in {input_dir}")
    with open(camera_path) as f:
        return json.load(f)


def load_cond_views_from_input_views(
    input_views_dir: str,
    img_size: int = 512,
    bbox_size: float = 1.0,
):
    """
    Load conditioning view images and camera data from a sample's input_views folder.
    Does not use any variables from the generation stage.

    Expects camera.json with frame_filenames, mask_filenames, normalized_cam_positions,
    cam_dists, cam_fovs, object_lwh.

    Returns:
        cond_x_wb: [num_cond, 3, H, W] tensor in [-1, 1], or None if no cond views.
        c2w_cond: (num_cond, 4, 4) camera poses, or None.
        fovs_cond: (num_cond,) FOVs, or None.
        dists_cond: (num_cond,) distances, or None.
        lwh: (3,) from camera.json, or None.
    """
    if not os.path.isdir(input_views_dir):
        return None, None, None, None, None
    cam_data = _load_camera_metadata(input_views_dir)
    frame_filenames = cam_data.get("frame_filenames", [])
    mask_filenames = cam_data.get("mask_filenames", [])
    cond_paths = [os.path.join(input_views_dir, f) for f in frame_filenames]
    mask_paths = [os.path.join(input_views_dir, f) for f in mask_filenames]
    if not cond_paths:
        return None, None, None, None, None
    if len(mask_paths) != len(cond_paths):
        raise ValueError(
            f"Expected one mask per conditioning view in {input_views_dir}, got "
            f"{len(mask_paths)} masks for {len(cond_paths)} images"
        )
    cam_poses_raw = cam_data["normalized_cam_positions"]
    dists_list = cam_data["cam_dists"]
    fov_list = cam_data["cam_fovs"]
    lwh_list = cam_data["object_lwh"]
    num_cond = len(cond_paths)
    cam_poses = np.array(cam_poses_raw, dtype=np.float64).reshape(num_cond, 3)
    dists = np.array(dists_list, dtype=np.float64).reshape(num_cond)
    fovs = np.array(fov_list, dtype=np.float64).reshape(num_cond)
    lwh = np.asarray(lwh_list, dtype=np.float64).reshape(3)

    to_tensor = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    cond_tensors = []
    for i, p in enumerate(cond_paths):
        img = Image.open(p).convert("RGB")
        mask_path = mask_paths[i] if i < len(mask_paths) else ""
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(f"Missing conditioning-view mask: {mask_path}")
        mask = Image.open(mask_path)
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=mask)
        img = background
        cond_tensors.append(to_tensor(img))
    cond_x_wb = torch.stack(cond_tensors, dim=0)

    c2w_cond = []
    for i in range(num_cond):
        x = cam_poses[i][1] * -1
        y = cam_poses[i][2]
        z = cam_poses[i][0]
        azimuth = np.arctan2(z, x) * 180 / np.pi + 90
        elevation = np.arcsin(y) * 180 / np.pi * -1
        cam_pose = orbit_camera(elevation, azimuth, radius=dists[i] / max(lwh) * bbox_size, opengl=True)
        cam_pose[:3, 1:3] *= -1
        R_world_x = np.array(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ]
        )
        cam_pose = R_world_x @ cam_pose
        c2w_cond.append(cam_pose)
    c2w_cond = np.array(c2w_cond)
    fovs_cond = np.array(fovs, dtype=np.float32)
    dists_cond = np.array(dists, dtype=np.float32)
    return cond_x_wb, c2w_cond, fovs_cond, dists_cond, lwh


def build_ttt_data_dict(
    images_np: list,
    fov: float,
    dist: float,
    lwh,
    cond_x_wb: torch.Tensor,
    c2w_cond: np.ndarray,
    fovs_cond: np.ndarray,
    dists_cond: np.ndarray,
    lwh_cond,
    img_size: int,
    bbox_size: float = 1.0,
    device: str = "cpu",
) -> SimpleNamespace:
    """
    Build the TTT data_dict from 16 generated images and loaded cond views.
    Used by run_inference to pass into TokengsLiftingRunner.run_lifting(..., data_dict=...).
    """
    to_tensor = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    input_tensors = [to_tensor(Image.fromarray(im)) for im in images_np[:16]]
    x_wb = torch.cat([torch.stack(input_tensors, dim=0), cond_x_wb], dim=0)
    lwh_arr = np.atleast_1d(lwh).astype(np.float64)
    dist_norm = (dist / max(lwh_arr) * bbox_size) if max(lwh_arr) > 0 else (dist * bbox_size)
    c2w_16 = get_orbit_c2w_relatives(dist_norm, 16, device=device)
    c2w = torch.cat([c2w_16, torch.from_numpy(c2w_cond).float()], dim=0)
    fovs = torch.cat(
        [
            torch.full((16,), fov, dtype=torch.float32),
            torch.from_numpy(fovs_cond.astype(np.float32)),
        ],
        dim=0,
    )
    dists = torch.cat(
        [
            torch.full((16,), dist, dtype=torch.float32),
            torch.from_numpy(dists_cond.astype(np.float32)),
        ],
        dim=0,
    )
    lwh_np = np.array(
        lwh_cond if lwh_cond is not None else lwh,
        dtype=np.float32,
    )
    return SimpleNamespace(
        n_target=16,
        x_white_background=x_wb,
        c2w_relatives=c2w,
        fovs=fovs,
        dists=dists,
        lwh=lwh_np,
    )


def save_cond_view_comparisons(
    output_dir: str,
    pred_cond: torch.Tensor,
    gt_cond: torch.Tensor,
) -> None:
    """
    Save GT vs rendered cond view images and side-by-side comparisons to
    output_dir/cond_views_comparison/.
    """
    num_cond = pred_cond.shape[0]
    comp_dir = os.path.join(output_dir, "cond_views_comparison")
    os.makedirs(comp_dir, exist_ok=True)
    for i in range(num_cond):
        gt_np = (gt_cond[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pred_np = (pred_cond[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(gt_np).save(os.path.join(comp_dir, f"condview_{i}_gt.png"))
        Image.fromarray(pred_np).save(os.path.join(comp_dir, f"condview_{i}_pred.png"))
        comparison = np.concatenate([gt_np, pred_np], axis=1)
        comp_pil = Image.fromarray(comparison)
        draw = ImageDraw.Draw(comp_pil)
        draw.text((10, 10), "GT", fill=(255, 255, 255))
        draw.text((gt_np.shape[1] + 10, 10), "Rendered", fill=(255, 255, 255))
        comp_pil.save(os.path.join(comp_dir, f"condview_{i}_comparison.png"))
    print(f"   Saved {num_cond} cond view comparisons to {comp_dir}")


def build_cameras_and_rays(
    dist: float,
    fov: float,
    num_views: int,
    res: int,
    device: str = "cuda",
    dtype: torch.dtype | None = None,
):
    """
    Build orbit cameras (c2w), intrinsics, and ray embeddings (Plucker, rays_os, rays_ds)
    in objaverse-style normalized space. All tensors on the given device with batch dim 1.

    Returns:
        c2w: [1, V, 4, 4] float
        intrinsics: [1, V, 4] float
        plucker: [1, V, 6, H, W]
        rays_os: [1, V, 3, H, W]
        rays_ds: [1, V, 3, H, W]
    """
    if dtype is None:
        dtype = torch.float32
    azimuths = [i * 360.0 / num_views for i in range(num_views)]
    elevation = 0.0

    c2w_list = []
    for azi in azimuths:
        cam = orbit_camera(elevation, azi, radius=dist, opengl=True)
        cam = torch.from_numpy(cam).float()
        cam[:3, 1:3] *= -1  # OpenGL to colmap for gs renderer
        c2w_list.append(cam)
    c2w = torch.stack(c2w_list, dim=0)  # [V, 4, 4]

    # Objaverse-style camera normalization
    ref = torch.tensor(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, -dist],
            [0, 0, 0, 1],
        ],
        dtype=torch.float32,
        device=c2w.device,
    )
    ref = ref @ torch.inverse(c2w[0]).unsqueeze(0)
    c2w = ref @ c2w  # [V, 4, 4]

    K_np = _intrinsics_from_fov(fov, res, res)
    intrinsics = torch.from_numpy(K_np).float().unsqueeze(0).expand(num_views, -1)  # [V, 4]

    c2w = c2w.unsqueeze(0).to(device=device, dtype=torch.float32)  # [1, V, 4, 4]
    intrinsics = intrinsics.unsqueeze(0).to(device=device, dtype=torch.float32)  # [1, V, 4]

    plucker, rays_os, rays_ds = ray_condition(
        intrinsics.cpu().float(), c2w.cpu().float(), res, res, device="cpu", flip_flag=None
    )
    # ray_condition returns [V, C, H, W] after [0]; add batch dim
    plucker = plucker.unsqueeze(0).to(device=device, dtype=dtype)
    rays_os = rays_os.unsqueeze(0).to(device=device, dtype=dtype)
    rays_ds = rays_ds.unsqueeze(0).to(device=device, dtype=dtype)

    return c2w, intrinsics, plucker, rays_os, rays_ds


def build_model_input_from_mv_images(
    mv_images_np: list,
    fov: float,
    dist: float,
    lwh: list | tuple | np.ndarray,
    img_size: int = 512,
    device: str = "cuda",
    num_views: int = NUM_INPUT_VIEWS,
    bbox_size: float = 1.0,
    dtype: torch.dtype | None = None,
):
    """
    Build TokenGS ModelInput from a list of multi-view RGB images (numpy [H,W,3] uint8)
    and camera parameters. Uses orbit cameras and objaverse normalization.

    Args:
        mv_images_np: List of V numpy images [H, W, 3] in [0, 255].
        fov: Field of view in degrees.
        dist: Camera distance (will be normalized by max(lwh) and scaled by bbox_size).
        lwh: (length, width, height) for normalization.
        img_size: Target resolution (default 512).
        device: Target device.
        num_views: Number of views (default 16).
        bbox_size: Bounding box size scaling factor (default 1.0).
        dtype: Tensor dtype for model input (default float32).

    Returns:
        model_input: ModelInput with encoder (16 views) and decoder (16 views for rendering).
    """
    if dtype is None:
        dtype = torch.float32
    lwh = np.atleast_1d(lwh).astype(np.float64)
    dist_norm = (dist / max(lwh)) * bbox_size if max(lwh) > 0 else dist * bbox_size

    # Use first num_views images
    images = [mv_images_np[i] for i in range(min(num_views, len(mv_images_np)))]
    if len(images) < num_views:
        raise ValueError(f"Need at least {num_views} views, got {len(images)}")

    c2w, intrinsics, plucker, rays_os, rays_ds = build_cameras_and_rays(
        dist_norm, fov, num_views, img_size, device=device, dtype=dtype
    )

    # Stack and preprocess images: [V, 3, H, W], float [0,1] then ImageNet normalize
    rgb = np.stack(images[:num_views], axis=0).astype(np.float32) / 255.0
    rgb = torch.from_numpy(rgb).permute(0, 3, 1, 2)  # [V, 3, H, W]
    if rgb.shape[2] != img_size or rgb.shape[3] != img_size:
        rgb = F.interpolate(rgb, size=(img_size, img_size), mode="bilinear", align_corners=False)
    rgb = TF.normalize(rgb, IMAGENET_MEAN, IMAGENET_STD)
    rgb = rgb.unsqueeze(0).to(device=device, dtype=dtype)  # [1, V, 3, H, W]

    encoder = ModelInputEncoder(
        images_rgb=rgb,
        plucker=plucker,
        rays_os=rays_os,
        rays_ds=rays_ds,
        intrinsics_input=intrinsics,
        cam_to_world_input=c2w,
        time_embedding_input=None,
        time_embedding_target=None,
        images_rgb_unnormalized=None,
    )

    # Decoder for rendering: same 16 orbit views (inverse in float32 for bf16/fp16)
    cam_view = torch.inverse(c2w.float()).to(dtype=c2w.dtype).transpose(-2, -1)  # [1, V, 4, 4]
    decoder = ModelInputDecoder(
        time_embedding_target=None,
        cam_view=cam_view,
        intrinsics=intrinsics,
    )

    return ModelInput(encoder=encoder, decoder=decoder)


def build_model_input_and_supervision_with_cond_views(
    mv_images_np: list,
    data_dict,
    img_size: int = 512,
    device: str = "cuda",
    num_input_views: int = NUM_INPUT_VIEWS,
    ttt_supervision_mode: str = "both",
    bbox_size: float = 1.0,
    dtype: torch.dtype | None = None,
):
    """
    Build ModelInput and (optionally) ModelSupervision when conditioning views from
    data_dict are available for TTT. Uses data_dict cameras for both the 16 generated
    views and the cond views so the coordinate system is consistent.

    Args:
        mv_images_np: List of 16 generated view images [H,W,3] uint8.
        data_dict: AttrDict from preproc (with n_target, x_white_background, c2w_relatives,
                   fovs, dists; optional x_msk). After run_inference trim, cond views
                   are data_dict.x_white_background[data_dict.n_target:], etc.
        img_size: Resolution (512).
        device: Target device.
        num_input_views: 16.
        ttt_supervision_mode: "input_only", "cond_only", or "both".
        bbox_size: Bounding box size scaling factor (default 1.0).

    Returns:
        model_input: ModelInput (encoder = 16 generated views; decoder includes cond view
                     cameras when cond_only or both).
        supervision: ModelSupervision for TTT, or None if not doing TTT.
    """
    n_target = data_dict.n_target
    # Cond view data (slices; data_dict is already trimmed in run_inference)
    cond_x = data_dict.x_white_background[n_target:]  # [num_cond, 3, H, W] in [-1, 1]
    cond_c2w = data_dict.c2w_relatives[n_target:]  # [num_cond, 4, 4]
    cond_fovs = data_dict.fovs[n_target:]
    num_cond = cond_x.shape[0]
    if dtype is None:
        dtype = torch.float32
    if num_cond == 0:
        return build_model_input_from_mv_images(
            mv_images_np,
            data_dict.fovs[0].item(),
            data_dict.dists[0].item(),
            getattr(data_dict, "lwh", np.array([1.0, 1.0, 1.0])),
            img_size=img_size,
            device=device,
            num_views=num_input_views,
            bbox_size=bbox_size,
            dtype=dtype,
        ), None

    # Intrinsics for cond views from FOV (robust to data_dict.intrinsics not being trimmed)
    cond_K = []
    for i in range(num_cond):
        fov = cond_fovs[i].item() if torch.is_tensor(cond_fovs[i]) else float(cond_fovs[i])
        K = _intrinsics_from_fov(fov, img_size, img_size)
        cond_K.append(K)
    cond_intrinsics = torch.from_numpy(np.stack(cond_K)).to(device=device, dtype=dtype)  # [num_cond, 4]

    # 16 input views: use same build_cameras_and_rays as no-TTT so encoder input is identical
    # (avoids any mismatch between get_orbit_c2w_relatives and build_cameras_and_rays)
    lwh_arr = np.atleast_1d(getattr(data_dict, "lwh", np.array([1.0, 1.0, 1.0]))).astype(np.float64)
    dist_0 = data_dict.dists[0].item() if torch.is_tensor(data_dict.dists[0]) else float(data_dict.dists[0])
    dist_norm = (dist_0 / max(lwh_arr) * bbox_size) if max(lwh_arr) > 0 else (dist_0 * bbox_size)
    fov_0 = data_dict.fovs[0].item() if torch.is_tensor(data_dict.fovs[0]) else float(data_dict.fovs[0])
    c2w_input, intrinsics_input, plucker, rays_os, rays_ds = build_cameras_and_rays(
        dist_norm, fov_0, num_input_views, img_size, device=device, dtype=dtype
    )
    # build_cameras_and_rays returns c2w [1, V, 4, 4]; encoder expects cam_to_world_input same shape
    c2w_input = c2w_input.squeeze(0)  # [V, 4, 4] for encoder cam_to_world_input.unsqueeze(0) below

    # RGB for 16 input views: from mv_images_np, ImageNet normalized
    rgb_np = np.stack([mv_images_np[i] for i in range(num_input_views)], axis=0).astype(np.float32) / 255.0
    rgb = torch.from_numpy(rgb_np).permute(0, 3, 1, 2).to(device=device, dtype=dtype)
    if rgb.shape[2] != img_size or rgb.shape[3] != img_size:
        rgb = F.interpolate(rgb, size=(img_size, img_size), mode="bilinear", align_corners=False)
    rgb = TF.normalize(rgb, IMAGENET_MEAN, IMAGENET_STD)
    rgb = rgb.unsqueeze(0)  # [1, 16, 3, H, W]
    rgb_unnorm = rgb_np.transpose(0, 3, 1, 2) if rgb_np.shape[-1] == 3 else rgb_np
    rgb_unnorm = torch.from_numpy(rgb_unnorm).to(device=device, dtype=dtype).unsqueeze(0)  # [1, 16, 3, H, W]

    encoder = ModelInputEncoder(
        images_rgb=rgb,
        plucker=plucker,
        rays_os=rays_os,
        rays_ds=rays_ds,
        intrinsics_input=intrinsics_input,
        cam_to_world_input=c2w_input.unsqueeze(0),
        time_embedding_input=None,
        time_embedding_target=None,
        images_rgb_unnormalized=rgb_unnorm,
    )

    # Decoder: cond view cameras (and optionally 16 orbit for "both")
    cond_c2w = cond_c2w.to(device=device, dtype=dtype)
    # Ensure cond_c2w is [num_cond, 4, 4]
    if cond_c2w.dim() == 2:
        cond_c2w = cond_c2w.unsqueeze(0)  # [4, 4] -> [1, 4, 4]
    elif cond_c2w.dim() == 4:
        cond_c2w = cond_c2w.squeeze(0)  # [1, num_cond, 4, 4] -> [num_cond, 4, 4]

    cond_cam_view = torch.inverse(cond_c2w.float()).to(dtype=cond_c2w.dtype).transpose(-2, -1)
    if cond_cam_view.dim() == 3:
        cond_cam_view = cond_cam_view.unsqueeze(0)  # [num_cond, 4, 4] -> [1, num_cond, 4, 4]
    cond_intrinsics = cond_intrinsics.unsqueeze(0)  # [1, num_cond, 4]

    if ttt_supervision_mode == "cond_only":
        decoder = ModelInputDecoder(
            time_embedding_target=None,
            cam_view=cond_cam_view,
            intrinsics=cond_intrinsics,
        )
        # Supervision: cond view images in [0, 1]
        cond_images_01 = (cond_x.to(device=device, dtype=dtype) + 1.0) / 2.0
        cond_images_01 = cond_images_01.unsqueeze(0)
        has_mask = False
        cond_masks = torch.ones(1, num_cond, 1, img_size, img_size, device=device, dtype=dtype)
        supervision = ModelSupervision(
            images_output=cond_images_01,
            masks_output=cond_masks,
            has_mask=torch.tensor([has_mask], device=device, dtype=torch.bool),
            rays_os=None,
            rays_ds=None,
        )
        return ModelInput(encoder=encoder, decoder=decoder), supervision

    # both: decoder and supervision contain only cond views; forward_ttt_tokens will merge
    # with the 16 input views from to_ttt() to get 16 + num_cond target views.
    decoder = ModelInputDecoder(
        time_embedding_target=None,
        cam_view=cond_cam_view,
        intrinsics=cond_intrinsics,
    )

    # Supervision: cond views only (input view supervision is built inside forward_ttt_tokens via to_ttt)
    cond_images_01 = (cond_x.to(device=device, dtype=dtype) + 1.0) / 2.0
    cond_images_01 = cond_images_01.unsqueeze(0)
    masks_output = torch.ones(1, num_cond, 1, img_size, img_size, device=device, dtype=dtype)
    has_mask = False
    supervision = ModelSupervision(
        images_output=cond_images_01,
        masks_output=masks_output,
        has_mask=torch.tensor([has_mask], device=device, dtype=torch.bool),
        rays_os=None,
        rays_ds=None,
    )
    return ModelInput(encoder=encoder, decoder=decoder), supervision


def _parse_metadata_int(raw) -> int | None:
    """Parse an int from safetensors metadata (values are often strings)."""
    if raw is None:
        return None
    if isinstance(raw, (int, np.integer)):
        return int(raw)
    s = str(raw).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        j = json.loads(s)
        if isinstance(j, int):
            return j
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def read_tokengs_safetensors_config(ckpt_path: str) -> dict:
    """
    Read lifting config from a .safetensors file header metadata.

    Supported keys (string or int values):
      - input_res: square input resolution for lifting
      - num_gs_tokens or num_gs_token: number of Gaussian tokens

    Returns:
        Dict with optional keys ``input_res`` and ``num_gs_tokens`` (ints).
    """
    result: dict = {}
    if not ckpt_path or not isinstance(ckpt_path, str) or not ckpt_path.endswith(".safetensors"):
        return result
    if not os.path.isfile(ckpt_path):
        return result
    try:
        from safetensors import safe_open

        with safe_open(ckpt_path, framework="pt", device="cpu") as f:
            meta = f.metadata()
        if not meta:
            return result
        ir = meta.get("input_res")
        if ir is not None:
            v = _parse_metadata_int(ir)
            if v is not None and v > 0:
                result["input_res"] = v
        ntok = meta.get("num_gs_tokens")
        if ntok is None:
            ntok = meta.get("num_gs_token")
        if ntok is not None:
            v = _parse_metadata_int(ntok)
            if v is not None and v > 0:
                result["num_gs_tokens"] = v
    except OSError as e:
        print(f"TokenGS: could not read safetensors metadata from {ckpt_path}: {e}")
    except Exception as e:
        print(f"TokenGS: could not parse safetensors metadata from {ckpt_path}: {e}")
    return result


def load_tokengs_model(
    ckpt_path: str,
    device: str = "cuda",
    dtype: torch.dtype | None = None,
    **opt_overrides,
):
    """
    Load TokenGS model with tokengs subcommand defaults and optional overrides.
    Tolerantly loads checkpoint (matching keys/shapes only).

    For ``.safetensors`` checkpoints, header metadata can set ``input_res`` and
    ``num_gs_tokens`` (or ``num_gs_token``); when present, these override defaults
    and any matching ``opt_overrides`` before the model is built.

    Returns:
        model: TokenGS on device, eval mode.
        opt: Options used.
    """
    if dtype is None:
        dtype = torch.float32
    opt = copy.deepcopy(config_defaults["tokengs"])
    opt.resume = ckpt_path
    opt.evaluating = True
    opt.batch_size = 1
    for k, v in opt_overrides.items():
        if hasattr(opt, k):
            setattr(opt, k, v)

    meta_cfg = read_tokengs_safetensors_config(ckpt_path) if ckpt_path and ckpt_path != "None" else {}
    if "input_res" in meta_cfg:
        r = meta_cfg["input_res"]
        opt.img_size = (r, r)
    if "num_gs_tokens" in meta_cfg:
        opt.num_gs_tokens = meta_cfg["num_gs_tokens"]
    if meta_cfg:
        print(
            f"TokenGS safetensors metadata applied: {meta_cfg} -> img_size={opt.img_size}, num_gs_tokens={opt.num_gs_tokens}"
        )

    model = model_registry["tokengs"](opt)
    if ckpt_path and ckpt_path != "None":
        if ckpt_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            ckpt = load_file(ckpt_path, device="cpu")
        else:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
        state_dict = model.state_dict()
        for k, v in ckpt.items():
            if k in state_dict and state_dict[k].shape == v.shape:
                state_dict[k].copy_(v)
        print("TokenGS model loaded from", ckpt_path)
    model = model.to(device=device, dtype=dtype).eval()
    return model, opt


class TokengsLiftingRunner:
    """
    Drop-in replacement for LGMTester: load TokenGS and run Gaussian lifting from
    multi-view images, with optional TTT using cond views from data_dict.
    """

    def __init__(
        self,
        ckpt_path: str,
        use_ttt: bool = False,
        ttt_n_steps: int = 50,
        ttt_lr: float = 1e-3,
        ttt_supervision_mode: str = "both",
        device: str = "cuda",
        bbox_size: float = 1.0,
        dtype: torch.dtype = torch.float16,
        render_img_size: int | None = None,
    ):
        self.device = device
        self.dtype = dtype
        self.use_ttt = use_ttt
        self.ttt_n_steps = ttt_n_steps
        self.ttt_lr = ttt_lr
        self.ttt_supervision_mode = ttt_supervision_mode
        self.bbox_size = float(bbox_size)
        opt_overrides = dict(
            use_ttt_for_eval=use_ttt,
            ttt_n_steps=ttt_n_steps,
            ttt_lr=ttt_lr,
            ttt_supervision_mode=ttt_supervision_mode,
        )
        self.model, self.opt = load_tokengs_model(
            ckpt_path,
            device=device,
            dtype=dtype,
            **opt_overrides,
        )
        self.img_size = self.opt.img_size if isinstance(self.opt.img_size, (list, tuple)) else (self.opt.img_size,) * 2
        self.img_size = self.img_size[0]
        self.render_img_size = render_img_size if render_img_size is not None else self.img_size

    def run_lifting(
        self,
        mv_images_np: list,
        fov: float,
        dist: float,
        lwh,
        data_dict=None,
    ):
        """
        Run Gaussian reconstruction from multi-view images. Optionally run TTT when
        data_dict is provided and use_ttt is True (cond views used as supervision).

        Args:
            mv_images_np: List of numpy [H,W,3] uint8 images (16 generated views).
            fov: Field of view in degrees.
            dist: Camera distance.
            lwh: (length, width, height) for normalization.
            data_dict: Optional AttrDict from preproc (with n_target, x_white_background,
                       c2w_relatives, fovs, dists) for TTT cond views.

        Returns:
            gaussians: Tensor [1, N, 14]. Call render_orbit_views(gaussians, fov, dist, lwh) to visualize.
        """
        self.model.eval()
        lwh_np = np.atleast_1d(lwh).astype(np.float64) if lwh is not None else np.array([1.0, 1.0, 1.0])
        if self.use_ttt and data_dict is not None:
            model_input, supervision = build_model_input_and_supervision_with_cond_views(
                mv_images_np,
                data_dict,
                img_size=self.img_size,
                device=self.device,
                num_input_views=NUM_INPUT_VIEWS,
                ttt_supervision_mode=self.ttt_supervision_mode,
                bbox_size=self.bbox_size,
                dtype=self.dtype,
            )
            if supervision is not None:
                gaussians = self.model.forward_ttt(
                    model_input,
                    supervision,
                    n_steps=self.ttt_n_steps,
                    lr=self.ttt_lr,
                    method="tokens",
                )
            else:
                gaussians = self.model.forward_gaussians(model_input)
        else:
            model_input = build_model_input_from_mv_images(
                mv_images_np,
                fov,
                dist,
                lwh_np,
                img_size=self.img_size,
                device=self.device,
                num_views=NUM_INPUT_VIEWS,
                bbox_size=self.bbox_size,
                dtype=self.dtype,
            )
            gaussians = self.model.forward_gaussians(model_input)

        return gaussians

    def render_orbit_views(
        self,
        gaussians: torch.Tensor,
        fov: float,
        dist: float,
        lwh,
    ) -> torch.Tensor:
        """
        Render Gaussians to 16 orbit view images. Call after run_lifting with the same
        fov, dist, and lwh used for lifting (orbit cameras match the lifting setup).

        Args:
            gaussians: [1, N, 14] from run_lifting.
            fov: Field of view in degrees.
            dist: Camera distance.
            lwh: (length, width, height) for distance normalization (same as run_lifting).

        Returns:
            Rendered images [V, 3, H, W] uint8 on CPU.
        """
        lwh_np = np.atleast_1d(lwh).astype(np.float64) if lwh is not None else np.array([1.0, 1.0, 1.0])
        dist_norm = (dist / max(lwh_np)) * self.bbox_size if max(lwh_np) > 0 else (dist * self.bbox_size)
        c2w, intrinsics, _, _, _ = build_cameras_and_rays(
            dist_norm, fov, 80, self.render_img_size, device=self.device, dtype=self.dtype
        )
        cam_view = torch.inverse(c2w.float()).to(dtype=c2w.dtype).transpose(-2, -1)
        dec = ModelInputDecoder(time_embedding_target=None, cam_view=cam_view, intrinsics=intrinsics)
        with torch.no_grad():
            out = self.model.render_gaussians(gaussians, dec, output_size=(self.render_img_size, self.render_img_size))
        pred = out["images_pred"]  # [1, V, 3, H, W]
        pred = pred.squeeze(0).float().clamp(0, 1)
        rendered = (pred * 255).clamp(0, 255).to(torch.uint8).cpu()
        return rendered

    def render_at_cond_views(
        self,
        gaussians: torch.Tensor,
        data_dict,
    ):
        """
        Render the reconstructed Gaussians from conditioning view camera poses.
        Used after TTT to compare rendered vs GT cond views.

        Args:
            gaussians: [1, N, 14] from run_lifting.
            data_dict: SimpleNamespace with n_target, c2w_relatives, fovs, x_white_background, lwh.

        Returns:
            pred: [num_cond, 3, H, W] float [0, 1] on CPU, or None if no cond views.
            gt: [num_cond, 3, H, W] float [0, 1] on CPU, or None.
        """
        n_target = data_dict.n_target
        if not hasattr(data_dict, "c2w_relatives") or data_dict.c2w_relatives is None:
            return None, None
        if data_dict.c2w_relatives.shape[0] <= n_target:
            return None, None
        cond_c2w = data_dict.c2w_relatives[n_target:].to(self.device)
        cond_fovs = data_dict.fovs[n_target:]
        num_cond = cond_c2w.shape[0]
        img_size = self.img_size

        # Same scaling as in build_model_input_and_supervision_with_cond_views
        cond_K = []
        for i in range(num_cond):
            fov = cond_fovs[i].item() if torch.is_tensor(cond_fovs[i]) else float(cond_fovs[i])
            K = _intrinsics_from_fov(fov, img_size, img_size)
            cond_K.append(K)
        cond_intrinsics = torch.from_numpy(np.stack(cond_K)).float().to(self.device).unsqueeze(0)
        cond_cam_view = torch.inverse(cond_c2w.float()).to(dtype=cond_c2w.dtype).transpose(-2, -1).unsqueeze(0)
        dec = ModelInputDecoder(
            time_embedding_target=None,
            cam_view=cond_cam_view,
            intrinsics=cond_intrinsics,
        )
        with torch.no_grad():
            out = self.model.render_gaussians(gaussians, dec)
        pred = out["images_pred"].squeeze(0).float().clamp(0, 1).cpu()

        gt = data_dict.x_white_background[n_target:].float()
        if gt.is_cuda:
            gt = gt.cpu()
        gt = (gt + 1.0) / 2.0
        if gt.shape[-2] != img_size or gt.shape[-1] != img_size:
            gt = F.interpolate(
                gt,
                size=(img_size, img_size),
                mode="bilinear",
                align_corners=False,
            )
        gt = gt[:, :3].clamp(0, 1)
        return pred, gt

    def save_ply(self, gaussians: torch.Tensor, path: str, compatible: bool = True):
        """
        Save Gaussians to a PLY file (same API as run_inference.save_ply).
        Restores original asset scale by scaling back xyz positions and scales by 1/bbox_size
        so the exported asset is invariant to the bbox_size scaling factor.
        """
        # Clone to avoid modifying the original gaussians tensor
        gaussians_to_save = gaussians.clone()

        # Restore original scale: scale back xyz positions and scales by 1/bbox_size
        if self.bbox_size != 1.0:
            # Scale positions (xyz) and scales back to original
            gaussians_to_save[:, :, 0:3] = gaussians_to_save[:, :, 0:3] / self.bbox_size  # xyz positions
            gaussians_to_save[:, :, 4:7] = gaussians_to_save[:, :, 4:7] / self.bbox_size  # scaling

        self.model.gs.save_ply(gaussians_to_save, path, compatible=compatible)
