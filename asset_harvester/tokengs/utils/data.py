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

import numpy as np
import roma
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from einops import einsum
from packaging import version as pver


class ImageTransform:
    def __init__(
        self,
        crop_size,
        sample_size,
        max_crop,
        use_flip=False,
    ):
        self.use_flip = use_flip
        self.crop_size = crop_size
        self.max_crop = max_crop
        self.sample_size = sample_size
        self.crop_transform = transforms.CenterCrop(crop_size) if crop_size else lambda x: x
        self.resize_transform = transforms.Resize(sample_size) if sample_size else lambda x: x
        # if use_flip:
        #     self.flip_transform = RandomHorizontalFlipWithPose()

    def preprocess_images(self, images):
        # Returns the preprocessed images along with an image transform object
        # which describes the transformation on the image
        video = images

        if self.use_flip:
            assert False
            flip_flag = self.pixel_transforms[1].get_flip_flag(self.sample_n_frames)
        else:
            flip_flag = torch.zeros(images.shape[0], dtype=torch.bool, device=video.device)

        ori_h, ori_w = video.shape[-2:]
        if self.max_crop:
            # scale up to largest croppable size
            crop_ratio = min(ori_h / self.crop_size[0], ori_w / self.crop_size[1])
            new_crop_size = (int(self.crop_size[0] * crop_ratio), int(self.crop_size[1] * crop_ratio))
            # logger.info(f"Max crop ({new_crop_size})")
            self.crop_transform = transforms.CenterCrop(new_crop_size)

        video = self.crop_transform(video)
        new_h, new_w = video.shape[-2:]
        # NOTE! I'm using u,v convention here instead of h,w
        shift = ((new_w - ori_w) / 2, (new_h - ori_h) / 2)

        # resize:
        ori_h, ori_w = video.shape[-2:]
        # new_h, new_w = self.sample_size
        video = self.resize_transform(video)
        new_h, new_w = video.shape[-2:]
        scale = (new_w / ori_w, new_h / ori_h)

        if self.use_flip:
            video = self.flip_transform(video, flip_flag)
        return video, shift, scale, flip_flag

    def apply_img_transform(self, i, j, shift, scale):
        # takes pixel uv coordinates in un-transformed space and converts to new
        # coordinates of image after crop and resize

        # for compatibility with other camera models, explicitly define a image transformation
        # (due to crop and resize) as opposed to absorbing it into the pinhole
        # camera intrinsics matrix

        # first shift, then scale
        i = (i + shift[0]) * scale[0]
        j = (j + shift[1]) * scale[1]
        return i, j


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse("1.10"):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing="ij")


def get_grid_uvs(batch_shape, H, W, device, dtype=None, flip_flag=None, nh=None, nw=None, margin=0):
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
    if n_flip > 0:
        j_flip, i_flip = custom_meshgrid(
            torch.linspace(0, H - 1, nh, device=device, dtype=dtype),
            torch.linspace(W - 1, 0, nw, device=device, dtype=dtype),
        )
        i_flip = i_flip.reshape([1, 1, nh * nw]).expand(B, 1, nh * nw) + 0.5
        j_flip = j_flip.reshape([1, 1, nh * nw]).expand(B, 1, nh * nw) + 0.5
        i[:, flip_flag, ...] = i_flip
        j[:, flip_flag, ...] = j_flip
    return i, j


def get_rays_from_uvs(i, j, K, c2w):
    fx, fy, cx, cy = K.chunk(4, dim=-1)  # B,V, 1

    zs = torch.ones_like(i)  # [B, V, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)  # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, HW, 3
    rays_o = c2w[..., :3, 3]  # B, V, 3
    rays_o = rays_o[..., None, :].expand_as(rays_d)  # B, V, HW, 3
    return rays_o, rays_d


def project_to_uvs(pts, K, c2w):
    w2c = torch.linalg.inv(c2w)
    cam_pts = torch.einsum("...ij,...vj->...vi", w2c[..., :3, :3], pts) + w2c[..., None, :3, 3]

    fx, fy, cx, cy = K.chunk(4, dim=-1)  # B,V, 1

    xs = cam_pts[..., 0]
    ys = cam_pts[..., 1]
    zs = cam_pts[..., 2]

    us = (fx * xs / zs) + cx
    vs = (fy * ys / zs) + cy
    uvs = torch.stack([us, vs], dim=-1)
    return uvs, zs


def get_rays(K, c2w, H, W, device, flip_flag=None, nh=None, nw=None):
    i, j = get_grid_uvs(K.shape[:2], H=H, W=W, dtype=K.dtype, device=device, flip_flag=flip_flag, nh=nh, nw=nw)
    return get_rays_from_uvs(i, j, K, c2w)


def ray_condition(K, c2w, H, W, device, flip_flag=None):
    batch_shape = K.shape[:2]

    B, V = batch_shape
    rays_o, rays_d = get_rays(K, c2w, H, W, device, flip_flag=flip_flag)
    rays_dxo = torch.cross(rays_o, rays_d, dim=-1)  # B, V, HW, 3
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)[0].permute(0, 3, 1, 2).contiguous()  # B, V, 6, H, W
    rays_o = rays_o.reshape(B, c2w.shape[1], H, W, 3)[0].permute(0, 3, 1, 2).contiguous()
    rays_d = rays_d.reshape(B, c2w.shape[1], H, W, 3)[0].permute(0, 3, 1, 2).contiguous()
    return plucker, rays_o, rays_d


# Produce a 2D colored map for images representing the uv coordinates
def get_uv_map(img_size):
    # return a RGB color map for images representing the uv coordinates
    u = torch.linspace(0, 1, img_size[0])
    v = torch.linspace(0, 1, img_size[1])
    uv = torch.meshgrid(u, v)
    r = uv[0]  # Red channel
    g = uv[1]  # Green channel
    b = 1 - uv[0]  # Blue channel
    uv_map = torch.stack([r, g, b], dim=-1)  # Combine into RGB map
    return uv_map


def grid_distortion(images, strength=0.5):
    # images: [B, C, H, W]
    # num_steps: int, grid resolution for distortion
    # strength: float in [0, 1], strength of distortion

    B, C, H, W = images.shape

    num_steps = np.random.randint(8, 17)
    grid_steps = torch.linspace(-1, 1, num_steps)

    # have to loop batch...
    grids = []
    for b in range(B):
        # construct displacement
        x_steps = torch.linspace(0, 1, num_steps)  # [num_steps], inclusive
        x_steps = (x_steps + strength * (torch.rand_like(x_steps) - 0.5) / (num_steps - 1)).clamp(0, 1)  # perturb
        x_steps = (x_steps * W).long()  # [num_steps]
        x_steps[0] = 0
        x_steps[-1] = W
        xs = []
        for i in range(num_steps - 1):
            xs.append(torch.linspace(grid_steps[i], grid_steps[i + 1], x_steps[i + 1] - x_steps[i]))
        xs = torch.cat(xs, dim=0)  # [W]

        y_steps = torch.linspace(0, 1, num_steps)  # [num_steps], inclusive
        y_steps = (y_steps + strength * (torch.rand_like(y_steps) - 0.5) / (num_steps - 1)).clamp(0, 1)  # perturb
        y_steps = (y_steps * H).long()  # [num_steps]
        y_steps[0] = 0
        y_steps[-1] = H
        ys = []
        for i in range(num_steps - 1):
            ys.append(torch.linspace(grid_steps[i], grid_steps[i + 1], y_steps[i + 1] - y_steps[i]))
        ys = torch.cat(ys, dim=0)  # [H]

        # construct grid
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")  # [H, W]
        grid = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]

        grids.append(grid)

    grids = torch.stack(grids, dim=0).to(images.device)  # [B, H, W, 2]

    # grid sample
    images = F.grid_sample(images, grids, align_corners=False)

    return images


def orbit_camera_jitter(poses, strength=0.1):
    # poses: [B, 4, 4], assume orbit camera in opengl format
    # random orbital rotate

    B = poses.shape[0]
    rotvec_x = poses[:, :3, 1] * strength * np.pi * (torch.rand(B, 1, device=poses.device) * 2 - 1)
    rotvec_y = poses[:, :3, 0] * strength * np.pi / 2 * (torch.rand(B, 1, device=poses.device) * 2 - 1)

    rot = roma.rotvec_to_rotmat(rotvec_x) @ roma.rotvec_to_rotmat(rotvec_y)
    R = rot @ poses[:, :3, :3]
    T = rot @ poses[:, :3, 3:]

    new_poses = poses.clone()
    new_poses[:, :3, :3] = R
    new_poses[:, :3, 3:] = T

    return new_poses


def get_fov(intrinsics: torch.Tensor) -> torch.Tensor:
    intrinsics_inv = intrinsics.inverse()

    def process_vector(vector):
        vector = torch.tensor(vector, dtype=torch.float32, device=intrinsics.device)
        vector = einsum(intrinsics_inv, vector, "b i j, j -> b i")
        return vector / vector.norm(dim=-1, keepdim=True)

    left = process_vector([0, 0.5, 1])
    right = process_vector([1, 0.5, 1])
    top = process_vector([0.5, 0, 1])
    bottom = process_vector([0.5, 1, 1])
    fov_x = (left * right).sum(dim=-1).acos()
    fov_y = (top * bottom).sum(dim=-1).acos()
    return torch.stack((fov_x, fov_y), dim=-1)
