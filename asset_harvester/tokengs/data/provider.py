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

import random

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset

from ..utils.augmentation import augment_camera_uniform, random_reflect
from ..utils.data import ImageTransform, grid_distortion, orbit_camera_jitter, ray_condition
from ..utils.model import timestep_embedding
from .datafield import (
    DF_CAMERA_C2W_TRANSFORM,
    DF_CAMERA_INTRINSICS,
    DF_DEPTH,
    DF_FOREGROUND_MASK,
    DF_IMAGE_RGB,
)
from .registry import dataset_registry
from .static.assetharvest import AssetHarvest


class BreakingError(Exception):
    """A class for exceptions which should interrupt the training loop instead of being skipped silently."""

    pass


class Provider(Dataset):
    def __init__(self, dataset_name, opt, training=True, num_repeat=1):
        self.opt = opt
        self.norm_by_z_mean = "_zmean" in dataset_name
        if self.norm_by_z_mean:
            dataset_name = dataset_name.replace("_zmean", "")
        if "_scaled_" in dataset_name:
            # overwrite the scene scale
            scale_factor = float(dataset_name.split("_scaled_")[-1])
            dataset_name = dataset_name.split("_scaled_")[0]
            override_scene_scale = scale_factor
        else:
            override_scene_scale = None
        if opt.dataset_kwargs is not None:
            dataset_registry[dataset_name]["kwargs"].update(opt.dataset_kwargs)
        self.dataset = dataset_registry[dataset_name]["cls"](**dataset_registry[dataset_name]["kwargs"])
        self.scene_scale = (
            dataset_registry[dataset_name]["scene_scale"] if override_scene_scale is None else override_scene_scale
        )
        self.max_gap, self.min_gap = (
            dataset_registry[dataset_name]["max_gap"],
            dataset_registry[dataset_name]["min_gap"],
        )
        self.extrapolate_range = (
            dataset_registry[dataset_name]["extrapolate_range"]
            if "extrapolate_range" in dataset_registry[dataset_name]
            else None
        )
        self.training = training
        self.dataset.sample_list *= num_repeat

        if opt.evaluating:
            pass
        else:
            if training:
                self.dataset.sample_list = self.dataset.sample_list[: -self.opt.batch_size]
            else:
                self.dataset.sample_list = self.dataset.sample_list[-self.opt.batch_size :]

        self.rng = np.random.default_rng(self.opt.seed)
        self.generator = torch.Generator(device="cpu").manual_seed(self.opt.seed)

        self._setup_image_transforms(
            sample_size=self.opt.img_size,
            crop_size=self.opt.img_size,
            use_flip=False,
            max_crop=True,
        )

        self.data_fields = [DF_IMAGE_RGB, DF_CAMERA_C2W_TRANSFORM, DF_CAMERA_INTRINSICS, DF_FOREGROUND_MASK]

        if self.norm_by_z_mean:
            self.data_fields.append(DF_DEPTH)

    def set_rng_epoch(self, epoch: int) -> None:
        self.rng = np.random.default_rng(epoch + self.opt.seed)
        self.generator = torch.Generator(device="cpu").manual_seed(epoch + self.opt.seed)

    def __len__(self):
        return len(self.dataset)

    def _setup_image_transforms(self, sample_size, crop_size, use_flip, max_crop=False):
        self.image_transform = ImageTransform(
            crop_size=crop_size, sample_size=sample_size, use_flip=use_flip, max_crop=max_crop
        )
        self.input_normalizer = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False
        )

    def _normalize_camera_mean_cam(self, c2ws):
        input_c2ws = c2ws[: self.opt.num_input_views]
        # normalize input camera poses
        position_avg = input_c2ws[:, :3, 3].mean(0)  # (3,)
        forward_avg = input_c2ws[:, :3, 2].mean(0)  # (3,)
        down_avg = input_c2ws[:, :3, 1].mean(0)  # (3,)
        # gram-schmidt process
        forward_avg = F.normalize(forward_avg, dim=0)
        down_avg = F.normalize(down_avg - down_avg.dot(forward_avg) * forward_avg, dim=0)
        right_avg = torch.cross(down_avg, forward_avg)
        pos_avg = torch.stack([right_avg, down_avg, forward_avg, position_avg], dim=1)  # (3, 4)
        pos_avg = torch.cat([pos_avg, torch.tensor([[0, 0, 0, 1]], device=pos_avg.device).float()], dim=0)  # (4, 4)
        pos_avg_inv = torch.inverse(pos_avg)

        c2ws = torch.matmul(pos_avg_inv.unsqueeze(0), c2ws)
        return c2ws

    def _get_scene_scale_from_point_depth(self, c2ws, depths, intrinsics):
        H, W = depths.shape[-2:]
        # Get rays using ray_condition function
        # ray_condition expects [B, V, ...] shape, so add batch dimension
        _, rays_o, rays_d = ray_condition(intrinsics[None], c2ws[None], H, W, device="cpu", flip_flag=None)
        # rays_o: [N, 3, H, W], rays_d: [N, 3, H, W]

        # Rearrange to [N, H, W, 3] for easier computation
        rays_o = rays_o.permute(0, 2, 3, 1)  # [N, H, W, 3]
        rays_d = rays_d.permute(0, 2, 3, 1)  # [N, H, W, 3]
        depths_hw = depths.squeeze(1)  # [N, H, W]

        # Compute 3D points: point = ray_origin + ray_direction * depth
        points_3d = rays_o + rays_d * depths_hw[..., None]  # [N, H, W, 3]

        # Return mean z
        return self.scene_scale / points_3d[..., 2].mean()

    def _preprocess(self, rgbs, masks, depths, c2ws, intrinsics, timesteps, has_mask, has_depth):
        rgbs, shift, scale, flip_flag = self.image_transform.preprocess_images(rgbs)
        masks, _, _, _ = self.image_transform.preprocess_images(masks)
        depths, _, _, _ = self.image_transform.preprocess_images(depths)
        intrinsics = torch.stack(
            [
                intrinsics[..., 0] * scale[0],
                intrinsics[..., 1] * scale[1],
                (intrinsics[..., 2] + shift[0]) * scale[0],
                (intrinsics[..., 3] + shift[1]) * scale[1],
            ],
            dim=-1,
        )

        # relative pose
        if self.opt.camera_normalization_method == "objaverse":
            distance = c2ws[0, :3, 3].norm()
            c2ws = (
                torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -distance], [0, 0, 0, 1]], dtype=torch.float32)
                @ torch.inverse(c2ws[0])
            ).unsqueeze(0) @ c2ws
            # to reproduce the results of LGM
            # change -1.5 to +1.5
            # c2ws[:, :3, 1:3] *= -1 before and after the normalization
        elif self.opt.camera_normalization_method == "mean_cam":
            c2ws = self._normalize_camera_mean_cam(c2ws)
        elif self.opt.camera_normalization_method == "first_cam":
            c2ws = torch.inverse(c2ws[0]).unsqueeze(0) @ c2ws
        else:
            raise ValueError(f"Invalid camera normalization method: {self.opt.camera_normalization_method}")

        # compute the scene scale
        if self.norm_by_z_mean:
            assert has_depth, "Depth is required for z-mean normalization"
            final_scene_scale = self._get_scene_scale_from_point_depth(
                c2ws[: self.opt.num_input_views],
                depths[: self.opt.num_input_views],
                intrinsics[: self.opt.num_input_views],
            )
        elif self.opt.camera_scale_method == "constant":
            final_scene_scale = self.scene_scale
        elif self.opt.camera_scale_method == "distance":
            dist = max(torch.max(torch.norm(c2ws[: self.opt.num_input_views, :3, 3] - c2ws[0:1, :3, 3], dim=1)), 1e-6)
            final_scene_scale = self.scene_scale / dist
        elif self.opt.camera_scale_method == "bound":
            position_max = c2ws[: self.opt.num_input_views, :3, 3].abs().max()
            final_scene_scale = self.scene_scale / position_max
        else:
            raise ValueError(f"Invalid camera scale method: {self.opt.camera_scale_method}")

        # process input

        # data augmentation
        # random reflection
        if self.training and self.opt.random_reflect:
            rgbs, c2ws = random_reflect(rgbs, c2ws, generator=self.generator)
        # random scaling
        if self.training and self.opt.random_scale:
            min_scale, max_scale = self.opt.scale_range
            # log-uniform sampling for symmetric scaling
            log_min = torch.log(torch.tensor(min_scale, device=c2ws.device))
            log_max = torch.log(torch.tensor(max_scale, device=c2ws.device))
            random_log_scale = (
                torch.rand(1, device=c2ws.device, generator=self.generator)[0] * (log_max - log_min) + log_min
            )
            random_scale = torch.exp(random_log_scale)
            final_scene_scale = final_scene_scale * random_scale
        # camera augmentation to input
        c2ws_input = c2ws.clone()
        intrinsics_input = intrinsics.clone()
        if (self.training and self.opt.camera_augmentation) or self.opt.test_with_camera_augmentation:
            c2ws_input, intrinsics_input = augment_camera_uniform(
                self.generator,
                c2ws_input,
                intrinsics_input,
                self.opt.rot_deg_range,
                self.opt.trans_range,
                self.opt.intrin_range,
                self.opt.camera_augmentation_probability,
            )

        # finally apply the scene scale after augmentations
        c2ws[:, :3, 3] = c2ws[:, :3, 3] * final_scene_scale
        c2ws_input[:, :3, 3] = c2ws_input[:, :3, 3] * final_scene_scale

        # add perturbation to input images and camera poses
        # objaverse only!
        images_input = rgbs.clone()
        if self.training and self.opt.num_input_views > 1:
            # apply random grid distortion to simulate 3D inconsistency
            if random.random() < self.opt.prob_grid_distortion:
                images_input[1:] = grid_distortion(images_input[1:], strength=self.opt.grid_distortion_strength)
            # apply camera jittering (only to input!)
            if random.random() < self.opt.prob_cam_jitter:
                c2ws_input[1:] = orbit_camera_jitter(c2ws_input[1:])

        images_input = self.input_normalizer(images_input)
        # time embedding
        if self.opt.time_embedding:
            timesteps = (timesteps - timesteps.min()) / (
                timesteps.max() - timesteps.min()
            )  # normalize to 0 to 1 # [TV]
            time_embeddings = timestep_embedding(timesteps, self.opt.time_embedding_dim)  # [TV, D]
            time_embeddings = time_embeddings[..., None, None] * torch.ones_like(images_input[:, :1])  # [TV, D, H, W]
            images_input = torch.cat([images_input, time_embeddings], dim=1)
        # camera embedding
        plucker_embedding, rays_os, rays_ds = ray_condition(
            intrinsics_input[None],
            c2ws_input[None],
            self.opt.img_size[0],
            self.opt.img_size[1],
            device="cpu",
            flip_flag=flip_flag,
        )
        final_input = torch.cat([images_input, plucker_embedding], dim=1)

        return {
            "input": final_input,
            "rays_os": rays_os,
            "rays_ds": rays_ds,
            "images_all": rgbs,
            "images_input": rgbs[: self.opt.num_input_views],
            "images_output": rgbs[self.opt.num_input_views :],
            "intrinsics_all": intrinsics,
            "intrinsics": intrinsics[self.opt.num_input_views :],
            "intrinsics_input": intrinsics[: self.opt.num_input_views],
            "cam_view_all": torch.inverse(c2ws).transpose(1, 2),  # [V, 4, 4]
            "cam_view": torch.inverse(c2ws[self.opt.num_input_views :]).transpose(1, 2),  # [V, 4, 4]
            "masks_all": masks,
            "masks_output": masks[self.opt.num_input_views :],
            "has_mask": has_mask,
            "cam_to_world_input": c2ws[: self.opt.num_input_views],  # [V, 4, 4]
            "cam_view_input": torch.inverse(c2ws[: self.opt.num_input_views]).transpose(1, 2),  # [V, 4, 4]
        }

    def get_rng(self, idx: int) -> np.random.Generator:
        """
        Get a random number generator for the given index.
        If training, use a shared RNG to produce different samples across batches.
        If eval, use a fixed RNG for a given index to produce the same sample across evaluations.
        """
        if self.training:
            return self.rng
        else:
            return np.random.default_rng(self.opt.seed + idx)

    def _get_indices_dynamic(self, idx):
        rng = self.get_rng(idx)
        total_num_frames = self.dataset.count_frames(idx)
        camera_count = self.dataset.count_cameras(idx)
        assert total_num_frames >= self.opt.num_input_views, (
            f"Frame number {total_num_frames} is smaller than number of input views {self.opt.num_input_views}."
        )
        context_gap = rng.integers(self.min_gap, self.max_gap + 1)
        context_gap = max(min(total_num_frames - 1, context_gap), self.opt.num_input_views - 1)

        start_frame = rng.integers(0, total_num_frames - context_gap)
        inbetween_indices = np.sort(
            rng.permutation(np.arange(start_frame + 1, start_frame + context_gap))[: self.opt.num_input_views - 2]
        )
        frame_indices = np.array([start_frame, *inbetween_indices, start_frame + context_gap])
        target_index = rng.permutation(np.arange(start_frame, start_frame + context_gap + 1))[:1]

        if not self.opt.use_interp_target:
            target_index = rng.permutation(frame_indices)[:1]

        # append to frame indices
        frame_indices = np.concatenate([frame_indices, target_index])

        if self.opt.num_input_views > camera_count:
            view_indices = rng.permutation(np.arange(self.opt.num_input_views) % camera_count)
        else:
            view_indices = rng.permutation(np.arange(camera_count))[: self.opt.num_input_views]

        return frame_indices, view_indices

    def _get_indices_static(self, idx):
        rng = self.get_rng(idx)

        total_num_frames = self.dataset.count_frames(idx)
        assert total_num_frames >= max(self.opt.num_input_views, self.opt.num_views - self.opt.num_input_views), (
            f"Frame number {total_num_frames} is smaller than number of input views {max(self.opt.num_input_views, self.opt.num_views - self.opt.num_input_views)}."
        )
        context_gap = rng.integers(self.min_gap, self.max_gap + 1)
        context_gap = max(min(total_num_frames - 1, context_gap), self.opt.num_input_views - 1)
        start_frame = rng.integers(0, total_num_frames - context_gap)
        inbetween_indices = np.sort(
            rng.permutation(np.arange(start_frame + 1, start_frame + context_gap))[: self.opt.num_input_views - 2]
        )
        frame_indices = np.array([start_frame, *inbetween_indices, start_frame + context_gap])
        target_index = rng.permutation(np.arange(start_frame, start_frame + context_gap + 1))[
            : self.opt.num_views - self.opt.num_input_views
        ]

        # append to frame indices
        frame_indices = np.concatenate([frame_indices, target_index])

        return frame_indices, []

    def _get_indices_static_objaverse(self, idx):
        rng = self.get_rng(idx)
        context_views = rng.permutation(np.arange(16))[: self.opt.num_input_views]
        target_views = rng.permutation(np.arange(16, 16 + 32))[: self.opt.num_views - self.opt.num_input_views]
        frame_indices = np.concatenate([context_views, target_views])

        return frame_indices, []

    def _get_indices_static_asset_harvest(self, idx):
        context_views = np.arange(16)[: self.opt.num_input_views]
        cond_views_indices = np.arange(16, self.dataset.count_frames(idx))
        frame_indices = np.concatenate([context_views, cond_views_indices])
        return frame_indices, []

    def _get_indices_static_extrapolate(self, idx):
        """
        Get indices for static scenes with extrapolation.
        Target frames can fall outside the context frame range.

        Example: context frames [30, 40] -> target frames [25, 35, 45]

        The extrapolation range is controlled by opt.extrapolate_range.
        """
        rng = self.get_rng(idx)

        total_num_frames = self.dataset.count_frames(idx)
        assert total_num_frames >= self.opt.num_input_views, (
            f"Frame number {total_num_frames} is smaller than number of input views {self.opt.num_input_views}."
        )

        # Sample context frames (same as _get_indices_static)
        context_gap = rng.integers(self.min_gap, self.max_gap + 1)
        context_gap = max(min(total_num_frames - 1, context_gap), self.opt.num_input_views - 1)
        start_frame = rng.integers(0, total_num_frames - context_gap)
        inbetween_indices = np.sort(
            rng.permutation(np.arange(start_frame + 1, start_frame + context_gap))[: self.opt.num_input_views - 2]
        )
        frame_indices = np.array([start_frame, *inbetween_indices, start_frame + context_gap])

        extrapolate_range = self.extrapolate_range
        # Define extended range for sampling targets (can go beyond context range)
        extended_start = max(0, start_frame - extrapolate_range)
        extended_end = min(total_num_frames - 1, start_frame + context_gap + extrapolate_range)

        # Sample target indices from the extended range
        num_targets = self.opt.num_views - self.opt.num_input_views
        available_indices = np.arange(extended_start, extended_end + 1)
        target_index = rng.choice(available_indices, size=num_targets, replace=False)

        # Append to frame indices
        frame_indices = np.concatenate([frame_indices, target_index])

        return frame_indices, []

    def _get_indices_eval(self, idx):
        """
        Get indices for evaluation datasets that have predefined context and target frames.
        Uses the get_context_target_frames method from the dataset.
        """
        context_frames, target_frames = self.dataset.get_context_target_frames(idx)

        # Convert to numpy arrays
        context_frames = np.array(context_frames)
        target_frames = np.array(target_frames)

        # Concatenate context and target frames
        frame_indices = np.concatenate([context_frames, target_frames])

        return frame_indices, []

    def _curate_batch_static(self, all_rgbs, all_masks, all_depths, all_c2ws, all_intrinsics, frame_indices):
        return all_rgbs, all_masks, all_depths, all_c2ws, all_intrinsics, frame_indices

    def _curate_batch_dynamic(self, all_rgbs, all_masks, all_depths, all_c2ws, all_intrinsics, frame_indices):
        rgbs, masks, depths, c2ws, intrinsics, timesteps = [], [], [], [], [], []
        all_rgbs = all_rgbs.reshape([self.opt.num_input_views, self.opt.num_input_views + 1, *all_rgbs.shape[1:]])
        all_masks = all_masks.reshape([self.opt.num_input_views, self.opt.num_input_views + 1, *all_masks.shape[1:]])
        all_depths = all_depths.reshape([self.opt.num_input_views, self.opt.num_input_views + 1, *all_depths.shape[1:]])
        all_c2ws = all_c2ws.reshape([self.opt.num_input_views, self.opt.num_input_views + 1, *all_c2ws.shape[1:]])
        all_intrinsics = all_intrinsics.reshape(
            [self.opt.num_input_views, self.opt.num_input_views + 1, *all_intrinsics.shape[1:]]
        )
        # input views
        for v in range(self.opt.num_input_views):
            rgbs.append(all_rgbs[v, v])
            masks.append(all_masks[v, v])
            depths.append(all_depths[v, v])
            c2ws.append(all_c2ws[v, v])
            intrinsics.append(all_intrinsics[v, v])
            timesteps.append(frame_indices[v])
        # supervision views
        assert self.opt.num_views <= self.opt.num_input_views * 2, (
            f"Total views should be less than twice of input views {self.opt.num_input_views}, instead got {self.opt.num_views}"
        )
        for v in range(self.opt.num_views - self.opt.num_input_views):
            rgbs.append(all_rgbs[v, -1])
            masks.append(all_masks[v, -1])
            depths.append(all_depths[v, -1])
            c2ws.append(all_c2ws[v, -1])
            intrinsics.append(all_intrinsics[v, -1])
            timesteps.append(frame_indices[-1])
        rgbs, masks, depths, c2ws, intrinsics, timesteps = (
            torch.stack(rgbs),
            torch.stack(masks),
            torch.stack(depths),
            torch.stack(c2ws),
            torch.stack(intrinsics),
            torch.stack(timesteps),
        )
        return rgbs, masks, depths, c2ws, intrinsics, timesteps

    def _curate_batch_dynamic_eval(self, all_rgbs, all_masks, all_depths, all_c2ws, all_intrinsics, frame_indices):
        """
        Curate batch for dynamic evaluation datasets (e.g., DyCheckMVEval).

        For evaluation:
        - First num_input_views frames are context (from one camera, specific timesteps)
        - Remaining frames are targets (from other cameras, various timesteps)
        - No grid structure needed, data is already in the correct order
        """
        # Split context and target
        num_context = self.opt.num_input_views

        # Context frames
        rgbs_context = all_rgbs[:num_context]
        masks_context = all_masks[:num_context]
        depths_context = all_depths[:num_context]
        c2ws_context = all_c2ws[:num_context]
        intrinsics_context = all_intrinsics[:num_context]
        timesteps_context = frame_indices[:num_context]

        # Target frames
        rgbs_target = all_rgbs[num_context:]
        masks_target = all_masks[num_context:]
        depths_target = all_depths[num_context:]
        c2ws_target = all_c2ws[num_context:]
        intrinsics_target = all_intrinsics[num_context:]
        timesteps_target = frame_indices[num_context:]

        # Concatenate context and target
        rgbs = torch.cat([rgbs_context, rgbs_target], dim=0)
        masks = torch.cat([masks_context, masks_target], dim=0)
        depths = torch.cat([depths_context, depths_target], dim=0)
        c2ws = torch.cat([c2ws_context, c2ws_target], dim=0)
        intrinsics = torch.cat([intrinsics_context, intrinsics_target], dim=0)
        timesteps = torch.cat([timesteps_context, timesteps_target], dim=0)

        return rgbs, masks, depths, c2ws, intrinsics, timesteps

    def get_item(self, idx):
        if hasattr(self.dataset, "get_context_target_frames"):
            _get_indices_fn = self._get_indices_eval
            # Use special curate function for dynamic evaluation datasets
            _curate_batch_fn = self._curate_batch_static if self.dataset.is_static else self._curate_batch_dynamic_eval
        elif self.extrapolate_range is not None:
            _get_indices_fn = self._get_indices_static_extrapolate
            _curate_batch_fn = self._curate_batch_static
        else:
            _get_indices_fn = self._get_indices_static if self.dataset.is_static else self._get_indices_dynamic
            if self.opt.use_objaverse_sampling:
                _get_indices_fn = self._get_indices_static_objaverse
            _curate_batch_fn = self._curate_batch_static if self.dataset.is_static else self._curate_batch_dynamic
        # override for asset harvest dataset
        if isinstance(self.dataset, AssetHarvest):
            _get_indices_fn = self._get_indices_static_asset_harvest
            _curate_batch_fn = self._curate_batch_static

        frame_indices, view_indices = _get_indices_fn(idx)
        original_output_dict = self.dataset.get_data(
            idx, data_fields=self.data_fields, frame_indices=frame_indices, view_indices=view_indices
        )

        if not (has_mask := (DF_FOREGROUND_MASK in original_output_dict)):
            original_output_dict[DF_FOREGROUND_MASK] = torch.ones_like(original_output_dict[DF_IMAGE_RGB][:, 0:1, ...])
        if not (has_depth := (DF_DEPTH in original_output_dict)):
            original_output_dict[DF_DEPTH] = torch.ones_like(original_output_dict[DF_IMAGE_RGB][:, 0:1, ...])
            has_depth = False

        all_rgbs, all_c2ws, all_intrinsics, all_masks, all_depths = (
            original_output_dict[DF_IMAGE_RGB],
            original_output_dict[DF_CAMERA_C2W_TRANSFORM],
            original_output_dict[DF_CAMERA_INTRINSICS],
            original_output_dict[DF_FOREGROUND_MASK],
            original_output_dict[DF_DEPTH],
        )

        rgbs, masks, depths, c2ws, intrinsics, timesteps = _curate_batch_fn(
            all_rgbs, all_masks, all_depths, all_c2ws, all_intrinsics, torch.from_numpy(frame_indices).float()
        )

        return {
            **self._preprocess(rgbs, masks, depths, c2ws, intrinsics, timesteps, has_mask, has_depth),
            "__key__": str(original_output_dict["__key__"]),
        }

    def __getitem__(self, idx):
        while True:
            try:
                results = self.get_item(idx)
                break
            except BreakingError:
                raise  # these are errors that should stop training
            except Exception as e:
                if self.opt.debug:
                    print(f"data loader error: {e}")
                idx = self.rng.integers(0, len(self.dataset))
        return results
