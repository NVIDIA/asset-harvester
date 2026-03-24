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

import glob
import os
import shutil
import tarfile
import tempfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import boto3
import cv2
import numpy as np
import torch

from ...utils import orbit_camera
from ..datafield import (
    DF_CAMERA_C2W_TRANSFORM,
    DF_CAMERA_INTRINSICS,
    DF_FOREGROUND_MASK,
    DF_IMAGE_RGB,
)


class AssetHarvest:
    """
    Asset Harvesting dataset

    Args:
        datadir: Path to the dataset directory
        realfocal: Whether to use real focal length from data
        bbox_size: Bounding box size scaling factor
        ttt_supervision_mode: Mode for TTT supervision. Options:
            - "input_only": Use only input images for TTT supervision (default)
            - "cond_only": Use only cond_views for TTT supervision
            - "both": Use both input images and cond_views for TTT supervision
            Note: Cond_views images are always loaded for visualization regardless of mode
    """

    def __init__(self, datadir, realfocal=False, bbox_size=1.0, ttt_supervision_mode="input_only"):
        self.realfocal = realfocal
        self.bbox_size = float(bbox_size)
        self.ttt_supervision_mode = ttt_supervision_mode

        # Validate supervision mode
        valid_modes = ["input_only", "cond_only", "both"]
        if self.ttt_supervision_mode not in valid_modes:
            raise ValueError(f"ttt_supervision_mode must be one of {valid_modes}, got {self.ttt_supervision_mode}")

        self.sample_list = [
            str(subdir) for subdir in Path(datadir).iterdir() if subdir.is_dir() and "wandb" not in subdir.name
        ]

        # default camera intrinsics
        fovy = 49.1
        self.tan_half_fov = np.tan(0.5 * np.deg2rad(fovy))

        self.is_static = True

    def __len__(self):
        return len(self.sample_list)

    def count_frames(self, idx):
        # Always count cond_views if they exist (for visualization)
        path = self.sample_list[idx]
        cond_views_folder = path + "/cond_views"
        if os.path.exists(cond_views_folder):
            # Count jpeg files in cond_views
            num_cond_views = len(glob.glob(os.path.join(cond_views_folder, "*.jpeg")))
            return 16 + num_cond_views
        return 16

    def count_cameras(self, idx):
        return 1

    def get_data(
        self,
        video_idx: int,
        frame_indices: list[int],
        view_indices: list[int],
        data_fields: list[str],
    ):
        path = self.sample_list[video_idx]
        input_folder = path + "/recon_input"
        cond_views_folder = path + "/cond_views"

        if self.realfocal:
            fov_input = np.loadtxt(input_folder + "/fov.txt")
            dist_input = np.loadtxt(input_folder + "/dist.txt")
            lwh_input = np.loadtxt(input_folder + "/lwh.txt")
            dist_input = dist_input / max(lwh_input)
            dist_input = dist_input * self.bbox_size

        # Initialize cond_views camera parameters
        fov_cond = None
        dist_cond = None
        cam_pos_cond = None

        # Load cond_views camera parameters if available (always load for visualization)
        if os.path.exists(cond_views_folder):
            if self.realfocal and os.path.exists(cond_views_folder + "/fov.txt"):
                fov_cond = np.loadtxt(cond_views_folder + "/fov.txt")
                dist_cond = np.loadtxt(cond_views_folder + "/dist.txt")
                lwh_cond = np.loadtxt(cond_views_folder + "/lwh.txt")
                # Normalize distance by lwh (same as recon_input)
                dist_cond = dist_cond / max(lwh_cond)
                dist_cond = dist_cond * self.bbox_size
                # cam_pos_cond is [3,] array with normalized direction vector
                cam_pos_cond = (
                    np.loadtxt(cond_views_folder + "/cam_pos.txt") * -1
                )  # the saved camera poses are inverted

                if len(cam_pos_cond.shape) < 2:
                    cam_pos_cond = np.array([cam_pos_cond])
                    dist_cond = np.array([dist_cond])
                    fov_cond = np.array([fov_cond])
                cam_pos_cond = cam_pos_cond / np.linalg.norm(cam_pos_cond, axis=1, keepdims=True)

        N_INPUT_VIEWS = 16

        # A dictionary that maps from data_field name to the list of raw data
        loaded_data = defaultdict(list)

        for frame_idx in frame_indices:
            # Determine if this is from recon_input or cond_views
            if frame_idx < N_INPUT_VIEWS:
                # Load from recon_input
                if DF_IMAGE_RGB in data_fields:
                    image_path = os.path.join(input_folder, f"{frame_idx}.png")
                    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    loaded_data[DF_IMAGE_RGB].append(image)

                if DF_FOREGROUND_MASK in data_fields:
                    mask = np.ones_like(image[..., 0])
                    loaded_data[DF_FOREGROUND_MASK].append(mask)

                if DF_CAMERA_INTRINSICS in data_fields or DF_CAMERA_C2W_TRANSFORM in data_fields:
                    img_size = image.shape[0]  # square image
                    azi = (360 / N_INPUT_VIEWS * frame_idx + 90 + 180) % 360
                    elevation = 0
                    if not self.realfocal:
                        c2w = orbit_camera(elevation, azi, radius=1.5, opengl=True)
                        f = img_size / (2 * self.tan_half_fov)
                        K = np.array([f, f, img_size / 2.0, img_size / 2.0])
                    else:
                        c2w = orbit_camera(elevation, azi, radius=dist_input, opengl=True)
                        tan_half_fov_input = np.tan(0.5 * np.deg2rad(fov_input))
                        f = img_size / (2 * tan_half_fov_input)
                        K = np.array([f, f, img_size / 2.0, img_size / 2.0])

                    if DF_CAMERA_INTRINSICS in data_fields:
                        loaded_data[DF_CAMERA_INTRINSICS].append(K)

                    if DF_CAMERA_C2W_TRANSFORM in data_fields:
                        # opengl to colmap camera for gaussian renderer
                        c2w[:3, 1:3] *= -1  # invert up & forward direction
                        loaded_data[DF_CAMERA_C2W_TRANSFORM].append(c2w)
            else:
                # Load from cond_views
                cond_idx = frame_idx - N_INPUT_VIEWS

                # Load mask first (if available) to apply to the image
                mask_path = os.path.join(cond_views_folder, f"{cond_idx}_mask.png")
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                    if len(mask.shape) == 3:
                        mask = mask[..., 0]
                    has_mask = True
                else:
                    mask = None
                    has_mask = False

                if DF_IMAGE_RGB in data_fields:
                    image_path = os.path.join(cond_views_folder, f"{cond_idx}.jpeg")
                    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    # Apply mask to make background white
                    if has_mask:
                        # Normalize mask to [0, 1]
                        mask_normalized = mask.astype(np.float32) / 255.0
                        # Expand mask to 3 channels
                        mask_3ch = mask_normalized[..., None]
                        # Blend: foreground * mask + white_background * (1 - mask)
                        white_bg = np.ones_like(image, dtype=np.float32) * 255.0
                        image = (image.astype(np.float32) * mask_3ch + white_bg * (1 - mask_3ch)).astype(np.uint8)

                    loaded_data[DF_IMAGE_RGB].append(image)

                if DF_FOREGROUND_MASK in data_fields:
                    if has_mask:
                        loaded_data[DF_FOREGROUND_MASK].append(mask)
                    else:
                        loaded_data[DF_FOREGROUND_MASK].append(np.ones_like(image[..., 0]))

                if DF_CAMERA_INTRINSICS in data_fields or DF_CAMERA_C2W_TRANSFORM in data_fields:
                    img_size = image.shape[0]  # square image

                    if self.realfocal and fov_cond is not None:
                        # Use the actual camera position from cam_pos.txt
                        tan_half_fov_cond = np.tan(0.5 * np.deg2rad(fov_cond[cond_idx]))
                        f = img_size / (2 * tan_half_fov_cond)
                        K = np.array([f, f, img_size / 2.0, img_size / 2.0])

                        x = cam_pos_cond[cond_idx][1] * -1
                        y = cam_pos_cond[cond_idx][2]
                        z = cam_pos_cond[cond_idx][0]

                        # Compute azimuth and elevation from camera position
                        azimuth = np.arctan2(z, x) * 180 / np.pi + 180
                        elevation = np.arcsin(y) * 180 / np.pi

                        c2w = orbit_camera(elevation, azimuth, radius=dist_cond[cond_idx], opengl=True)
                    else:
                        raise ValueError(
                            f"Unsupported camera configuration for cond_views: realfocal={self.realfocal}, fov_cond={fov_cond is not None}, dist_cond={dist_cond is not None}"
                        )

                    if DF_CAMERA_INTRINSICS in data_fields:
                        loaded_data[DF_CAMERA_INTRINSICS].append(K)

                    if DF_CAMERA_C2W_TRANSFORM in data_fields:
                        # opengl to colmap camera for gaussian renderer
                        c2w[:3, 1:3] *= -1  # invert up & forward direction
                        loaded_data[DF_CAMERA_C2W_TRANSFORM].append(c2w)

        output_dict = {}
        # Use filename as key
        output_dict["__key__"] = path
        for data_field in data_fields:
            if data_field == DF_IMAGE_RGB:
                rgb_np = np.stack(loaded_data[DF_IMAGE_RGB], axis=0).astype(np.float32) / 255.0
                rgb_torch = torch.from_numpy(rgb_np).moveaxis(-1, 1).contiguous()  # BHWC -> BCHW
                output_dict[data_field] = rgb_torch

            elif data_field == DF_CAMERA_C2W_TRANSFORM:
                c2w_np = np.stack(loaded_data[DF_CAMERA_C2W_TRANSFORM], axis=0).astype(np.float32)
                c2w_torch = torch.from_numpy(c2w_np).contiguous()
                output_dict[data_field] = c2w_torch

            elif data_field == DF_CAMERA_INTRINSICS:
                intrinsics_np = np.stack(loaded_data[DF_CAMERA_INTRINSICS], axis=0).astype(np.float32)
                intrinsics_torch = torch.from_numpy(intrinsics_np).contiguous()
                output_dict[data_field] = intrinsics_torch

            elif data_field == DF_FOREGROUND_MASK:
                mask_np = np.stack(loaded_data[DF_FOREGROUND_MASK], axis=0).astype(np.float32) / 255.0
                mask_torch = torch.from_numpy(mask_np)[:, None].contiguous()
                output_dict[data_field] = mask_torch

        return output_dict


class AssetHarvestTrain:
    def __init__(
        self,
        datalist,
        s3_config=None,
        fixed_only=False,
        bucket_name="objaverse_av",
        prefix="static_aug4",
        local_cache_dir=None,
        split_tar_mode=False,
    ):
        self.client = boto3.client("s3", **s3_config) if s3_config is not None else None

        self.sample_list = []
        with open(datalist) as f:
            for line in f.readlines():
                self.sample_list.append(line.strip())

        self.length = len(self.sample_list)
        self.is_static = True

        # default camera intrinsics
        self.size = 1024
        self.fovy = 49.1
        tan_half_fov = np.tan(0.5 * np.deg2rad(self.fovy))
        self.f = self.size / (2 * tan_half_fov)

        self.sensor_width = 32
        self.bucket_name = bucket_name
        self.prefix = prefix

        self.fixed_only = fixed_only
        self.local_cache_dir = local_cache_dir
        self.split_tar_mode = split_tar_mode  # True: load uid/fixed.tar and uid/random_av.tar, False: load uid.tar

    def __len__(self):
        return len(self.sample_list)

    def count_frames(self, idx):
        if self.fixed_only:
            # only 16 fixed cameras
            return 16
        else:
            return 16 + 32

    def count_cameras(self, idx):
        return 1

    @staticmethod
    def load_np_array_from_tar(tar, path):
        array_file = BytesIO()
        array_file.write(tar.extractfile(path).read())
        array_file.seek(0)
        return np.load(array_file)

    def _load_tar(self, tar_key):
        """Helper function to load a tar file from S3 or local cache."""
        if self.local_cache_dir is not None:
            # Create cache directory structure if it doesn't exist
            cache_dir = os.path.join(self.local_cache_dir, self.prefix)
            local_path = os.path.join(cache_dir, tar_key)

            # Create subdirectory for the tar files if needed
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            # Download from S3 if not cached locally (using atomic writes to prevent incomplete files)
            if not os.path.exists(local_path):
                s3_key = self.prefix + "/" + tar_key
                response = self.client.get_object(Bucket=self.bucket_name, Key=s3_key)
                # Write to temporary file first, then move atomically
                with tempfile.NamedTemporaryFile(mode="wb", delete=False, dir=os.path.dirname(local_path)) as tmp_file:
                    tmp_file.write(response["Body"].read())
                    tmp_path = tmp_file.name
                shutil.move(tmp_path, local_path)

            # Open from local file
            return tarfile.open(local_path, "r")
        else:
            # Original behavior: download from S3 directly
            s3_key = self.prefix + "/" + tar_key
            response = self.client.get_object(Bucket=self.bucket_name, Key=s3_key)
            return tarfile.open(fileobj=BytesIO(response["Body"].read()))

    def get_data(
        self,
        video_idx: int,
        frame_indices: list[int],
        view_indices: list[int],
        data_fields: list[str],
    ):
        uid = self.sample_list[video_idx]

        if self.split_tar_mode:
            # Mode 2: Load separate tar files for fixed and random_av
            tar_files = {
                "fixed": self._load_tar(f"{uid}/fixed.tar"),
                "random_av": self._load_tar(f"{uid}/random_av.tar"),
            }
        else:
            # Mode 1: Load single tar file containing both subfolders
            tar_filename = uid + ".tar"
            tar = self._load_tar(tar_filename)
            tar_files = None

        # A dictionary that maps from data_field name to the list of raw data
        loaded_data = defaultdict(list)

        for frame_idx in frame_indices:
            if frame_idx < 16:
                t = frame_idx
                subfolder = "fixed"
            else:
                t = frame_idx - 16
                subfolder = "random_av"

            # Select the appropriate tar file based on mode
            if self.split_tar_mode:
                current_tar = tar_files[subfolder]
                path_prefix = "."  # In split mode, files are at root of each tar
            else:
                current_tar = tar
                path_prefix = os.path.join(".", subfolder)  # In single tar mode, files are in subfolders

            if DF_IMAGE_RGB in data_fields:
                image_path = os.path.join(path_prefix, "img", f"{t:03d}.jpg")
                image = np.frombuffer(current_tar.extractfile(image_path).read(), np.uint8)
                image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)  # [512, 512, 4] in [0, 1]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                loaded_data[DF_IMAGE_RGB].append(image)

            if DF_FOREGROUND_MASK in data_fields:
                mask_path = os.path.join(path_prefix, "mask", f"{t:03d}.png")
                mask = np.frombuffer(current_tar.extractfile(mask_path).read(), np.uint8)
                mask = cv2.imdecode(mask, cv2.IMREAD_UNCHANGED)  # [512, 512] in [0, 1]
                loaded_data[DF_FOREGROUND_MASK].append(mask)

            if DF_CAMERA_INTRINSICS in data_fields or DF_CAMERA_C2W_TRANSFORM in data_fields:
                if DF_CAMERA_INTRINSICS in data_fields:
                    focal_path = os.path.join(path_prefix, "camera", "focal.npy")
                    focal = self.load_np_array_from_tar(current_tar, focal_path)[t, None]
                    focal = float(focal)
                    fov = np.arctan(self.sensor_width / (2 * focal)) * 2 * 180 / np.pi
                    f = self.size / (2 * np.tan(0.5 * np.deg2rad(fov)))
                    intrinsics_data = np.asarray(
                        [
                            f,
                            f,
                            self.size / 2,
                            self.size / 2,
                        ],
                        dtype=np.float32,
                    )
                    loaded_data[DF_CAMERA_INTRINSICS].append(intrinsics_data)

                if DF_CAMERA_C2W_TRANSFORM in data_fields:
                    elevation_path = os.path.join(path_prefix, "camera", "elevation.npy")
                    rotation_path = os.path.join(path_prefix, "camera", "rotation.npy")
                    distance_path = os.path.join(path_prefix, "camera", "distance.npy")

                    azi = self.load_np_array_from_tar(current_tar, rotation_path)[t, None]
                    elevation = (
                        self.load_np_array_from_tar(current_tar, elevation_path)[t, None] * -1
                    )  # to align with pretrained LGM
                    distance = self.load_np_array_from_tar(current_tar, distance_path)[t, None]
                    azi = float(azi)
                    elevation = float(elevation)
                    distance = float(distance)

                    c2w = orbit_camera(elevation, azi, radius=distance, opengl=True)

                    # opengl to colmap camera for gaussian renderer
                    c2w[:3, 1:3] *= -1  # invert up & forward direction

                    loaded_data[DF_CAMERA_C2W_TRANSFORM].append(c2w)  # check OpenCV

        # Close tar files
        if self.split_tar_mode:
            tar_files["fixed"].close()
            tar_files["random_av"].close()
        else:
            tar.close()

        output_dict = {}
        # Use filename as key
        output_dict["__key__"] = uid.split("/")[-1].replace(".tar", "")
        for data_field in data_fields:
            if data_field == DF_IMAGE_RGB:
                rgb_np = np.stack(loaded_data[DF_IMAGE_RGB], axis=0).astype(np.float32) / 255.0
                rgb_torch = torch.from_numpy(rgb_np).moveaxis(-1, 1).contiguous()  # BHWC -> BCHW
                output_dict[data_field] = rgb_torch

            elif data_field == DF_CAMERA_C2W_TRANSFORM:
                c2w_np = np.stack(loaded_data[DF_CAMERA_C2W_TRANSFORM], axis=0).astype(np.float32)
                c2w_torch = torch.from_numpy(c2w_np).contiguous()
                output_dict[data_field] = c2w_torch

            elif data_field == DF_CAMERA_INTRINSICS:
                intrinsics_np = np.stack(loaded_data[DF_CAMERA_INTRINSICS], axis=0).astype(np.float32)
                intrinsics_torch = torch.from_numpy(intrinsics_np).contiguous()
                output_dict[data_field] = intrinsics_torch

            elif data_field == DF_FOREGROUND_MASK:
                mask_np = np.stack(loaded_data[DF_FOREGROUND_MASK], axis=0).astype(np.float32) / 255.0
                mask_torch = torch.from_numpy(mask_np)[:, None].contiguous()
                output_dict[data_field] = mask_torch

        return output_dict
