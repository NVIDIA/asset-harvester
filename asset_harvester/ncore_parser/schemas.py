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


from pydantic import BaseModel, Field


class NCoreParserConfig(BaseModel):
    """Configuration for the ncore parser pipeline."""

    target_resolution: int = Field(default=512, description="Output crop resolution")
    num_lidar_ref_frames: int = Field(default=10, description="Max lidar frames to sample per track")
    cam_pose_flip: list[int] = Field(default=[1, 1, -1], description="Pose coordinate flip factors")
    max_threads: int = Field(default=4, description="Max threads for torch")
    occ_rate_threshold: float = Field(default=0.5, description="Occlusion rate threshold to filter views")
    crop_min_area_ratio: float = Field(default=0.002, description="Min bbox area ratio to process")
    mask_exceed_threshold: float = Field(default=0.5, description="Mask exceed threshold")
    min_instance_pixels: int = Field(default=100, description="Min pixels for valid instance")
    mask_overlap_threshold: float = Field(default=0.3, description="Mask overlap threshold")
    camera_ids: list[str] = Field(
        default=[
            "camera_front_wide_120fov",
            "camera_rear_right_70fov",
            "camera_rear_left_70fov",
            "camera_cross_left_120fov",
            "camera_cross_right_120fov",
        ],
        description="Camera IDs to use",
    )

    segmentation_ckpt: str = Field(description="Path to Mask2Former checkpoint")


class MultiViewData(BaseModel):
    """Serialized multi-view data for an asset."""

    frames_count: int
    cam_poses: list[list[float]]
    dists: list[float]
    fov: list[float]
    sensor_ids: list[str]
    lwh: list[float]


class Asset(BaseModel):
    """Single extracted asset with metadata."""

    clip_id: str
    track_id: str
    label_class: str
    cuboids_dims: list[float]
    multiview_data: MultiViewData


class AssetHarvestingMetadata(BaseModel):
    """Output metadata for a harvesting run."""

    clip_id: str
    config: NCoreParserConfig
    assets: dict[str, Asset] | None = None
