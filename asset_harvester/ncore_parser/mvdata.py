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

import io
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image


def image_bytesio(images: list[np.ndarray], enc: str) -> list[io.BytesIO]:
    """Convert numpy images to BytesIO objects with specified encoding."""
    bufs = []
    for image in images:
        if image.dtype != np.uint8:
            im = Image.fromarray((255 * image).astype(np.uint8))
        else:
            im = Image.fromarray(image)
        buf = io.BytesIO()
        im.save(buf, format=enc)
        buf.seek(0)
        bufs.append(buf)
    return bufs


@dataclass
class MVData:
    """Images and metadata of an object in a clip."""

    clip_id: str
    obj_id: str
    frames: np.ndarray
    cam_poses: np.ndarray
    dists: np.ndarray
    fov: np.ndarray
    npct: str
    bbox_pos: np.ndarray
    bbox_pix: list[np.ndarray]
    lwh: np.ndarray | None = field(default=None)
    ray_map: np.ndarray | None = field(default=None)
    og_image: np.ndarray | None = field(default=None)
    label_source: np.ndarray | None = field(default=None)
    sensor_id: list[str] = field(default_factory=list)
    caption: list[str] = field(default_factory=list)
    masks: np.ndarray | None = field(default=None)
    is_occluded: np.ndarray | None = field(default=None)
    masks_instance: list[np.ndarray] | None = field(default=None)
    masks_semantic: list[np.ndarray] | None = field(default=None)
    ori_instance_seg: list[np.ndarray] | None = field(default=None)

    def append(
        self,
        frame: np.ndarray,
        cam_pose: np.ndarray,
        dist: np.ndarray,
        fov: np.ndarray,
        og_image: np.ndarray | None = None,
        label_source: np.ndarray | None = None,
        sensor_id: str | None = None,
        mask: np.ndarray | None = None,
        is_occluded: np.ndarray | None = None,
        ray_map: np.ndarray | None = None,
        bbox_pix: np.ndarray | None = None,
    ) -> None:
        """Append a new frame and its associated metadata."""
        self.frames = np.concatenate([self.frames, frame], axis=0)
        self.cam_poses = np.concatenate([self.cam_poses, cam_pose], axis=0)
        self.dists = np.concatenate([self.dists, dist], axis=0)
        self.fov = np.concatenate([self.fov, fov], axis=0)
        if bbox_pix is not None:
            self.bbox_pix.append(bbox_pix)
        if og_image is not None:
            assert self.og_image is not None
            self.og_image = np.concatenate([self.og_image, og_image], axis=0)
        if label_source is not None:
            assert self.label_source is not None
            self.label_source = np.concatenate([self.label_source, label_source], axis=0)
        if sensor_id is not None:
            self.sensor_id.append(sensor_id)
        if mask is not None:
            assert self.masks is not None
            self.masks = np.concatenate([self.masks, mask], axis=0)
        if is_occluded is not None:
            self.is_occluded = np.concatenate([self.is_occluded, is_occluded], axis=0)
        if ray_map is not None:
            assert self.ray_map is not None
            self.ray_map = np.concatenate([self.ray_map, ray_map], axis=0)

    def convert(self) -> dict[str, Any]:
        """Convert to dictionary format for serialization."""
        sample: dict[str, Any] = {"__key__": f"{self.clip_id}_{self.obj_id}"}

        image_buffers = image_bytesio([self.frames[i] for i in range(self.frames.shape[0])], "jpeg")
        for i, buf in enumerate(image_buffers):
            sample[f"img_{i:03d}.jpeg"] = buf.read()

        for name, arr in [
            ("cam_poses", self.cam_poses),
            ("dists", self.dists),
            ("fov", self.fov),
            ("bbox_pos", self.bbox_pos),
        ]:
            byte_io = io.BytesIO()
            np.save(byte_io, arr)
            byte_io.seek(0)
            sample[f"{name}.npy"] = byte_io.read()

        sample["category.txt"] = self.npct.encode("utf8")

        if self.lwh is not None:
            byte_io = io.BytesIO()
            np.save(byte_io, self.lwh)
            byte_io.seek(0)
            sample["lwh.npy"] = byte_io.read()

        if len(self.sensor_id) > 0:
            sample["sensor_id.txt"] = "\n".join(self.sensor_id).encode("utf8")

        if self.label_source is not None:
            byte_io = io.BytesIO()
            np.save(byte_io, self.label_source)
            byte_io.seek(0)
            sample["label_source.npy"] = byte_io.read()

        if self.ray_map is not None:
            byte_io = io.BytesIO()
            np.save(byte_io, self.ray_map)
            byte_io.seek(0)
            sample["ray_map.npy"] = byte_io.read()

        return sample
