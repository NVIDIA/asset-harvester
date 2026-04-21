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

"""High-level ncore parser with segmentation-based view filtering."""

import logging

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .image_segmentation import Mask2FormerSegmentationEstimator
from .mvdata import MVData
from .ncore_object_parser import NCoreObjectParser
from .schemas import NCoreParserConfig

logger = logging.getLogger(__name__)


def select_bestfit_instances(
    instance_seg: dict[str, np.ndarray],
    h: int,
    w: int,
    bbox: np.ndarray,
    exceed_threshold: float = 0.3,
    overlap_threshold: float = 1e-5,
) -> tuple[np.ndarray, bool]:
    """Select best matching instance mask for target bounding box."""
    vertices = cv2.convexHull(bbox)
    mask_img = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask_img, vertices, 255)
    target_mask = mask_img > 0
    num_target_pixels = np.sum(target_mask)
    max_instance_id = 0
    max_iou = 0

    if len(instance_seg["classes"]) == 0:
        return np.zeros((h, w), dtype=np.uint8), False

    instance_masks = np.unpackbits(instance_seg["instance_masks"]).reshape(-1, h, w)

    for idx in range(len(instance_seg["classes"])):
        area_a = np.sum(instance_masks[idx] > 0)
        exceed = np.sum(~target_mask & (instance_masks[idx] > 0))

        if exceed > exceed_threshold * area_a:
            continue

        overlap = np.sum(target_mask & (instance_masks[idx] > 0))
        union = area_a + num_target_pixels - overlap
        iou = float(overlap) / union
        if iou > max_iou:
            max_iou = iou
            max_instance_id = idx

    valid = max_iou > overlap_threshold
    return instance_masks[max_instance_id], valid


class NCoreParser:
    """Parse ncore inputs into filtered multi-view samples."""

    def __init__(self, config: NCoreParserConfig) -> None:
        logger.info("Initializing NCoreParser")
        self.config = config

        self.ncore_extractor = NCoreObjectParser(
            target_resolution=config.target_resolution,
            num_lidar_ref_frames=config.num_lidar_ref_frames,
            cam_pose_flip=config.cam_pose_flip,
            occ_rate_threshold=config.occ_rate_threshold,
            crop_min_area_ratio=config.crop_min_area_ratio,
            max_threads=config.max_threads,
        )

        self.segmentation_estimator = Mask2FormerSegmentationEstimator(model_path=config.segmentation_ckpt)
        logger.info("NCoreParser initialized")

    def extract(
        self,
        src_data_paths: list[str],
        target_root_path: str,
        target_track_ids: list[str] | None,
        camera_ids: list[str],
    ) -> dict[str, MVData]:
        mvdata_sets, occ_rate_dict = self.ncore_extractor.extract(src_data_paths, target_track_ids, camera_ids)

        for u_id, mvdata in mvdata_sets.items():
            mvdata.is_occluded = np.zeros(len(mvdata.frames))
            occ_rates = occ_rate_dict[u_id]
            sorted_idx = np.argsort(occ_rates)

            if occ_rates[sorted_idx[0]] == 0:
                for idx in sorted_idx:
                    if occ_rates[idx] > 0:
                        mvdata.is_occluded[idx] = True
            else:
                for idx in sorted_idx[4:]:
                    mvdata.is_occluded[idx] = True

        valid_instances_mask_dict = {}
        for u_id, mvdata in tqdm(mvdata_sets.items(), desc="Generating masks"):
            masks_instance, ori_instance_seg, valid_instances = self._generate_masks(mvdata)
            mvdata.masks_instance = masks_instance
            mvdata.ori_instance_seg = ori_instance_seg
            valid_instances_mask_dict[u_id] = valid_instances

        good_samples = self._filter_samples(mvdata_sets, valid_instances_mask_dict)
        logger.info("Generated %s good samples out of %s", len(good_samples), len(mvdata_sets))

        return good_samples

    def _generate_masks(self, mvdata: MVData):
        masks_instance = []
        ori_instance_seg = []
        valid_instances = []

        for img, target_bbox in zip(mvdata.frames, mvdata.bbox_pix):
            img_pil = Image.fromarray(img)
            semantic_seg, instance_seg = self.segmentation_estimator.predict(img_pil)
            w, h = img_pil.size
            ori_instance_seg.append(instance_seg["instance_masks"])

            instance_mask, valid = select_bestfit_instances(
                instance_seg,
                h,
                w,
                target_bbox,
                exceed_threshold=self.config.mask_exceed_threshold,
                overlap_threshold=self.config.mask_overlap_threshold,
            )

            valid_instances.append(1 if np.sum(instance_mask) > self.config.min_instance_pixels and valid else 0)
            masks_instance.append(instance_mask)

        return masks_instance, ori_instance_seg, valid_instances

    def _filter_samples(self, mvdata_sets, valid_instances_mask_dict) -> dict[str, MVData]:
        good_samples: dict[str, MVData] = {}

        for u_id, mvdata in tqdm(mvdata_sets.items(), desc="Filtering"):
            is_occluded = mvdata.is_occluded if mvdata.is_occluded is not None else np.zeros(len(mvdata.frames))

            new_frames: list[np.ndarray] = []
            new_cam_poses: list[np.ndarray] = []
            new_dists: list[np.ndarray] = []
            new_fov: list[np.ndarray] = []
            new_sensor_ids: list[str] = []
            new_masks: list[np.ndarray] = []

            for idx in range(len(mvdata.frames)):
                valid = valid_instances_mask_dict[u_id][idx]
                if is_occluded[idx] > 0:
                    valid = False

                if mvdata.masks_instance and idx < len(mvdata.masks_instance):
                    target_mask = mvdata.masks_instance[idx]
                    H, W = target_mask.shape
                    y, x = target_mask.nonzero()

                    if target_mask.sum() < self.config.min_instance_pixels:
                        valid = False
                    if len(y) > 0 and ((y.mean() / H - 0.5) ** 2 + (x.mean() / W - 0.5) ** 2 > 0.25**2):
                        valid = False
                else:
                    valid = False

                if valid:
                    new_frames.append(mvdata.frames[idx])
                    new_cam_poses.append(mvdata.cam_poses[idx])
                    new_dists.append(mvdata.dists[idx])
                    new_fov.append(mvdata.fov[idx])
                    new_sensor_ids.append(mvdata.sensor_id[idx])
                    new_masks.append(mvdata.masks_instance[idx])

            if len(new_frames) >= 1:
                good_samples[u_id] = MVData(
                    clip_id=mvdata.clip_id,
                    obj_id=mvdata.obj_id,
                    frames=np.array(new_frames),
                    cam_poses=np.array(new_cam_poses),
                    dists=np.array(new_dists),
                    fov=np.array(new_fov),
                    npct=mvdata.npct,
                    bbox_pos=mvdata.bbox_pos,
                    bbox_pix=mvdata.bbox_pix,
                    lwh=mvdata.lwh,
                    sensor_id=new_sensor_ids,
                    masks_instance=new_masks,
                )

        return good_samples

    def cleanup(self) -> None:
        if hasattr(self, "segmentation_estimator"):
            self.segmentation_estimator.cleanup()
            del self.segmentation_estimator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
