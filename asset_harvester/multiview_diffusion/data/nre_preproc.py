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

import copy
import io
import random
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from random import shuffle
from statistics import NormalDist

import cv2
import numpy as np
import torch
from packaging import version as pver
from PIL import Image
from torchvision.transforms.functional import hflip

from ..utils.misc import AttrDict

if not hasattr(np, "atan"):
    np.atan = np.arctan

from .inference_utils import build_eval_cams


def image_bytesio(images, enc):
    """
    Convert images to a BytesIO buffer with enc encoding. Images can be either a list of PIL.Image objects or a
    np.ndarray of shape NHWC
    :param images:
    :param enc:
    :return:
    """
    bufs = []
    for image in images:
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                im = Image.fromarray((255 * image).astype(np.uint8))
            else:
                im = Image.fromarray(image)
        else:
            assert isinstance(image, Image.Image)
            im = image
        buf = io.BytesIO()
        im.save(buf, format=enc)
        buf.seek(0)
        bufs.append(buf)
    return bufs


def pil_to_arrays(images):
    arrs = []
    for image in images:
        im = np.array(image)
        arrs.append(im)
    return np.stack(arrs, axis=0)


def arrays_to_pils(arrays: np.ndarray):
    if arrays.ndim == 3:  # H W C
        images = [
            Image.fromarray((255 * arrays).astype(np.uint8)),
        ]
        return images
    elif arrays.ndim == 4:  # N H W C
        images = []
        for arr in arrays:
            im = Image.fromarray((255 * arr).astype(np.uint8))
            images.append(im)

        return images
    else:
        raise ValueError(f"input must be of dim 3 or 4, but got {arrays.ndim}")


@dataclass
class MVData:
    """
    Images and metadata of an object in a clip
    """

    # identifiers:
    clip_id: str
    obj_id: str
    # properties from raw data:
    frames: [np.ndarray, list]  # (N H W C)
    cam_poses: np.ndarray  # (N 3)
    dists: np.ndarray  # (N)
    fov: np.ndarray
    npct: str  # vehicle category
    bbox_pos: [np.ndarray, None] = field(default=None)  # bbox pose relative to cam (rays_
    bbox_pix: [np.ndarray, None] = field(default=None)  # bbox in image space
    lwh: [np.ndarray, None] = field(default=None)  # dimension of vehicle in metric space
    og_image: [np.ndarray, list, None] = field(default=None)  # (N H W C)
    label_source: [np.ndarray, None] = field(default=None)  # (N) int label
    sensor_id: list[str] = field(default_factory=list)  # N camera source tag
    timestamp: list[int] = field(default_factory=list)  # timestamp in global datetime
    # properties to be added in post-proc
    ray_map: [np.ndarray, None] = field(default=None)
    caption: list[str] = field(default_factory=list)  # prompt descriptor
    masks: [np.ndarray, list, None] = field(default=None)  # FG mask
    dist_matrix_masked: [np.ndarray, None] = field(default=None)
    ori_instance_seg: [list[np.ndarray], None] = field(default=None)  # instance segmentation mask [ I x W x H]
    bbox_masks: [np.ndarray, None] = field(default=None)  # (N H W) projected 2D bbox mask
    is_occluded: [np.ndarray, None] = field(default=None)  # label for if its occluded
    auto_label: [np.ndarray, None] = field(default=None)  # filter label predicted by CLIP+MLP classifier
    # synthetic rendering under random cameras
    rdcam_frames: [np.ndarray, list, None] = field(default=None)  # (N H W C)
    rdcam_masks: [np.ndarray, list, None] = field(default=None)  # FG mask
    rdcam_cam_poses: [np.ndarray, None] = field(default=None)
    rdcam_dists: [np.ndarray, None] = field(default=None)
    rdcam_fov: [np.ndarray, None] = field(default=None)
    # skeleton tracking results
    skeleton_2d_center: [np.ndarray, list, None] = field(default=None)  # (N H W C) rendered keypoints for center track
    skeleton_2d_all: [np.ndarray, list, None] = field(default=None)  # (N H W C) rendered keypoints for all tracks
    skeleton_3d_center: [np.ndarray, None] = field(default=None)  # (N param_dim) SMPL params for center track
    skeleton_3d_all: [np.ndarray, None] = field(default=None)  # (num_tracks N param_dim) SMPL params for all tracks

    def append(
        self,
        frames: [np.ndarray, list[Image.Image]],
        cam_poses: np.ndarray,
        dists: np.ndarray,
        fov: np.ndarray,
        og_image: [np.ndarray, list[Image.Image], None] = None,
        label_source: [np.ndarray, None] = None,
        sensor_id: [list[str], None] = None,
        timestamp: [list[int], None] = None,
        ray_map: [np.ndarray, None] = None,
        masks: [np.ndarray, list[Image.Image], None] = None,
        dist_matrix_masked: [np.ndarray, None] = None,
        ori_instance_seg: [list[np.ndarray], None] = None,
        is_occluded: [np.ndarray, None] = None,
        auto_label: [np.ndarray, None] = None,
        bbox_masks=None,
        skeleton_2d_center: [np.ndarray, list[Image.Image], None] = None,
        skeleton_2d_all: [np.ndarray, list[Image.Image], None] = None,
        skeleton_3d_center: [np.ndarray, None] = None,
        skeleton_3d_all: [np.ndarray, None] = None,
    ):
        """
        Behavior of this function is actually more akin to extend. Any ndarray-typed field in this dataclass require
        corresponding argument with broadcastable dimensions. Any list-typed fields require list-typed arguments
        :param frames:
        :param cam_poses:
        :param dists:
        :param fov:
        :param og_image:
        :param label_source:
        :param sensor_id:
        :param timestamp:
        :param ray_map:
        :param masks:
        :param dist_matrix_masked:
        :param ori_instance_seg:
        :param is_occluded:
        :param auto_label:
        :return:
        """
        if isinstance(self.frames, list):
            if isinstance(frames[0], Image.Image):
                im_frame = frames
            elif isinstance(frames, np.ndarray):
                im_frame = arrays_to_pils(frames)
            else:
                raise TypeError(f"frame must be of type ndarray or frame but got {type(frames).__name__}")
            self.frames = self.frames + im_frame
        else:
            if isinstance(frames[0], Image.Image):
                np_frame = pil_to_arrays(frames)
            elif isinstance(frames, np.ndarray):
                np_frame = frames
            else:
                raise TypeError(f"frame must be of type ndarray or frame but got {type(frames).__name__}")
            self.frames = np.concatenate([self.frames, np_frame], axis=0)
        self.cam_poses = np.concatenate([self.cam_poses, cam_poses], axis=0)
        self.dists = np.concatenate([self.dists, dists], axis=0)
        self.fov = np.concatenate([self.fov, fov], axis=0)

        if bbox_masks is not None:
            self.bbox_masks = np.concatenate([self.bbox_masks, bbox_masks], axis=0)

        if og_image is not None:
            assert self.og_image is not None
            if isinstance(self.og_image, list):
                if isinstance(og_image[0], Image.Image):
                    im_og_image = og_image
                elif isinstance(og_image, np.ndarray):
                    im_og_image = arrays_to_pils(og_image)
                else:
                    raise TypeError(f"frame must be of type ndarray or frame but got {type(og_image).__name__}")
                self.og_image = self.og_image + im_og_image
            else:
                if isinstance(og_image[0], Image.Image):
                    np_og_image = pil_to_arrays(og_image)
                elif isinstance(og_image, np.ndarray):
                    np_og_image = og_image
                else:
                    raise TypeError(f"frame must be of type ndarray or frame but got {type(og_image).__name__}")
                self.og_image = np.concatenate([self.og_image, np_og_image], axis=0)
        if label_source is not None:
            assert self.label_source is not None
            self.label_source = np.concatenate([self.label_source, label_source], axis=0)
        if sensor_id is not None:
            assert self.sensor_id is not None
            self.sensor_id += sensor_id
        if timestamp is not None:
            assert self.timestamp is not None
            self.timestamp += timestamp
        if ray_map is not None:
            assert self.ray_map is not None
            self.ray_map = np.concatenate([self.ray_map, ray_map], axis=0)
        if masks is not None:
            assert self.masks is not None
            if isinstance(self.masks, list):
                if isinstance(masks[0], Image.Image):
                    im_mask = masks
                elif isinstance(masks, np.ndarray):
                    im_mask = arrays_to_pils(masks)
                else:
                    raise TypeError(f"frame must be of type ndarray or frame but got {type(masks).__name__}")
                self.masks = self.masks + im_mask
            else:
                if isinstance(masks[0], Image.Image):
                    np_mask = pil_to_arrays(masks)
                elif isinstance(masks, np.ndarray):
                    np_mask = masks
                else:
                    raise TypeError(f"frame must be of type ndarray or frame but got {type(masks).__name__}")
                self.masks = np.concatenate([self.masks, np_mask], axis=0)
        if dist_matrix_masked is not None:
            warnings.warn("appending dist_matrix_masked is not implemented and has no effect")
            # assert self.dist_matrix_masked is not None
            # self.dist_matrix_masked = np.concatenate([self.dist_matrix_masked, dist_matrix_masked], axis=0)
        if ori_instance_seg is not None:
            assert self.ori_instance_seg is not None
            self.ori_instance_seg = self.ori_instance_seg + ori_instance_seg
        if is_occluded is not None:
            self.is_occluded = np.concatenate([self.is_occluded, is_occluded], axis=0)
        if auto_label is not None:
            self.auto_label = np.concatenate([self.auto_label, auto_label], axis=0)

        if skeleton_2d_center is not None:
            assert self.skeleton_2d_center is not None
            if isinstance(self.skeleton_2d_center, list):
                if isinstance(skeleton_2d_center[0], Image.Image):
                    im_skeleton_2d_center = skeleton_2d_center
                elif isinstance(skeleton_2d_center, np.ndarray):
                    im_skeleton_2d_center = arrays_to_pils(skeleton_2d_center)
                else:
                    raise TypeError(
                        f"skeleton_2d_center must be of type ndarray or frame but got {type(skeleton_2d_center).__name__}"
                    )
                self.skeleton_2d_center = self.skeleton_2d_center + im_skeleton_2d_center
            else:
                if isinstance(skeleton_2d_center[0], Image.Image):
                    np_skeleton_2d_center = pil_to_arrays(skeleton_2d_center)
                elif isinstance(skeleton_2d_center, np.ndarray):
                    np_skeleton_2d_center = skeleton_2d_center
                else:
                    raise TypeError(
                        f"skeleton_2d_center must be of type ndarray or frame but got {type(skeleton_2d_center).__name__}"
                    )
                self.skeleton_2d_center = np.concatenate([self.skeleton_2d_center, np_skeleton_2d_center], axis=0)

        if skeleton_2d_all is not None:
            assert self.skeleton_2d_all is not None
            if isinstance(self.skeleton_2d_all, list):
                if isinstance(skeleton_2d_all[0], Image.Image):
                    im_skeleton_2d_all = skeleton_2d_all
                elif isinstance(skeleton_2d_all, np.ndarray):
                    im_skeleton_2d_all = arrays_to_pils(skeleton_2d_all)
                else:
                    raise TypeError(
                        f"skeleton_2d_all must be of type ndarray or frame but got {type(skeleton_2d_all).__name__}"
                    )
                self.skeleton_2d_all = self.skeleton_2d_all + im_skeleton_2d_all
            else:
                if isinstance(skeleton_2d_all[0], Image.Image):
                    np_skeleton_2d_all = pil_to_arrays(skeleton_2d_all)
                elif isinstance(skeleton_2d_all, np.ndarray):
                    np_skeleton_2d_all = skeleton_2d_all
                else:
                    raise TypeError(
                        f"skeleton_2d_all must be of type ndarray or frame but got {type(skeleton_2d_all).__name__}"
                    )
                self.skeleton_2d_all = np.concatenate([self.skeleton_2d_all, np_skeleton_2d_all], axis=0)

        if skeleton_3d_center is not None:
            assert self.skeleton_3d_center is not None
            self.skeleton_3d_center = np.concatenate([self.skeleton_3d_center, skeleton_3d_center], axis=0)

        if skeleton_3d_all is not None:
            assert self.skeleton_3d_all is not None
            self.skeleton_3d_all = np.concatenate([self.skeleton_3d_all, skeleton_3d_all], axis=0)

    def pop(self, idx):
        if isinstance(self.frames, list):
            frame = [self.frames.pop(idx)]
        else:
            frame = self.frames[idx]
            self.frames = np.delete(self.frames, idx, axis=0)
        cam_pose = self.cam_poses[idx]
        self.cam_poses = np.delete(self.cam_poses, idx, axis=0)
        dist = self.dists[idx]
        self.dists = np.delete(self.dists, idx, axis=0)
        fov = self.fov[idx]

        self.fov = np.delete(self.fov, idx, axis=0)
        ret = {"frames": frame, "cam_poses": cam_pose, "dists": dist, "fov": fov}

        if self.og_image is not None:
            if isinstance(self.og_image, list):
                ret["og_image"] = [self.og_image.pop(idx)]
            else:
                ret["og_image"] = self.og_image[idx]
                self.og_image = np.delete(self.og_image, idx, axis=0)

        if self.label_source is not None:
            ret["label_source"] = self.label_source[idx]
            self.label_source = np.delete(self.label_source, idx, axis=0)

        if self.sensor_id is not None and len(self.sensor_id) > 0:
            ret["sensor_id"] = self.sensor_id.pop(idx)

        if self.timestamp is not None and len(self.timestamp) > 0:
            ret["timestamp"] = self.timestamp.pop(idx)

        if self.ray_map is not None:
            ret["ray_map"] = self.ray_map[idx]  # Nx32x32x3
            self.ray_map = np.delete(self.ray_map, idx, axis=0)

        if self.masks is not None:
            if isinstance(self.masks, list):
                ret["masks"] = [self.masks.pop(idx)]
            else:
                ret["masks"] = self.masks[idx]
                self.masks = np.delete(self.masks, idx, axis=0)

        if self.bbox_masks is not None:
            if isinstance(self.bbox_masks, list):
                ret["bbox_masks"] = [self.bbox_masks.pop(idx)]
            else:
                ret["bbox_masks"] = self.bbox_masks[idx]
                self.bbox_masks = np.delete(self.bbox_masks, idx, axis=0)

        if self.dist_matrix_masked is not None:
            ret["dist_matrix_masked"] = np.array([0.0])
            self.dist_matrix_masked = np.delete(self.dist_matrix_masked, idx, axis=0)
            self.dist_matrix_masked = np.delete(self.dist_matrix_masked, idx, axis=1)

        if self.ori_instance_seg is not None:
            ori_instance_seg = [self.ori_instance_seg.pop(idx)]
            ret["ori_instance_seg"] = ori_instance_seg

        if self.is_occluded is not None:
            occ = self.is_occluded[idx]
            ret["is_occluded"] = occ
            self.is_occluded = np.delete(self.is_occluded, idx, axis=0)

        if self.auto_label is not None:
            ret["auto_label"] = self.auto_label[idx]
            self.auto_label = np.delete(self.auto_label, idx, axis=0)

        if self.skeleton_2d_center is not None:
            if isinstance(self.skeleton_2d_center, list):
                ret["skeleton_2d_center"] = [self.skeleton_2d_center.pop(idx)]
            else:
                ret["skeleton_2d_center"] = self.skeleton_2d_center[idx]
                self.skeleton_2d_center = np.delete(self.skeleton_2d_center, idx, axis=0)

        if self.skeleton_2d_all is not None:
            if isinstance(self.skeleton_2d_all, list):
                ret["skeleton_2d_all"] = [self.skeleton_2d_all.pop(idx)]
            else:
                ret["skeleton_2d_all"] = self.skeleton_2d_all[idx]
                self.skeleton_2d_all = np.delete(self.skeleton_2d_all, idx, axis=0)

        if self.skeleton_3d_center is not None:
            ret["skeleton_3d_center"] = self.skeleton_3d_center[idx]
            self.skeleton_3d_center = np.delete(self.skeleton_3d_center, idx, axis=0)

        if self.skeleton_3d_all is not None:
            ret["skeleton_3d_all"] = self.skeleton_3d_all[idx]
            self.skeleton_3d_all = np.delete(self.skeleton_3d_all, idx, axis=0)

        return ret

    def __len__(self):
        return len(self.frames)

    def _get(self, idx):
        assert 0 <= idx < len(self)
        if isinstance(self.frames, np.ndarray):
            frame = self.frames[idx : idx + 1]
        else:
            frame = [self.frames[idx]]
        cam_pose = self.cam_poses[idx : idx + 1]
        dist = self.dists[idx : idx + 1]
        fov = self.fov[idx : idx + 1]
        ret = {"frames": frame, "cam_poses": cam_pose, "dists": dist, "fov": fov}

        if self.og_image is not None:
            if isinstance(self.og_image, np.ndarray):
                ret["og_image"] = self.og_image[idx : idx + 1]
            else:
                ret["og_image"] = [self.og_image[idx]]

        if self.label_source is not None:
            ret["label_source"] = self.label_source[idx : idx + 1]

        if self.sensor_id is not None and len(self.sensor_id) > 0:
            ret["sensor_id"] = self.sensor_id[idx : idx + 1]

        if self.timestamp is not None and len(self.timestamp) > 0:
            ret["timestamp"] = self.timestamp[idx : idx + 1]

        if self.ray_map is not None:
            ret["ray_map"] = self.ray_map[idx : idx + 1]  # Nx32x32x3

        if self.masks is not None:
            if isinstance(self.masks, np.ndarray):
                ret["masks"] = self.masks[idx : idx + 1]
            else:
                ret["masks"] = [self.masks[idx]]

        if self.bbox_masks is not None:
            if isinstance(self.bbox_masks, np.ndarray):
                ret["bbox_masks"] = self.bbox_masks[idx : idx + 1]
            else:
                ret["bbox_masks"] = [self.bbox_masks[idx]]

        if self.dist_matrix_masked is not None:
            ret["dist_matrix_masked"] = np.array(
                [
                    0.0,
                ]
            )

        if self.ori_instance_seg is not None:
            ori_instance_seg = [self.ori_instance_seg[idx]]
            ret["ori_instance_seg"] = ori_instance_seg

        if self.is_occluded is not None:
            occ = self.is_occluded[idx : idx + 1]
            ret["is_occluded"] = occ

        if self.auto_label is not None:
            ret["auto_label"] = self.auto_label[idx : idx + 1]

        if self.rdcam_frames is not None:
            if isinstance(self.rdcam_frames, np.ndarray):
                ret["rdcam_frames"] = self.rdcam_frames[idx : idx + 1]
            else:
                ret["rdcam_frames"] = [self.rdcam_frames[idx]]
            if isinstance(self.rdcam_masks, np.ndarray):
                ret["rdcam_masks"] = self.rdcam_masks[idx : idx + 1]
            else:
                ret["rdcam_masks"] = [self.rdcam_masks[idx]]
            ret["rdcam_cam_poses"] = self.rdcam_cam_poses
            ret["rdcam_dists"] = self.rdcam_dists
            ret["rdcam_fov"] = self.rdcam_fov

        if self.skeleton_2d_center is not None:
            if isinstance(self.skeleton_2d_center, np.ndarray):
                ret["skeleton_2d_center"] = self.skeleton_2d_center[idx : idx + 1]
            else:
                ret["skeleton_2d_center"] = [self.skeleton_2d_center[idx]]

        if self.skeleton_2d_all is not None:
            if isinstance(self.skeleton_2d_all, np.ndarray):
                ret["skeleton_2d_all"] = self.skeleton_2d_all[idx : idx + 1]
            else:
                ret["skeleton_2d_all"] = [self.skeleton_2d_all[idx]]

        if self.skeleton_3d_center is not None:
            ret["skeleton_3d_center"] = self.skeleton_3d_center[idx : idx + 1]

        if self.skeleton_3d_all is not None:
            ret["skeleton_3d_all"] = self.skeleton_3d_all[idx : idx + 1]

        return ret

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if idx < 0:
                idx = len(self) + idx
            rets = self._get(idx)
            return MVData(
                self.clip_id,
                self.obj_id,
                rets.pop("frames"),
                rets.pop("cam_poses"),
                rets.pop("dists"),
                rets.pop("fov"),
                npct=self.npct,
                caption=self.caption,
                **rets,
                bbox_pix=self.bbox_pix,
                bbox_pos=self.bbox_pos,
                lwh=self.lwh,
            )
        elif isinstance(idx, list):
            if len(idx) == 0:
                return None
            ret = self._get(idx[0])
            ret = MVData(
                self.clip_id,
                self.obj_id,
                ret.pop("frames"),
                ret.pop("cam_poses"),
                ret.pop("dists"),
                ret.pop("fov"),
                npct=self.npct,
                caption=self.caption,
                bbox_pix=self.bbox_pix,
                bbox_pos=self.bbox_pos,
                lwh=self.lwh,
                **ret,
            )
            for i in range(1, len(idx)):
                vals = self._get(idx[i])
                ret.append(vals.pop("frames"), vals.pop("cam_poses"), vals.pop("dists"), vals.pop("fov"), **vals)
            return ret
        elif isinstance(idx, slice):
            # Get the start, stop, and step from the slice
            interped_idx = [ii for ii in range(*idx.indices(len(self)))]
            if len(interped_idx) == 0:
                return None
            ret = self._get(interped_idx[0])
            ret = MVData(
                self.clip_id,
                self.obj_id,
                ret.pop("frames"),
                ret.pop("cam_poses"),
                ret.pop("dists"),
                ret.pop("fov"),
                npct=self.npct,
                caption=self.caption,
                bbox_pix=self.bbox_pix,
                bbox_pos=self.bbox_pos,
                lwh=self.lwh,
                **ret,
            )
            for i in range(1, len(interped_idx)):
                vals = self._get(interped_idx[i])
                ret.append(vals.pop("frames"), vals.pop("cam_poses"), vals.pop("dists"), vals.pop("fov"), **vals)
            return ret
        else:
            raise TypeError(
                f"{type(self).__name__} indices must be integers, list, or slices, not {type(idx).__name__}"
            )

    @staticmethod
    def _decode_image(sample, frame_as_array):
        frames = []
        i = 0
        while "img_%03d.jpeg" % i in sample:
            with io.BytesIO(sample["img_%03d.jpeg" % i]) as stream:
                img = Image.open(stream)
                img.load()
            frames.append(img)
            i += 1

        og_frames = []
        i = 0
        while "og_img_%03d.jpeg" % i in sample:
            with io.BytesIO(sample["og_img_%03d.jpeg" % i]) as stream:
                img = Image.open(stream)
                img.load()
            og_frames.append(img)
            i += 1
        if len(og_frames) > 0:
            pass
        else:
            og_frames = None

        masks = []
        i = 0
        while "inst_mask_img_%03d.png" % (i) in sample:
            with io.BytesIO(sample["inst_mask_img_%03d.png" % (i)]) as stream:
                img = Image.open(stream)
                img.load()
            masks.append(img)
            i += 1
        if len(masks) > 0:
            assert len(masks) == len(frames)
        else:
            masks = None

        rdcam_frames = []
        i = 0
        while "rdcam_img_%03d.jpeg" % (i) in sample:
            with io.BytesIO(sample["rdcam_img_%03d.jpeg" % (i)]) as stream:
                img = Image.open(stream)
                img.load()
            rdcam_frames.append(img)
            i += 1

        if len(rdcam_frames) > 0:
            pass
        else:
            rdcam_frames = None

        rdcam_masks = []
        j = 0
        while "rdmask_inst_mask_img_%03d.png" % (j) in sample:
            with io.BytesIO(sample["rdmask_inst_mask_img_%03d.png" % (j)]) as stream:
                img = Image.open(stream)
                img.load()
            rdcam_masks.append(img)
            j += 1

        if len(rdcam_masks) > 0:
            pass
        else:
            rdcam_masks = None

        bbox_masks = []
        j = 0
        while "bbox_mask_img_%03d.png" % (j) in sample:
            with io.BytesIO(sample["bbox_mask_img_%03d.png" % (j)]) as stream:
                img = Image.open(stream)
                img.load()
            bbox_masks.append(img)

        if len(bbox_masks) > 0:
            pass
        else:
            bbox_masks = None

        skeleton_2d_center = []
        i = 0
        while "skeleton_2d_center_%03d.jpeg" % i in sample:
            with io.BytesIO(sample["skeleton_2d_center_%03d.jpeg" % i]) as stream:
                img = Image.open(stream)
                img.load()
            skeleton_2d_center.append(img)
            i += 1
        if len(skeleton_2d_center) > 0:
            pass
        else:
            skeleton_2d_center = None

        skeleton_2d_all = []
        i = 0
        while "skeleton_2d_all_%03d.jpeg" % i in sample:
            with io.BytesIO(sample["skeleton_2d_all_%03d.jpeg" % i]) as stream:
                img = Image.open(stream)
                img.load()
            skeleton_2d_all.append(img)
            i += 1
        if len(skeleton_2d_all) > 0:
            pass
        else:
            skeleton_2d_all = None

        if frame_as_array:
            frames = pil_to_arrays(frames)
            if og_frames is not None:
                og_frames = pil_to_arrays(og_frames)
            if masks is not None:
                masks = pil_to_arrays(masks)
            if rdcam_frames is not None:
                rdcam_frames = pil_to_arrays(rdcam_frames)
            if rdcam_masks is not None:
                rdcam_masks = pil_to_arrays(rdcam_masks)
            if bbox_masks is not None:
                bbox_masks = pil_to_arrays(bbox_masks)
            if skeleton_2d_center is not None:
                skeleton_2d_center = pil_to_arrays(skeleton_2d_center)
            if skeleton_2d_all is not None:
                skeleton_2d_all = pil_to_arrays(skeleton_2d_all)
        return frames, og_frames, masks, rdcam_frames, rdcam_masks, bbox_masks, skeleton_2d_center, skeleton_2d_all

    @classmethod
    def decode(cls, sample, frame_as_array=False):
        """
        :param sample:
        :param frame_as_array: if true, convert image-type data to numpy array
        :return:
        """

        # Mandatory Per-sample attributes:
        segs = sample["__key__"].split("_")
        clip_id = "_".join(segs[:-1])
        obj_id = segs[-1]
        try:
            npct = sample["category.txt"].decode("utf-8")
        except (AttributeError, KeyError, UnicodeDecodeError):
            npct = "vehicle"
        # Image-typed attribtues:
        frames, og_image, masks, rdcam_frames, rdcam_masks, bbox_masks, skeleton_2d_center, skeleton_2d_all = (
            MVData._decode_image(sample, frame_as_array)
        )

        # Mandatory per-frame attributes:
        with io.BytesIO(sample["cam_poses.npy"]) as stream:
            camposes = np.load(stream)
        with io.BytesIO(sample["dists.npy"]) as stream:
            dists = np.load(stream)
        with io.BytesIO(sample["fov.npy"]) as stream:
            fov = np.load(stream)

        # Optional attributes:
        bbox_pos = None
        if "bbox_pos.npy" in sample:
            with io.BytesIO(sample["bbox_pos.npy"]) as stream:
                bbox_pos = np.load(stream)

        bbox_pix = None
        if "bbox_pix.npy" in sample:
            with io.BytesIO(sample["bbox_pix.npy"]) as stream:
                bbox_pix = np.load(stream)

        lwh = None
        if "lwh.npy" in sample:
            with io.BytesIO(sample["lwh.npy"]) as stream:
                lwh = np.load(stream)

        label_source = None
        if "label_source.npy" in sample:
            with io.BytesIO(sample["label_source.npy"]) as stream:
                label_source = np.load(stream)

        sensor_id = []
        if "sensor_id.txt" in sample:
            sensor_id = list(filter(lambda x: len(x) > 0, sample["sensor_id.txt"].decode("utf-8").split("\n")))

        timestamp = []
        if "timestamp.txt" in sample:
            timestamp = list(
                map(int, filter(lambda x: len(x) > 0, sample["timestamp.txt"].decode("utf-8").split("\n")))
            )

        ray_map = None
        if "ray_map.npy" in sample:
            with io.BytesIO(sample["ray_map.npy"]) as stream:
                ray_map = np.load(stream)

        captions = []
        if "captions.txt" in sample:
            captions = sample["captions.txt"].decode("utf-8")

        dist_matrix_masked = None
        if "dist_matrix_masked.npy" in sample:
            with io.BytesIO(sample["dist_matrix_masked.npy"]) as stream:
                dist_matrix_masked = np.load(stream)

        # load original instance seg
        ori_instance_seg = None
        if "ori_instance_seg_%03d.npy" % (0) in sample:
            ori_instance_seg = []
            j = 0
            while "ori_instance_seg_%03d.npy" % (j) in sample:
                with io.BytesIO(sample["ori_instance_seg_%03d.npy" % (j)]) as stream:
                    if masks is not None:
                        if isinstance(masks, np.ndarray):
                            h, w = masks[0].shape
                        elif isinstance(masks[0], Image.Image):
                            h, w = masks[0].height, masks[0].width
                    else:
                        if isinstance(frames, np.ndarray):
                            h, w = frames.shape[1], frames.shape[2]
                        elif isinstance(frames[0], Image.Image):
                            h, w = frames[0].height, frames[0].width
                    ori_instance_seg.append(np.unpackbits(np.load(stream)).reshape(-1, h, w))
                j += 1
            assert len(ori_instance_seg) == len(frames)

        is_occluded = None
        if "is_occluded.npy" in sample:
            with io.BytesIO(sample["is_occluded.npy"]) as stream:
                is_occluded = np.load(stream)

        auto_label = None
        if "auto_label.npy" in sample:
            with io.BytesIO(sample["auto_label.npy"]) as stream:
                auto_label = np.load(stream)
        rdcam_cam_poses = None
        rdcam_dists = None
        rdcam_fov = None
        if rdcam_frames is not None:
            with io.BytesIO(sample["rdcam_cam_poses.npy"]) as stream:
                rdcam_cam_poses = np.load(stream)
            with io.BytesIO(sample["rdcam_dists.npy"]) as stream:
                rdcam_dists = np.load(stream)
            with io.BytesIO(sample["rdcam_fov.npy"]) as stream:
                rdcam_fov = np.load(stream)

        skeleton_3d_center = None
        if "skeleton_3d_center.npy" in sample:
            with io.BytesIO(sample["skeleton_3d_center.npy"]) as stream:
                skeleton_3d_center = np.load(stream, allow_pickle=True)

        skeleton_3d_all = None
        if "skeleton_3d_all.npy" in sample:
            with io.BytesIO(sample["skeleton_3d_all.npy"]) as stream:
                skeleton_3d_all = np.load(stream, allow_pickle=True)

        return cls(
            clip_id,
            obj_id,
            frames,
            camposes,
            dists,
            fov,
            npct,
            bbox_pos=bbox_pos,
            bbox_pix=bbox_pix,
            lwh=lwh,
            og_image=og_image,
            label_source=label_source,
            sensor_id=sensor_id,
            timestamp=timestamp,
            ray_map=ray_map,
            caption=captions,
            masks=masks,
            dist_matrix_masked=dist_matrix_masked,
            ori_instance_seg=ori_instance_seg,
            is_occluded=is_occluded,
            auto_label=auto_label,
            rdcam_frames=rdcam_frames,
            rdcam_masks=rdcam_masks,
            rdcam_cam_poses=rdcam_cam_poses,
            rdcam_dists=rdcam_dists,
            rdcam_fov=rdcam_fov,
            bbox_masks=bbox_masks,
            skeleton_2d_center=skeleton_2d_center,
            skeleton_2d_all=skeleton_2d_all,
            skeleton_3d_center=skeleton_3d_center,
            skeleton_3d_all=skeleton_3d_all,
        )

    def convert(self):
        sample = {"__key__": self.clip_id + "_" + str(self.obj_id)}
        image_buffers = image_bytesio(self.frames, "jpeg")
        for i, buf in enumerate(image_buffers):
            sample["img_%03d.jpeg" % i] = buf.read()

        byte_io = io.BytesIO()
        np.save(byte_io, self.cam_poses)
        byte_io.seek(0)
        sample["cam_poses.npy"] = byte_io.read()

        byte_io = io.BytesIO()
        np.save(byte_io, self.dists)
        byte_io.seek(0)
        sample["dists.npy"] = byte_io.read()

        byte_io = io.BytesIO()
        np.save(byte_io, self.fov)
        byte_io.seek(0)
        sample["fov.npy"] = byte_io.read()

        byte_io = io.BytesIO(self.npct.encode("utf8"))
        byte_io.seek(0)
        sample["category.txt"] = byte_io.read()

        if self.bbox_pos is not None:
            byte_io = io.BytesIO()
            np.save(byte_io, self.bbox_pos)
            byte_io.seek(0)
            sample["bbox_pos.npy"] = byte_io.read()

        if self.bbox_pix is not None:
            byte_io = io.BytesIO()
            np.save(byte_io, self.bbox_pix)
            byte_io.seek(0)
            sample["bbox_pix.npy"] = byte_io.read()

        if self.lwh is not None:
            byte_io = io.BytesIO()
            np.save(byte_io, self.lwh)
            byte_io.seek(0)
            sample["lwh.npy"] = byte_io.read()

        if self.og_image is not None:
            image_buffers = image_bytesio(self.og_image, "jpeg")
            for i, buf in enumerate(image_buffers):
                sample["og_img_%03d.jpeg" % i] = buf.read()

        if self.label_source is not None:
            byte_io = io.BytesIO()
            np.save(byte_io, self.label_source)
            byte_io.seek(0)
            sample["label_source.npy"] = byte_io.read()

        if len(self.sensor_id) > 0:
            strio = io.StringIO()
            sensor_id = self.sensor_id
            for sid in sensor_id:
                strio.write(sid + "\n")
            strio.seek(0)
            sample["sensor_id.txt"] = strio.read()

        if len(self.timestamp) > 0:
            strio = io.StringIO()
            timestamp = self.timestamp
            for sid in timestamp:
                strio.write(str(sid) + "\n")
            strio.seek(0)
            sample["timestamp.txt"] = strio.read()

        if self.ray_map is not None:
            byte_io = io.BytesIO()
            np.save(byte_io, self.ray_map)
            byte_io.seek(0)
            sample["ray_map.npy"] = byte_io.read()

        if len(self.caption) > 0:
            strio = io.StringIO()
            captions = self.caption
            for caption in captions:
                strio.write(caption + "\n")
            strio.seek(0)
            sample["captions.txt"] = strio.read()

        if self.masks is not None:
            mask_buffers = image_bytesio(self.masks, "png")
            for i, buf in enumerate(mask_buffers):
                sample["inst_mask_img_%03d.png" % (i)] = buf.read()

        if self.bbox_masks is not None:
            bbox_mask_buffers = image_bytesio(self.bbox_masks, "png")
            for i, buf in enumerate(bbox_mask_buffers):
                sample["bbox_mask_img_%03d.png" % i] = buf.read()

        if self.dist_matrix_masked is not None:
            byte_io = io.BytesIO()
            np.save(byte_io, self.dist_matrix_masked)
            byte_io.seek(0)
            sample["dist_matrix_masked.npy"] = byte_io.read()

        if self.ori_instance_seg is not None:
            for i in range(len(self.ori_instance_seg)):
                byte_io = io.BytesIO()
                np.save(byte_io, np.packbits(self.ori_instance_seg[i].flatten()))
                byte_io.seek(0)
                sample["ori_instance_seg_%03d.npy" % (i)] = byte_io.read()

        if self.is_occluded is not None:
            byte_io = io.BytesIO()
            np.save(byte_io, self.is_occluded)
            byte_io.seek(0)
            sample["is_occluded.npy"] = byte_io.read()

        if self.auto_label is not None:
            byte_io = io.BytesIO()
            np.save(byte_io, self.auto_label)
            byte_io.seek(0)
            sample["auto_label.npy"] = byte_io.read()

        if self.rdcam_frames is not None:
            image_buffers = image_bytesio(self.rdcam_frames, "jpeg")
            for i, buf in enumerate(image_buffers):
                sample["rdcam_img_%03d.jpeg" % i] = buf.read()

            mask_buffers = image_bytesio(self.rdcam_masks, "png")
            for i, buf in enumerate(mask_buffers):
                sample["rdmask_inst_mask_img_%03d.png" % (i)] = buf.read()

            byte_io = io.BytesIO()
            np.save(byte_io, self.rdcam_cam_poses)
            byte_io.seek(0)
            sample["rdcam_cam_poses.npy"] = byte_io.read()

            byte_io = io.BytesIO()
            np.save(byte_io, self.rdcam_dists)
            byte_io.seek(0)
            sample["rdcam_dists.npy"] = byte_io.read()

            byte_io = io.BytesIO()
            np.save(byte_io, self.rdcam_fov)
            byte_io.seek(0)
            sample["rdcam_fov.npy"] = byte_io.read()

        if self.skeleton_2d_center is not None:
            image_buffers = image_bytesio(self.skeleton_2d_center, "jpeg")
            for i, buf in enumerate(image_buffers):
                sample["skeleton_2d_center_%03d.jpeg" % i] = buf.read()

        if self.skeleton_2d_all is not None:
            image_buffers = image_bytesio(self.skeleton_2d_all, "jpeg")
            for i, buf in enumerate(image_buffers):
                sample["skeleton_2d_all_%03d.jpeg" % i] = buf.read()

        if self.skeleton_3d_center is not None:
            byte_io = io.BytesIO()
            np.save(byte_io, self.skeleton_3d_center)
            byte_io.seek(0)
            sample["skeleton_3d_center.npy"] = byte_io.read()

        if self.skeleton_3d_all is not None:
            byte_io = io.BytesIO()
            np.save(byte_io, self.skeleton_3d_all)
            byte_io.seek(0)
            sample["skeleton_3d_all.npy"] = byte_io.read()

        return sample


def decode_data(item, remove_occluded=True):
    data = MVData.decode(item, frame_as_array=True)
    if remove_occluded and data.is_occluded is not None:
        ids = []
        for i, occ in enumerate(data.is_occluded):
            if occ > 0:
                ids.append(i)
        for i in reversed(ids):
            data.pop(i)

    # remove images with empty instance masks or weight center of instance masks too far from image center
    if data.masks is not None:
        for i in reversed(list(range(data.masks.shape[0]))):
            if data.masks[i].sum() < 100:
                data.pop(i)

        for i in reversed(list(range(data.masks.shape[0]))):
            H, W = data.masks[i].shape
            y, x = data.masks[i].nonzero()
            if (y.mean() / float(H) - 0.5) ** 2 + (x.mean() / float(W) - 0.5) ** 2 > 0.25**2:
                data.pop(i)

    # remove images with more than 1 large instances (>  5% of image area)
    if data.ori_instance_seg is not None:
        for i in reversed(list(range(len(data.ori_instance_seg)))):
            N, H, W = data.ori_instance_seg[i].shape
            areas = np.sum(data.ori_instance_seg[i], axis=(1, 2))
            if np.sum(areas > H * W * 0.05) > 1:  # more than 1 instance
                data.pop(i)

    return data


def collate_fn(batch):
    "We forgo collating at the loader level since padding is done more efficiently after GPU-bound processing"
    return batch


def batched_collate_fn(batch):
    rets = {}
    for data in batch:
        for k, v in data.items():
            assert torch.is_tensor(v) or type(v) is list
            if k in rets:
                if torch.is_tensor(v):
                    rets[k] = torch.cat([rets[k], v], dim=0)
                else:
                    rets[k] += v
            else:
                rets[k] = v
    return AttrDict(rets)


def length_filter(item):
    if len(item.frames) < 2:  # need at least a input frame and a target frame
        return False
    return True


def campos_filter(item):
    max_dist = np.max(np.abs(np.max(item.cam_poses, axis=0) - np.min(item.cam_poses, axis=0)))
    if max_dist < 0.1:  # if all camera's x,y,z are within 0.2 of each other
        return False
    return True


def auto_label_filter(item: MVData):
    if item.auto_label is not None:
        if sum(item.auto_label) < 2:
            return False
    return True


def symmetry_augment(x: list, cam_pos: list, aug_p: float = 0.5):
    """
    Divides the images into bins based on camera location, randomly flip the poses in each bin along the Y dimension,
    and flip the image horizontally
    Args:
        x:
        cam_pos:
        aug_p:

    Returns:

    """
    x = copy.deepcopy(x)
    cam_pos_np = np.stack(cam_pos, axis=0)
    azimuth = np.arctan2(cam_pos_np[:, 0], cam_pos_np[:, 1])
    n_bins = 20
    bins = np.linspace(-np.pi, np.pi, n_bins)
    inds = np.digitize(azimuth, bins)
    for j in range(n_bins):
        do_mirror = np.random.rand(1) < aug_p
        if do_mirror:
            target_inds = np.arange(len(cam_pos), dtype=int)
            target_inds = target_inds[inds == j]
            for k in target_inds:
                cam_pos[k][1] = -cam_pos[k][1]
                x[k] = hflip(x[k])
    return x, cam_pos


def rig_to_open_cv(cam_pos) -> np.ndarray:
    """
    convert from campos coordinate to opencv world coordinate
    Args:
        cam_pos: array: [N, 3] or [3]

    Returns: array: [N, 3] or [3]

    """
    M = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
    if cam_pos.ndim == 2:
        return np.einsum("ni,ij->nj", cam_pos, M.T)
    else:
        return M @ cam_pos


def create_lookat_matrix(
    cam_pos: np.ndarray,
    target_pos: np.ndarray = np.array([0.0, 0.0, 0.0]),
    world_up: np.ndarray = np.array([0.0, -1.0, 0.0]),
) -> np.ndarray:
    """
    Create a camera-to-world transformation matrix from camera position and look-at point.
    Uses OpenCV coordinate convention (z forward, y down, x right).

    Args:
        cam_pos (np.ndarray): Camera position [x,y,z]
        target_pos (np.ndarray): Point the camera is looking at [x,y,z]. Default to the origin
        world_up (np.ndarray): Up direction of the world. Default to [0,-1,0] as reference (OpenCV y points down)
    Returns:
        np.ndarray: 4x4 homogeneous camera-to-world transformation matrix
    """
    # Convert inputs to numpy arrays if they aren't already
    camera_pos = np.array(cam_pos)
    target_pos = np.array(target_pos)

    # Calculate forward direction (z-axis)
    # In OpenCV, positive z points forward from camera
    forward = target_pos - camera_pos
    forward = forward / np.linalg.norm(forward)

    # Calculate right direction (x-axis)
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)

    # Calculate up direction (y-axis)
    up = np.cross(right, forward)
    down = -up
    # Create rotation matrix
    R = np.eye(4)
    R[0:3, 0] = right
    R[0:3, 1] = down
    R[0:3, 2] = forward

    # Create translation matrix
    T = np.eye(4)
    T[0:3, 3] = camera_pos

    # Combine rotation and translation
    # The camera-to-world matrix is T * R
    camera_to_world = T @ R

    return camera_to_world


def create_intrisics(fov: np.ndarray, H: int = 512, W: int = 512) -> np.ndarray:
    """
    Compute the fx,fy,cx,cy intrinsics array given fov and resolution parameters
    """
    fov = np.array(fov)
    fov_radians = np.deg2rad(fov)
    # For a pinhole camera:
    # focal_length = (resolution/2) / tan(fov/2)
    fx = (W / 2.0) / np.tan(fov_radians / 2.0)
    fy = (H / 2.0) / np.tan(fov_radians / 2.0)
    cx = W * np.ones_like(fov) / 2
    cy = H * np.ones_like(fov) / 2
    intrinsics = np.stack([fx, fy, cx, cy], axis=-1)
    return intrinsics


def transform_camera_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Convert camera-to-world matrix B to be expressed in camera A's coordinate space.

    Args:
        A (np.ndarray): 4x4 camera-to-world transformation matrix for camera A
        B (np.ndarray): 4x4 camera-to-world transformation matrix for camera B

    Returns:
        np.ndarray: 4x4 transformation matrix expressing B's pose relative to camera A's coordinate frame
    """
    # Get world-to-camera matrix for A by inverting it
    A_inv = np.linalg.inv(A)

    # Transform B into A's camera space by pre-multiplying with A's inverse
    # This gives us B's pose relative to camera A
    B_in_A = A_inv @ B

    return B_in_A


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse("1.10"):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing="ij")


def ray_condition(K, c2w, H, W, device, plucker_scene_scale=1.0, flip_flag=None):
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B, V = K.shape[:2]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5  # [B, V, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5  # [B, V, HxW]

    n_flip = torch.sum(flip_flag).item() if flip_flag is not None else 0
    if n_flip > 0:
        j_flip, i_flip = custom_meshgrid(
            torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
            torch.linspace(W - 1, 0, W, device=device, dtype=c2w.dtype),
        )
        i_flip = i_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        j_flip = j_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        i[:, flip_flag, ...] = i_flip
        j[:, flip_flag, ...] = j_flip

    fx, fy, cx, cy = K.chunk(4, dim=-1)  # B,V, 1

    zs = torch.ones_like(i)  # [B, V, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)  # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)  # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)  # B, V, HW, 3
    rays_o = c2w[..., :3, 3] / plucker_scene_scale  # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)  # B, V, HW, 3
    # c2w @ dirctions
    rays_dxo = torch.cross(rays_o, rays_d)  # B, V, HW, 3
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)  # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker.permute(0, 1, 4, 2, 3)


def zoom_image(image, zoom_factor):
    """
    Zoom in or out on an image based on a zoom factor.

    Args:
        image (numpy.ndarray): Input image
        zoom_factor (float): Zoom factor. > 1.0 for zoom in, < 1.0 for zoom out

    Returns:
        numpy.ndarray: Zoomed image
    """
    if zoom_factor == 1.0:
        return image.copy()

    height, width = image.shape[:2]

    # Calculate new dimensions
    new_height = int(height * zoom_factor)
    new_width = int(width * zoom_factor)

    # Resize the image
    zoomed_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # If zooming in, crop the center portion to match original dimensions
    if zoom_factor > 1.0:
        y_start = (new_height - height) // 2
        y_end = y_start + height
        x_start = (new_width - width) // 2
        x_end = x_start + width
        zoomed_image = zoomed_image[y_start:y_end, x_start:x_end]

    # If zooming out, add padding to match original dimensions
    elif zoom_factor < 1.0:
        y_padding = (height - new_height) // 2
        x_padding = (width - new_width) // 2

        # Determine if the image has color channels
        if len(image.shape) == 3:
            channels = image.shape[2]
            zoomed_image = np.zeros((height, width, channels), dtype=image.dtype)
            zoomed_image[y_padding : y_padding + new_height, x_padding : x_padding + new_width] = cv2.resize(
                image, (new_width, new_height)
            )
        else:
            zoomed_image = np.zeros((height, width), dtype=image.dtype)
            zoomed_image[y_padding : y_padding + new_height, x_padding : x_padding + new_width] = cv2.resize(
                image, (new_width, new_height)
            )

    return zoomed_image


def move_mask_center(mask, image, apply_scaling=False, target_area_ratio=0.15):
    # Convert mask to 2D if it has channel dimension
    if len(mask.shape) == 3:
        mask_2d = mask[:, :, 0] if mask.shape[2] > 1 else mask.squeeze(-1)
    else:
        mask_2d = mask

    # Convert to float for center of mass calculation
    mask_float = mask_2d.astype(float)

    # Calculate center of mass of the mask
    total_mass = mask_float.sum()
    if total_mass == 0:
        # If mask is empty, return original inputs
        return mask, image

    height, width = mask_2d.shape

    # Step 1: Optionally scale if needed
    scaled_mask = mask.copy()
    scaled_image = image.copy()

    if apply_scaling:
        # Calculate mask area as ratio of total image area
        total_image_area = height * width
        mask_area = np.sum(mask_2d)
        mask_area_ratio = mask_area / total_image_area

        if mask_area_ratio > target_area_ratio:
            # Calculate zoom factor needed to reduce mask area to target
            # Since area scales with zoom_factor^2, we use square root
            zoom_factor = np.sqrt(target_area_ratio / mask_area_ratio)

            # Clamp zoom factor to avoid making things too small
            zoom_factor = max(zoom_factor, 0.5)

            # Apply scaling
            scaled_image = zoom_image(image, zoom_factor)
            if len(mask.shape) == 3:
                scaled_mask = np.zeros_like(mask)
                for c in range(mask.shape[2]):
                    scaled_mask[:, :, c] = zoom_image(mask[:, :, c].astype(float), zoom_factor) > 0
            else:
                scaled_mask = zoom_image(mask.astype(float), zoom_factor) > 0

    # Step 2: Recalculate center of mass on the scaled mask
    if len(scaled_mask.shape) == 3:
        scaled_mask_2d = scaled_mask[:, :, 0] if scaled_mask.shape[2] > 1 else scaled_mask.squeeze(-1)
    else:
        scaled_mask_2d = scaled_mask

    scaled_mask_float = scaled_mask_2d.astype(float)
    total_mass = scaled_mask_float.sum()

    if total_mass == 0:
        # If mask became empty after scaling, return scaled results
        return scaled_mask, scaled_image

    y_indices, x_indices = np.indices((height, width))

    center_y = (y_indices * scaled_mask_float).sum() / total_mass
    center_x = (x_indices * scaled_mask_float).sum() / total_mass

    # Calculate translation to move center of mass to image center
    image_center_y = height / 2.0
    image_center_x = width / 2.0

    shift_y = int(round(image_center_y - center_y))
    shift_x = int(round(image_center_x - center_x))

    # If no translation needed, return scaled results
    if shift_y == 0 and shift_x == 0:
        return scaled_mask, scaled_image

    # Step 3: Apply translation
    translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])

    # Translate the mask
    if len(scaled_mask.shape) == 3:
        translated_mask = np.zeros_like(scaled_mask)
        for c in range(scaled_mask.shape[2]):
            translated_mask[:, :, c] = cv2.warpAffine(
                scaled_mask[:, :, c].astype(np.uint8),
                translation_matrix,
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ).astype(bool)
    else:
        translated_mask = cv2.warpAffine(
            scaled_mask.astype(np.uint8),
            translation_matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)

    # Translate the image
    if len(scaled_image.shape) == 3:
        translated_image = cv2.warpAffine(
            scaled_image,
            translation_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    else:
        translated_image = cv2.warpAffine(
            scaled_image,
            translation_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    return translated_mask, translated_image


def preproc(
    item,
    image_transform,  # since we don't use clip anymore, we only need one transform for the vae
    resolution: int,  # needed for plucker conversion etc.
    max_views: int = 20,
    conditioning_mode: str = "random_n",
    conditioning_min_n: int = 1,  # inclusive
    conditioning_max_n: int = 4,  # inclusive
    symmetry_aug_p: float = 0.0,
    use_relative_coords: bool = True,
    mask_out_background_target: bool | str = "white",
    mask_out_background_cond: bool | str = "random_blocks",
    shuffle_inds=None,
    eval_mode: bool = False,  # generate spherical camera coordinates for target views, pad x with zeros in these views
    eval_cam_sampler: Callable = build_eval_cams,  # a functon that takes n_views and MVData as input, e.g. build_eval_cams()
    plucker_scene_scale: float = 30.0,
    camera_aug_enabled: bool = False,
    camera_aug_sigma_sample_p_mean: float = 0.0,
    camera_aug_sigma_sample_p_std: float = 1.0,
    camera_fov_aug_multiplier: float = 1.0,
    camera_aug_sigma_multiplier: float = 1.0,
    fov_aug_enabled: bool = False,
    fov_aug_on_target: bool = True,  # also apply fov augmentation to target views
    fov_aug_min: float = 0.7,
    fov_aug_max: float = 1.5,
    **kwargs,
):
    if eval_mode:
        assert conditioning_mode == "n"
    assert conditioning_max_n < max_views, "must have at least one output view"

    if item.auto_label is not None:
        to_pop = []
        for i, label in enumerate(item.auto_label):
            if label == 0:
                to_pop.append(i)
        to_pop.reverse()
        for i in to_pop:
            _ = item.pop(i)

    n_avail = len(item.frames)

    if item.lwh is None:
        item.lwh = np.array([1.0, 1.0, 1.8])

    if eval_mode:
        assert n_avail > 0, "must have at least 1 available view in data"
    else:
        assert n_avail > 1, "must have at least 2 available views in data"

    valid_frame_inds = list(range(n_avail))

    # Determine how many images to use as conditioning
    if eval_mode:
        conditioning_max_usable = min(n_avail, conditioning_max_n)
    else:
        conditioning_max_usable = min(n_avail - 1, conditioning_max_n)

    if conditioning_mode == "n":
        n_cond = conditioning_max_usable
    elif conditioning_mode == "random_n":
        conditioning_min_n_usable = min(conditioning_min_n, conditioning_max_usable)
        n_cond = np.random.randint(low=conditioning_min_n_usable, high=conditioning_max_usable + 1)
    else:
        raise NotImplementedError

    if eval_mode:
        # In eval mode, we can set an arbitrary n_target view as every image in the MVData can be used as conditioning
        n_valid = min(conditioning_max_n, n_avail)
        n_target = max_views - conditioning_max_n
    else:
        n_valid = min(max_views, n_avail)
        n_target = n_valid - n_cond

    if shuffle_inds is None:
        shuffle_inds = valid_frame_inds
        shuffle(shuffle_inds)
        shuffle_inds = shuffle_inds[:n_valid]

    if eval_mode and conditioning_max_n == 1:
        # compute the area of all masks in item.masks, select the median area, use the corresponding index as shuffle_inds
        mask_areas = []
        for idx in range(len(item.masks)):
            mask = item.masks[idx] > 0
            area = np.sum(mask)
            mask_areas.append((area, idx))

        # Sort by area and find the median
        mask_areas.sort(key=lambda x: x[0])
        median_idx = len(mask_areas) // 2
        median_area, median_frame_idx = mask_areas[median_idx]

        # Use the frame with median mask area as the single conditioning frame
        shuffle_inds = [median_frame_idx]

    # Determine if synthetic data and rdcam group is available
    synthetic_shuffle_inds = []
    if item.rdcam_frames is not None and len(item.rdcam_frames) > 0:
        synthetic_shuffle_inds = list(range(len(item.rdcam_frames)))
        shuffle(synthetic_shuffle_inds)

    # compute fov aug
    if fov_aug_enabled:
        random_fov_scale = np.random.uniform(fov_aug_min, fov_aug_max)
    else:
        random_fov_scale = 1.0

    # For eval, create cameras and prepend eval camera indices
    eval_cams = []
    if eval_mode:
        areas = []
        dists = []
        fovs = []
        for i, ind in enumerate(shuffle_inds):
            mask = item.masks[ind] > 0
            area = np.sum(mask) * 1.0 / (mask.shape[0] * mask.shape[1])

            dist = item.dists[ind]
            fov = item.fov[ind]

            areas.append(area)
            dists.append(dist)
            fovs.append(fov)

        # if np.mean(areas) > 0.15:
        if np.mean(areas) > 10000:
            fov_aug_enabled = False
            fov_aug_on_target = False

            projection_scale = np.sqrt(np.mean(areas) / 0.1)

            fov = np.mean(fovs)
            dist = np.mean(dists)

            width = dist * (2 * np.tan(np.deg2rad(fov / 2)))
            new_fov = np.rad2deg(np.atan(projection_scale * width / (dist * 2.0))) * 2.0

            random_fov_scale = new_fov / fov

        else:
            fov_aug_enabled = False
            fov_aug_on_target = False
            random_fov_scale = 1.0

        eval_cams = eval_cam_sampler(n_target, item, fov_scale=random_fov_scale)
        shuffle_inds = [
            0,
        ] * len(eval_cams) + shuffle_inds

    x_original = []
    x_white_background = []
    x = []
    x_masks = []
    cam_poses = []
    dists = []
    fovs = []
    cam_poses_proc = []
    intrinsics = []
    c2w_relatives = []
    c2w_canonicals = []
    is_valid_frame = []

    for i, ind in enumerate(shuffle_inds):
        if eval_mode and i < n_target:
            cam_pos, dist, fov = eval_cams[i]
            x_array = np.zeros_like(item.frames[0])
            mask = np.zeros_like(item.frames[0]).astype(bool)
        else:
            # Determine if we are drawing from the rdcam group or the default cam group
            # prioritize using rdcam in target instead of cond
            if not eval_mode and len(synthetic_shuffle_inds) > 0 and i < n_target and random.random() < 0.5:
                rd_id = synthetic_shuffle_inds.pop()
                x_array = item.rdcam_frames[rd_id]
                mask = (np.expand_dims(item.rdcam_masks[rd_id], axis=-1) > 0).repeat(3, axis=2)
                cam_pos = item.rdcam_cam_poses[rd_id]
                dist = item.rdcam_dists[rd_id]
                fov = item.rdcam_fov[rd_id]
            else:
                x_array = item.frames[ind]
                mask = (np.expand_dims(item.masks[ind], axis=-1) > 0).repeat(3, axis=2)
                cam_pos = item.cam_poses[ind]
                dist = item.dists[ind]
                fov = item.fov[ind]

        if fov_aug_enabled:
            if i >= n_target or fov_aug_on_target:
                # perform fov augmentation and image re-cropping
                new_fov = random_fov_scale * fov
                # scale factor is tan(new fov) / tan(old fov) = (opposite_new / dist) / (opposite_old / dist) = opposite_new / opposite_old
                scale = np.tan(np.deg2rad(new_fov / 2)) / np.tan(np.deg2rad(fov / 2))
                # zoom factor is the inverse of how real scale changes
                zoom = 1 / scale
                x_array = zoom_image(x_array, zoom)
                mask = zoom_image(mask.astype(float), zoom) > 0
                fov = new_fov

        if eval_mode and conditioning_max_n == 1:
            # move the weight center of the image mask to the center of the image by translation
            mask, x_array = move_mask_center(mask, x_array, apply_scaling=False)

        # make target GT image white background
        if i < n_target:
            mask_flag = mask_out_background_target
        else:
            mask_flag = mask_out_background_cond

        if mask_flag == "white":
            # white background
            background_color = np.ones((mask.shape[0], mask.shape[1], 3)).astype(float) * 255
            masked_img = x_array * mask + ~mask * background_color
        elif mask_flag == "black":
            # black background
            background_color = np.zeros((mask.shape[0], mask.shape[1], 3)).astype(float)
            masked_img = x_array * mask + ~mask * background_color
        elif mask_flag == "rand_gray":
            # random gray background
            background_color = np.ones((mask.shape[0], mask.shape[1], 3)).astype(float) * np.random.randint(255)
            masked_img = x_array * mask + ~mask * background_color
        elif mask_flag == "random_blocks":
            # random color blocks of 16x16 pixels
            background_color = (
                np.random.rand(mask.shape[0] // 16, mask.shape[1] // 16, 3).repeat(16, 0).repeat(16, 1) * 255
            )
            masked_img = x_array * mask + ~mask * background_color
        elif mask_flag == "noise":
            background_color = np.random.rand(mask.shape[0], mask.shape[1], 3) * 255
            masked_img = x_array * mask + ~mask * background_color
        elif not mask_flag or mask_flag == "False":
            masked_img = x_array
        else:
            raise NotImplementedError

        x_original.append(Image.fromarray(x_array.astype(np.uint8)))
        x.append(Image.fromarray(masked_img.astype(np.uint8)))
        x_masks.append(Image.fromarray(mask.astype(np.uint8) * 255))

        # store white background image
        background_color = np.ones((mask.shape[0], mask.shape[1], 3)).astype(float) * 255
        masked_img = x_array * mask + ~mask * background_color

        x_white_background.append(Image.fromarray(masked_img.astype(np.uint8)))

        cam_poses.append(cam_pos)
        dists.append(dist)
        fovs.append(fov)

    x, cam_poses = symmetry_augment(x, cam_poses, symmetry_aug_p)
    cam_aug_sigmas = []
    for i in range(len(cam_poses)):
        cam_pos = cam_poses[i]
        dist = dists[i]
        fov = fovs[i]

        cam_pos_cv2_scaled = dist * rig_to_open_cv(cam_pos)

        if camera_aug_enabled:
            if i >= n_target:
                gaussian_dist = NormalDist(mu=camera_aug_sigma_sample_p_mean, sigma=camera_aug_sigma_sample_p_std)
                cdf_val = np.random.uniform()

                log_sigma = gaussian_dist.inv_cdf(cdf_val)
                sigma = np.exp(log_sigma)
                cam_pos_cv2_scaled += sigma * np.random.randn(*cam_pos_cv2_scaled.shape)
                # fov is roughly  in the same order of magnitude as cam pos, so I default to 1.0 multiplier
                fov += camera_fov_aug_multiplier * sigma * np.random.randn(*fov.shape)
                cam_aug_sigmas.append(camera_aug_sigma_multiplier * sigma)
            else:
                cam_aug_sigmas.append(0.0)

        c2w = create_lookat_matrix(cam_pos_cv2_scaled)
        cam_poses_proc.append(cam_pos_cv2_scaled)
        c2w_canonicals.append(c2w)

        intrinsic = create_intrisics(fov, resolution, resolution)
        intrinsics.append(intrinsic)

        is_valid_frame.append(1)
    if len(cam_aug_sigmas) > 0:
        cam_aug_sigmas = torch.tensor(cam_aug_sigmas).float()
    else:
        cam_aug_sigmas = None

    if use_relative_coords:
        c2w_relatives = []
        base_campos = rig_to_open_cv(cam_poses[min(n_target, len(cam_poses) - 1)])
        transform_c2w = create_lookat_matrix(np.array([0, 0, 0]), base_campos)

        for i in range(len(cam_poses)):
            c2w = c2w_canonicals[i]
            c2w_rel = transform_camera_matrix(transform_c2w, c2w)
            c2w_relatives.append(c2w_rel)
    else:
        c2w_relatives = c2w_canonicals

    intrinsics = torch.tensor(np.stack(intrinsics, 0)).float()
    c2w_relatives = torch.tensor(np.stack(c2w_relatives, 0)).float()
    plucker_images = ray_condition(
        intrinsics.unsqueeze(0), c2w_relatives.unsqueeze(0), resolution, resolution, "cpu", plucker_scene_scale
    ).squeeze(0)
    plucker_images = plucker_images.clamp(-1.0, 1.0)

    if eval_mode or len(x_white_background) == n_target:
        relative_brightness = [0.0] * len(x_white_background)
    else:
        avg_brightness = []
        for img, mask in zip(x_white_background, x_masks):
            img_np = np.array(img.convert("L"))  # convert image to grayscale
            mask_np = np.array(mask.convert("L")) > 0  # assume non-zero in mask is foreground

            foreground_pixels = img_np[mask_np]
            mean_brightness = foreground_pixels.mean() if foreground_pixels.size > 0 else 0
            avg_brightness.append(mean_brightness)

        cond_brightbess = np.mean(avg_brightness[n_target:])

        relative_brightness = [b - cond_brightbess for b in avg_brightness]

        for ii in range(n_target, len(relative_brightness)):
            relative_brightness[ii] = 0.0

    ##
    return_dict = AttrDict(
        lwh=item.lwh,
        clip_id=item.clip_id,
        obj_id=item.obj_id,
        n_target=n_target,
        x=torch.stack([image_transform(ip) for ip in x], dim=0),
        x_original=torch.stack([image_transform(ip) for ip in x_original], dim=0),
        x_white_background=torch.stack([image_transform(ip) for ip in x_white_background], dim=0),
        x_msk=torch.stack([image_transform(ip) for ip in x_masks], dim=0),
        # global_orients=torch.tensor(np.stack(global_orients, 0)).float(),  # New: 3-dimensional global orientations
        cam_poses=torch.tensor(np.stack(cam_poses, 0)).float(),
        cam_poses_proc=torch.tensor(np.stack(cam_poses_proc, 0)).float(),
        dists=torch.tensor(np.stack(dists, 0)).float(),
        fovs=torch.tensor(np.stack(fovs, 0)).float(),
        intrinsics=intrinsics,
        c2w_relatives=c2w_relatives,
        c2w_canonicals=torch.tensor(np.stack(c2w_canonicals, 0)).float(),
        is_valid_frame=torch.tensor(is_valid_frame, dtype=torch.int),
        plucker_image=plucker_images,
        cam_aug_sigmas=cam_aug_sigmas,
        relative_brightness=torch.tensor(relative_brightness).float(),
    )

    return return_dict
