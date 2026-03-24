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

import logging
from collections import defaultdict
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tqdm
from ncore.data import BBox3, FrameTimepoint, OpenCVPinholeCameraModelParameters, ShutterType
from ncore.data.v4 import SequenceComponentGroupsReader, SequenceLoaderV4
from ncore.impl.common.transformations import (
    PoseInterpolator,
    bbox_pose,
    pose_bbox,
    se3_inverse,
    transform_bbox,
    transform_point_cloud,
)
from ncore.impl.data.compat import CameraSensorProtocol, LidarSensorProtocol
from ncore.impl.data.types import CuboidTrackObservation
from ncore.sensors import CameraModel, OpenCVPinholeCameraModel

from .mvdata import MVData
from .ray_aabb import ray_aabb_intersect

# Map raw NCore label_class values to canonical downstream labels.
# Any label not in this map is lowercased and used as-is.
_LABEL_NORMALIZATION: dict[str, str] = {
    "VRU_pedestrians": "person",
    "VRU_pedestrian": "person",
    "consumer_vehicles": "automobile",
    "consumer_vehicle": "automobile",
}

# Labels to exclude from parsing — these are typically vehicle sub-parts, not standalone objects.
_EXCLUDED_LABELS: set[str] = {"protruding_object"}


def normalize_label(raw_label: str) -> str:
    """Normalize a raw NCore label_class to a canonical downstream label."""
    return _LABEL_NORMALIZATION.get(raw_label, raw_label.lower())


logger = logging.getLogger(__name__)


def clean_track_id(track_id: str) -> str:
    """Strip annotation source suffix (e.g. '@scene:obstacles:autolabels:v2') from a track ID."""
    return track_id.split("@", 1)[0]


def get_corners(bbox: BBox3) -> np.ndarray:
    """Compute 8 corner points of a 3D bounding box in world coordinates."""
    le2 = bbox.dim[0] / 2
    wi2 = bbox.dim[1] / 2
    he2 = bbox.dim[2] / 2

    corners = np.array(
        [
            [-le2, -wi2, -he2],
            [le2, -wi2, -he2],
            [le2, wi2, -he2],
            [-le2, wi2, -he2],
            [-le2, -wi2, he2],
            [le2, -wi2, he2],
            [le2, wi2, he2],
            [-le2, wi2, he2],
        ]
    )
    bbox_to_world = bbox_pose(bbox.to_array())
    rotations = bbox_to_world[..., 0:3, 0:3]
    translations = bbox_to_world[..., 0:3, 3]
    return np.matmul(rotations, corners.transpose()).transpose() + translations


def look_at_eye_zero2(at: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Create a view matrix looking at a target from origin."""
    w = at
    w /= w.norm()
    u = torch.cross(up, -w, dim=-1)
    u /= u.norm()
    v = torch.cross(w, u, dim=-1)

    return torch.FloatTensor(
        [
            [u[0], u[1], u[2], 0],
            [v[0], v[1], v[2], 0],
            [w[0], w[1], w[2], 0],
            [0, 0, 0, 1],
        ]
    ).to(at.device)


def get_up_dir(points_cam_view: torch.Tensor) -> torch.Tensor:
    """Compute the up direction from bounding box corners in camera view."""
    up_cam_view = points_cam_view[5:] - points_cam_view[1:5]
    up_cam_view = up_cam_view.mean(dim=0)
    return up_cam_view / up_cam_view.norm()


@dataclass
class TrackData:
    """Processed track data from CuboidTrackObservations."""

    poses: list[np.ndarray]
    dimension: np.ndarray
    label_class: str
    timestamps_us: list[int]
    interpolator: PoseInterpolator | None = None


def build_tracks_from_observations(
    observations: Generator[CuboidTrackObservation, None, None],
    pose_graph,
) -> dict[str, TrackData]:
    """Build track data dictionary from CuboidTrackObservation generator."""
    grouped: dict[str, list[CuboidTrackObservation]] = defaultdict(list)
    for obs in observations:
        grouped[obs.track_id].append(obs)

    all_tracks: dict[str, TrackData] = {}
    for track_id, obs_list in grouped.items():
        obs_list.sort(key=lambda x: x.reference_frame_timestamp_us)

        poses = []
        timestamps_us: list[int] = []
        label_class = obs_list[0].class_id
        if normalize_label(label_class) in _EXCLUDED_LABELS:
            continue
        dimension = np.array(obs_list[0].bbox3.dim, dtype=np.float32)

        for obs in obs_list:
            if timestamps_us and obs.reference_frame_timestamp_us == timestamps_us[-1]:
                continue
            T_ref_world = pose_graph.evaluate_poses(
                obs.reference_frame_id,
                "world",
                np.array(obs.reference_frame_timestamp_us, dtype=np.uint64),
            )
            bbox_world = BBox3.from_array(transform_bbox(obs.bbox3.to_array(), T_ref_world))
            pose_world = bbox_pose(bbox_world.to_array())
            poses.append(pose_world)
            timestamps_us.append(obs.reference_frame_timestamp_us)

        interpolator = None
        if len(poses) > 1:
            interpolator = PoseInterpolator(np.stack(poses), timestamps_us)

        all_tracks[track_id] = TrackData(
            poses=poses,
            dimension=dimension,
            label_class=label_class,
            timestamps_us=timestamps_us,
            interpolator=interpolator,
        )

    return all_tracks


def compute_track_existences(
    all_tracks: dict[str, TrackData],
    camera_sensors: dict[str, CameraSensorProtocol],
) -> dict[str, dict[int, list[str]]]:
    """Compute which tracks are visible in each camera frame."""
    track_existences: dict[str, dict[int, list[str]]] = {}
    for camera_id, camera_sensor in camera_sensors.items():
        track_existences[camera_id] = {}
        for frame_idx in range(camera_sensor.frames_count):
            track_existences[camera_id][frame_idx] = []
            frame_end_ts = camera_sensor.get_frame_timestamp_us(frame_idx, FrameTimepoint.END)
            for track_id, track in all_tracks.items():
                if track.timestamps_us[0] <= frame_end_ts <= track.timestamps_us[-1]:
                    track_existences[camera_id][frame_idx].append(track_id)

    return track_existences


class NCoreObjectParser:
    """Parse tracked-object views from ncore V4 data."""

    def __init__(
        self,
        target_resolution: int,
        num_lidar_ref_frames: int,
        cam_pose_flip: list[int],
        occ_rate_threshold: float,
        crop_min_area_ratio: float,
        max_threads: int,
    ) -> None:
        self.target_resolution = target_resolution
        self.num_lidar_ref_frames = num_lidar_ref_frames
        self.cam_pose_flip = cam_pose_flip
        self.occ_rate_threshold = occ_rate_threshold
        self.crop_min_area_ratio = crop_min_area_ratio
        torch.set_num_threads(max_threads)
        self.ncore_datasets: dict[str, dict[str, Any]] = {}
        logger.info("NCoreObjectParser initialized")

    def _load_ncore_data(self, src_data_paths: list[str]) -> str:
        logger.info("Loading ncore V4 data from %s", src_data_paths)
        reader = SequenceComponentGroupsReader([Path(p) for p in src_data_paths])
        loader = SequenceLoaderV4(reader)
        seq_id = loader.sequence_id

        if seq_id not in self.ncore_datasets:
            lidar_ids = loader.lidar_ids
            assert len(lidar_ids) == 1, "Only one lidar is supported"
            lidar = loader.get_lidar_sensor(lidar_ids[0])

            cameras = {cam_id: loader.get_camera_sensor(cam_id) for cam_id in loader.camera_ids}
            camera_models = {
                cam_id: CameraModel.from_parameters(cameras[cam_id].model_parameters, device="cpu")
                for cam_id in cameras.keys()
            }

            all_tracks = build_tracks_from_observations(loader.get_cuboid_track_observations(), loader.pose_graph)
            track_existences = compute_track_existences(all_tracks, cameras)

            self.ncore_datasets[seq_id] = {
                "loader": loader,
                "lidar": lidar,
                "cameras": cameras,
                "camera_models": camera_models,
                "all_tracks": all_tracks,
                "track_existences": track_existences,
            }

        return seq_id

    def _sample_lidar_reference_timestamps(self, track: TrackData) -> list[int]:
        """Uniformly sample timestamps from track for lidar frame lookup."""
        max_len = len(track.timestamps_us)
        if max_len > self.num_lidar_ref_frames:
            interval = max_len // self.num_lidar_ref_frames
            start_idx = np.random.randint(0, interval)
            return track.timestamps_us[start_idx::interval]
        return track.timestamps_us

    def _get_lidar_and_bbox_data_single_frame(
        self,
        track: TrackData,
        lidar: LidarSensorProtocol,
    ) -> Generator[tuple[str, int, BBox3], None, None]:
        """Yields (obj_cls, lidar_timestamp, bbox_world) for each valid frame in track.

        Note on V4 vs V3 timestamp handling:
            In ncore V3, track timestamps came directly from lidar frame iteration, so
            track.timestamps == lidar frame timestamps (exact match expected).

            In ncore V4, track timestamps come from CuboidTrackObservation.reference_frame_timestamp_us,
            which are observation timestamps that differ from lidar frame timestamps by ~25-50ms.

            We handle this by: (1) finding the closest lidar frame to each track timestamp,
            (2) using the lidar frame's timestamp for pose interpolation.
        """
        frame_timestamps = self._sample_lidar_reference_timestamps(track)
        for frame_timestamp in frame_timestamps:
            lidar_frame_idx = lidar.get_closest_frame_index(frame_timestamp, relative_frame_time=1.0)
            if lidar_frame_idx == 0:
                continue
            lidar_frame_timestamp = lidar.get_frame_timestamp_us(lidar_frame_idx, FrameTimepoint.END)

            if not (track.timestamps_us[0] <= lidar_frame_timestamp <= track.timestamps_us[-1]):
                continue

            obj_cls = normalize_label(track.label_class)
            bbox_pose_world = (
                track.interpolator.interpolate_to_timestamps([lidar_frame_timestamp])[0]
                if track.interpolator
                else track.poses[0]
            )
            bbox_world = BBox3.from_array(pose_bbox(bbox_pose_world, track.dimension))
            yield obj_cls, lidar_frame_timestamp, bbox_world

    def _compute_transforms(
        self,
        bbox_points: np.ndarray,
        camera_sensor: CameraSensorProtocol,
        camera_model: CameraModel,
        lidar_timestamp: int,
        crop_min_area_ratio: float = 0.002,
    ) -> tuple[bool, float, torch.Tensor | None, Any]:
        """Compute camera transforms and validate bbox projection."""
        camera_frame_idx = camera_sensor.get_closest_frame_index(lidar_timestamp, relative_frame_time=1.0)
        crop_min_area = camera_model.resolution[0] * camera_model.resolution[1] * crop_min_area_ratio
        T_world_sensor_start = camera_sensor.get_frames_T_source_sensor("world", camera_frame_idx, FrameTimepoint.START)
        T_world_sensor_end = camera_sensor.get_frames_T_source_sensor("world", camera_frame_idx, FrameTimepoint.END)
        timestamp_start = camera_sensor.get_frame_timestamp_us(camera_frame_idx, FrameTimepoint.START)
        timestamp_end = camera_sensor.get_frame_timestamp_us(camera_frame_idx, FrameTimepoint.END)

        image_points = camera_model.world_points_to_image_points_shutter_pose(
            bbox_points,
            T_world_sensor_start,
            T_world_sensor_end,
            start_timestamp_us=timestamp_start,
            end_timestamp_us=timestamp_end,
            return_T_world_sensors=True,
            return_valid_indices=True,
            return_timestamps=True,
        )

        if image_points.image_points.shape[0] < 9 or (
            image_points.valid_indices is not None and len(image_points.valid_indices) < 9
        ):
            return False, 0.0, None, image_points

        uvmx, uvmn = image_points.image_points.max(dim=0).values, image_points.image_points.min(dim=0).values
        box_dim = uvmx - uvmn
        if torch.prod(box_dim).abs().cpu().item() < crop_min_area:
            return False, 0.0, None, image_points

        bbox_points_cam_space = transform_point_cloud(
            torch.tensor(np.expand_dims(bbox_points, 1)).to(image_points.T_world_sensors),
            image_points.T_world_sensors,
        )[:, 0, :]
        centroid_cam_space = bbox_points_cam_space[0:1]
        corners_cam_space = bbox_points_cam_space[1:]
        dist = centroid_cam_space.norm()
        bbox_fov = 2.5 * torch.rad2deg(
            torch.arctan2(0.5 * (corners_cam_space.max(dim=0)[0] - corners_cam_space.min(dim=0)[0]).max(), dist)
        )

        return True, bbox_fov.item(), bbox_points_cam_space, image_points

    def _get_cropped_frame(
        self,
        src_camera_model: CameraModel,
        fov: torch.Tensor,
        points_cam_view: torch.Tensor,
        src_image: torch.Tensor,
        target_res: int = 512,
    ) -> tuple[torch.Tensor, np.ndarray, torch.Tensor]:
        target_dim_range = torch.arange(target_res, dtype=torch.int32, device=src_camera_model.device)
        target_pixels_x, target_pixels_y = torch.meshgrid(target_dim_range, target_dim_range, indexing="xy")
        target_pixels = torch.stack([target_pixels_x.flatten(), target_pixels_y.flatten()], dim=1)
        target_image_points = target_pixels + 0.5
        target_principal_point = target_res / 2
        target_focal_length = target_res / (2.0 * torch.tan(torch.deg2rad(fov) * 0.5))
        target_rays = (target_image_points - target_principal_point) / target_focal_length
        target_rays = torch.cat([target_rays, torch.ones_like(target_rays[:, :1])], dim=1)
        target_cam_parameters = OpenCVPinholeCameraModelParameters(
            np.array([target_res, target_res], dtype=np.uint64),
            principal_point=np.array([target_principal_point, target_principal_point], dtype=np.float32),
            focal_length=np.array(
                [target_focal_length.cpu().numpy().item(), target_focal_length.cpu().numpy().item()],
                dtype=np.float32,
            ),
            radial_coeffs=np.zeros(6, np.float32),
            tangential_coeffs=np.zeros(2, np.float32),
            thin_prism_coeffs=np.zeros(4, np.float32),
            shutter_type=ShutterType.GLOBAL,
        )
        target_cam_model = OpenCVPinholeCameraModel(target_cam_parameters, device=src_camera_model.device)

        new_up = get_up_dir(points_cam_view)
        R = look_at_eye_zero2(points_cam_view[0].to(points_cam_view.device), new_up)
        R_inv = torch.concat(
            [
                torch.concat([R[:3, :3].T, torch.zeros((3, 1)).to(R)], dim=-1),
                torch.tensor([0.0, 0.0, 0.0, 1.0]).to(R).unsqueeze(0),
            ],
            dim=0,
        )
        target_rays_cam = target_rays @ R_inv[:3, :3].T + R_inv[:3, 3]

        points_target_cam_view = transform_point_cloud(points_cam_view.unsqueeze(0), R.unsqueeze(0))[0]
        bbox_corner_pixels = target_cam_model.camera_rays_to_pixels(points_target_cam_view)
        valid_bbox_corner_pixels = bbox_corner_pixels.pixels[bbox_corner_pixels.valid_flag]

        source_pixels = src_camera_model.camera_rays_to_pixels(target_rays_cam)
        valid = source_pixels.valid_flag
        valid_source_pixels = source_pixels.pixels[valid]
        valid_target_pixels = target_pixels[valid]
        valid_rgb = src_image[valid_source_pixels[:, 1], valid_source_pixels[:, 0]]
        target_image = torch.zeros((target_res, target_res, 3), dtype=torch.uint8, device=valid_rgb.device)
        target_image[valid_target_pixels[:, 1], valid_target_pixels[:, 0]] = valid_rgb
        return target_image, new_up.cpu().numpy(), valid_bbox_corner_pixels

    def extract(
        self,
        src_data_paths: list[str],
        target_track_ids: list[str] | None,
        target_camera_ids: list[str] | None,
    ) -> tuple[dict[str, MVData], dict[str, list[float]]]:
        logger.info("Extracting views from %s", src_data_paths)
        crop_views_dict: dict[str, MVData] = {}
        occ_rate_dict: dict[str, list[float]] = {}

        seq_id = self._load_ncore_data(src_data_paths)
        ncore_dataset = self.ncore_datasets[seq_id]
        all_tracks = ncore_dataset["all_tracks"]
        track_existences = ncore_dataset["track_existences"]
        lidar = ncore_dataset["lidar"]
        camera_sensors = ncore_dataset["cameras"]
        camera_models = ncore_dataset["camera_models"]
        if target_camera_ids is None or len(target_camera_ids) == 0:
            target_camera_ids = list(camera_sensors.keys())

        logger.info("Target cameras: %s", target_camera_ids)

        normalized_targets = set(clean_track_id(t) for t in target_track_ids) if target_track_ids else None

        with tqdm.tqdm(total=len(all_tracks), position=0) as pbar:
            for track_id, track in all_tracks.items():
                if normalized_targets is not None and clean_track_id(track_id) not in normalized_targets:
                    continue

                track_data: dict[str, list[Any]] = {
                    "timestamps": [],
                    "src_images": [],
                    "box_cam_view": [],
                    "fov": [],
                    "image_points": [],
                    "camera_sensors": [],
                    "obj_cls": [],
                    "bbox_world_coord": [],
                    "sensor_id": [],
                    "occ_rate": [],
                }

                n_lidar_frames = 0
                n_projection_skip = 0
                n_occlusion_skip = 0

                for obj_cls, lidar_frame_timestamp, bbox_world_coord in self._get_lidar_and_bbox_data_single_frame(
                    track, lidar
                ):
                    n_lidar_frames += 1

                    transformed_bbox_corners = get_corners(bbox_world_coord)
                    T_box_world = bbox_pose(bbox_world_coord.to_array())
                    bbox_points = np.concatenate(
                        [np.array(bbox_world_coord.centroid).reshape((1, 3)), transformed_bbox_corners], axis=0
                    )

                    for sensor_id in target_camera_ids:
                        if sensor_id not in camera_sensors:
                            available_sensors = sorted(camera_sensors.keys())
                            raise ValueError(
                                f"{sensor_id} not found in camera_sensors.\n"
                                f"Available cameras: {', '.join(available_sensors)}\n"
                                f"Use --camera-ids to select from the above, e.g.:\n"
                                f'  --camera-ids "{",".join(available_sensors)}"'
                            )

                        camera_sensor = camera_sensors[sensor_id]
                        camera_model = camera_models[sensor_id]

                        is_valid, fov, bbox_cam_view, image_points = self._compute_transforms(
                            bbox_points, camera_sensor, camera_model, lidar_frame_timestamp, self.crop_min_area_ratio
                        )

                        camera_frame_idx = camera_sensor.get_closest_frame_index(
                            lidar_frame_timestamp, relative_frame_time=1.0
                        )
                        if not is_valid:
                            n_projection_skip += 1
                            continue

                        track_candidates = track_existences[sensor_id][camera_frame_idx]
                        frame_end_timestamp_us = camera_sensor.get_frame_timestamp_us(
                            camera_frame_idx, FrameTimepoint.END
                        )
                        track_aabb_mins: list[np.ndarray] = []
                        track_aabb_maxs: list[np.ndarray] = []
                        camera_T_sensor_world = camera_sensor.get_frames_T_sensor_target(
                            "world", camera_frame_idx, FrameTimepoint.END
                        )
                        ray_s_world = camera_T_sensor_world[:3, 3]
                        track_dist_to_cam = np.linalg.norm(np.array(bbox_world_coord.centroid) - ray_s_world)

                        for track_id_t in track_candidates:
                            if track_id_t == track_id:
                                continue
                            track_t = all_tracks[track_id_t]
                            if track_t.interpolator is not None:
                                track_pose = track_t.interpolator.interpolate_to_timestamps([frame_end_timestamp_us])[0]
                            else:
                                track_pose = track_t.poses[0]
                            track_dims = track_t.dimension
                            track_bbox = BBox3.from_array(pose_bbox(track_pose, track_dims))
                            dist_to_cam = np.linalg.norm(np.array(track_bbox.centroid) - ray_s_world)
                            if dist_to_cam > track_dist_to_cam:
                                continue
                            track_bbox_corners = get_corners(track_bbox)
                            track_aabb_mins.append(np.min(track_bbox_corners, axis=0))
                            track_aabb_maxs.append(np.max(track_bbox_corners, axis=0))

                        occ_rate = 0.0
                        if len(track_aabb_mins) > 0:
                            device = bbox_cam_view.device if bbox_cam_view is not None else torch.device("cpu")
                            track_aabb_mins_tensor = torch.tensor(
                                np.array(track_aabb_mins), dtype=torch.float32, device=device
                            )
                            track_aabb_maxs_tensor = torch.tensor(
                                np.array(track_aabb_maxs), dtype=torch.float32, device=device
                            )
                            num_points = 128
                            rays_s_world = torch.tensor(
                                np.repeat(ray_s_world.reshape(1, 3), num_points, axis=0),
                                dtype=torch.float32,
                                device=device,
                            ).contiguous()
                            rays_e_bbox = (np.random.rand(num_points, 3).astype(np.float32) - 0.5) * np.array(
                                bbox_world_coord.dim
                            )
                            rays_e_world = (T_box_world[:3, :3] @ rays_e_bbox.T + T_box_world[:3, 3:4]).T
                            rays_e_world = torch.tensor(rays_e_world, dtype=torch.float32, device=device)
                            rays_d = rays_e_world - rays_s_world
                            normlized_rays_d = (rays_d / torch.linalg.norm(rays_d, dim=1).unsqueeze(-1)).contiguous()
                            n_hits, _ = ray_aabb_intersect(
                                rays_s_world, normlized_rays_d, track_aabb_mins_tensor, track_aabb_maxs_tensor
                            )
                            if torch.sum(n_hits) > 0:
                                n_hits = torch.sum(n_hits, dim=1)
                                occ_rate = torch.sum(n_hits > 0, dtype=torch.float32).item() / n_hits.shape[0]
                                if occ_rate > self.occ_rate_threshold:
                                    n_occlusion_skip += 1
                                    continue

                        image_array = camera_sensor.get_frame_image_array(camera_frame_idx)
                        image = torch.tensor(
                            image_array, device=bbox_cam_view.device if bbox_cam_view is not None else "cpu"
                        )

                        track_data["timestamps"].append(lidar_frame_timestamp)
                        track_data["src_images"].append(image)
                        track_data["box_cam_view"].append(bbox_cam_view)
                        track_data["fov"].append(fov)
                        track_data["image_points"].append(image_points)
                        track_data["camera_sensors"].append(camera_sensor)
                        track_data["bbox_world_coord"].append(bbox_world_coord)
                        track_data["obj_cls"].append(obj_cls)
                        track_data["sensor_id"].append(sensor_id)
                        track_data["occ_rate"].append(occ_rate)

                n_kept = len(track_data["src_images"])
                logger.info(
                    "Track %s: %d kept from %d lidar frames x %d cameras — filtered: projection=%d, occlusion=%d",
                    track_id,
                    n_kept,
                    n_lidar_frames,
                    len(target_camera_ids),
                    n_projection_skip,
                    n_occlusion_skip,
                )

                for i, _ in enumerate(track_data["timestamps"]):
                    camera_model = camera_models[track_data["sensor_id"][i]]
                    cropped_image, bbox_pose_persp_cam_view, bbox_corner_pixels = self._get_cropped_frame(
                        camera_model,
                        torch.tensor(track_data["fov"][i]).to(track_data["box_cam_view"][i].device),
                        track_data["box_cam_view"][i],
                        track_data["src_images"][i],
                        self.target_resolution,
                    )
                    ego_world_to_sensor = track_data["image_points"][i].T_world_sensors[0]
                    ego_pose = torch.tensor(se3_inverse(ego_world_to_sensor.cpu().numpy())).to(ego_world_to_sensor)[
                        :3, -1
                    ]
                    bbox_world_coord = track_data["bbox_world_coord"][i]
                    T_bbox_world_to_bbox = torch.tensor(se3_inverse(bbox_pose(bbox_world_coord.to_array()))).to(
                        ego_pose
                    )
                    ego_cam_pose_rig_coord = (
                        T_bbox_world_to_bbox[:3, :3] @ ego_pose.unsqueeze(-1) + T_bbox_world_to_bbox[:3, 3:4]
                    )
                    ego_cam_pose_our_coord = ego_cam_pose_rig_coord.squeeze() * torch.FloatTensor(
                        self.cam_pose_flip
                    ).to(ego_cam_pose_rig_coord)
                    dist = torch.norm(ego_cam_pose_our_coord)
                    cam_dir_our_coord = ego_cam_pose_our_coord / dist

                    uid = clean_track_id(track_id)
                    if uid not in crop_views_dict:
                        imset = MVData(
                            seq_id,
                            clean_track_id(track_id),
                            cropped_image.unsqueeze(0).cpu().numpy(),
                            cam_dir_our_coord.unsqueeze(0).cpu().numpy(),
                            dist.unsqueeze(0).cpu().numpy(),
                            np.array([track_data["fov"][i]]),
                            track_data["obj_cls"][i],
                            bbox_pose_persp_cam_view,
                            [bbox_corner_pixels.cpu().numpy()],
                            sensor_id=[track_data["sensor_id"][i]],
                            lwh=np.array(bbox_world_coord.dim),
                        )
                        crop_views_dict[uid] = imset
                        occ_rate_dict[uid] = [track_data["occ_rate"][i]]
                    else:
                        crop_views_dict[uid].append(
                            cropped_image.unsqueeze(0).cpu().numpy(),
                            cam_dir_our_coord.unsqueeze(0).cpu().numpy(),
                            dist.unsqueeze(0).cpu().numpy(),
                            np.array([track_data["fov"][i]]),
                            sensor_id=track_data["sensor_id"][i],
                            bbox_pix=bbox_corner_pixels.cpu().numpy(),
                        )
                        occ_rate_dict[uid].append(track_data["occ_rate"][i])

                pbar.update(1)

        return crop_views_dict, occ_rate_dict
