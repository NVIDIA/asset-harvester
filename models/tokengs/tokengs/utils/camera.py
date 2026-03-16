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
from scipy import interpolate
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def orbit_camera(
    elevation: float,
    azimuth: float,
    radius: float = 1.0,
    is_degree: bool = True,
    target: np.ndarray | None = None,
    opengl: bool = True,
) -> np.ndarray:
    """
    Construct a camera-to-world (c2w) pose matrix orbiting a target with elevation and azimuth.

    Convention (OpenGL / kiui-compatible):
    - World: +x=right, +y=up, +z=forward (right-handed).
    - Elevation in (-90, 90): from +y toward -y.
    - Azimuth in (-180, 180): +z(0) -> +x(90) -> -z(180) -> -x(270) -> +z(360).
    - Pose columns are [Right, Up, Forward, Position]; forward = target -> camera.

    Args:
        elevation: Elevation angle (vertical), from +y to -y.
        azimuth: Azimuth angle (horizontal), from +z toward +x.
        radius: Distance from target to camera.
        is_degree: If True, angles are in degrees; otherwise radians.
        target: Look-at target position, shape (3,). Defaults to origin.
        opengl: If True, use OpenGL camera convention (forward = target -> camera). Unused for now, kept for API compatibility.

    Returns:
        Camera pose (c2w) matrix, float [4, 4].
    """
    if is_degree:
        el = np.deg2rad(float(elevation))
        az = np.deg2rad(float(azimuth))
    else:
        el = float(elevation)
        az = float(azimuth)

    if target is None:
        target = np.zeros(3, dtype=np.float64)

    # Camera position on sphere: azimuth from +z, elevation from xz-plane
    # x = r*cos(el)*sin(az), y = r*sin(el), z = r*cos(el)*cos(az)
    cos_el = np.cos(el)
    sin_el = np.sin(el)
    cos_az = np.cos(az)
    sin_az = np.sin(az)
    position = radius * np.array([cos_el * sin_az, sin_el, cos_el * cos_az], dtype=np.float64)
    position = position + target

    # Forward = (camera - target) / radius (OpenGL: camera looks from target toward camera)
    forward = (position - target) / (radius + 1e-8)
    forward = forward / (np.linalg.norm(forward) + 1e-8)

    # Right = world_up x forward; Up = forward x right (right-handed)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(world_up, forward)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-8:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        right = right / right_norm
    up = np.cross(forward, right)
    up = up / (np.linalg.norm(up) + 1e-8)

    # c2w pose: columns [Right, Up, Forward, Position]
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = forward
    pose[:3, 3] = position
    return pose


def interp_cam_pose(c2ws: np.ndarray, num_frames: int) -> np.ndarray:
    """
    Interpolate camera poses along a trajectory.

    Args:
        c2ws: Camera-to-world transformation matrices of shape [num_cameras, 4, 4]
        num_frames: Number of frames to interpolate

    Returns:
        Interpolated camera-to-world matrices of shape [num_frames, 4, 4]
    """
    # Extract camera positions
    camera_positions = c2ws[:, :3, 3]

    # Create interpolation function for positions
    time_points = np.linspace(0, 1, c2ws.shape[0])
    position_interpolator = interpolate.interp1d(time_points, camera_positions, axis=0)

    # Interpolate positions
    new_time_points = np.linspace(0, 1, num_frames)
    interpolated_positions = position_interpolator(new_time_points)[..., None]

    # Extract rotation matrices
    rotation_matrices = c2ws[:, :3, :3]

    # Create rotation interpolator using spherical linear interpolation
    key_rotations = R.from_matrix(rotation_matrices)
    slerp_interpolator = Slerp(time_points, key_rotations)

    # Interpolate rotations
    interpolated_rotations = slerp_interpolator(new_time_points).as_matrix()

    # Combine rotations and translations
    interpolated_c2ws = np.concatenate([interpolated_rotations, interpolated_positions], axis=-1)

    # Add homogeneous coordinate row [0, 0, 0, 1]
    homogeneous_row = np.array([0.0, 0.0, 0.0, 1.0])[None, None, :].repeat(num_frames, axis=0)
    interpolated_c2ws = np.concatenate([interpolated_c2ws, homogeneous_row], axis=1)

    return interpolated_c2ws


def interp_cam_pose_spiral(
    c2ws: np.ndarray,
    num_frames: int,
    num_spirals: int = 2,
    x_scale: float = 1.0,
    y_scale: float = 1.0,
    z_scale: float = 0.5,
    distance_scale: float = 1.0,
) -> np.ndarray:
    """
    Generate spiral camera trajectory by interpolating poses.

    Args:
        c2ws: Camera-to-world transformation matrices of shape [num_cameras, 4, 4]
        num_frames: Number of frames to generate
        num_spirals: Number of spiral rotations
        x_scale: Scale factor for x-axis spiral motion
        y_scale: Scale factor for y-axis spiral motion
        z_scale: Scale factor for z-axis spiral motion
        distance_scale: Scale factor for spiral radius

    Returns:
        Interpolated camera-to-world matrices of shape [num_frames, 4, 4] as numpy.ndarray
    """
    # Extract camera positions and compute centroid
    camera_positions = c2ws[:, :3, 3]
    centroid = camera_positions.mean(axis=0)

    # Compute spiral radius based on average distance from centroid
    distances_from_centroid = np.linalg.norm(camera_positions - centroid, axis=1)
    spiral_radius = distances_from_centroid.mean() * distance_scale

    # Generate base positions at centroid
    base_positions = centroid[None, :].repeat(num_frames, axis=0)

    # Create spiral motion parameters
    spiral_angles = np.linspace(0, 2 * np.pi * num_spirals, num_frames)

    # Calculate spiral offset for each axis
    spiral_offset = np.stack(
        [
            spiral_radius * np.cos(spiral_angles * x_scale),
            spiral_radius * np.sin(spiral_angles * y_scale),
            spiral_radius * np.sin(spiral_angles * z_scale),
        ],
        axis=-1,
    )

    # Combine base positions with spiral offset
    final_positions = (base_positions + spiral_offset)[..., None]

    # Interpolate rotations using spherical linear interpolation for smooth transitions
    rotation_matrices = c2ws[:, :3, :3]
    key_rotations = R.from_matrix(rotation_matrices)

    # Use spherical linear interpolation for rotations
    time_points = np.linspace(0, 1, len(c2ws))
    slerp_interpolator = Slerp(time_points, key_rotations)

    # Interpolate rotations
    new_time_points = np.linspace(0, 1, num_frames)
    interpolated_rotations = slerp_interpolator(new_time_points).as_matrix()

    # Combine rotations and translations
    interpolated_c2ws = np.concatenate([interpolated_rotations, final_positions], axis=-1)

    # Add homogeneous coordinate row
    homogeneous_row = np.array([0.0, 0.0, 0.0, 1.0])[None, None, :].repeat(num_frames, axis=0)
    interpolated_c2ws = np.concatenate([interpolated_c2ws, homogeneous_row], axis=1)

    return interpolated_c2ws
