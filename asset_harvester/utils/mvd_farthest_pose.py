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


"""Farthest-point sampling on camera poses for MVD conditioning (matches nre asset_harvester)."""

from __future__ import annotations

import numpy as np


def farthest_point_sampling(
    points: np.ndarray,
    num_samples: int = 4,
    dist_threshold: float = 0.1,
    seed: int | None = None,
) -> np.ndarray:
    """Greedy FPS on rows of ``points`` (e.g. ``MVData.cam_poses``)."""
    points = np.asarray(points, dtype=np.float64)
    num_points = points.shape[0]
    if num_samples > num_points:
        raise ValueError("num_samples cannot be greater than the number of points.")
    if num_samples <= 0:
        return np.array([], dtype=np.int64)
    if num_points == 0:
        return np.array([], dtype=np.int64)

    if seed is not None:
        start_index = int(np.random.default_rng(seed).integers(0, num_points))
    else:
        start_index = int(np.random.randint(0, num_points))

    sampled_indices = np.zeros(num_samples, dtype=np.int64)
    distances = np.full(num_points, np.inf, dtype=np.float64)
    sampled_indices[0] = start_index
    farthest_point = points[start_index]

    for i in range(1, num_samples):
        dist = np.linalg.norm(points - farthest_point, axis=1)
        distances = np.minimum(distances, dist)
        next_farthest_index = int(np.argmax(distances))
        if distances[next_farthest_index] < dist_threshold:
            return sampled_indices[:i].copy()
        sampled_indices[i] = next_farthest_index
        farthest_point = points[next_farthest_index]

    return sampled_indices
