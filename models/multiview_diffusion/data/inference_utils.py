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

import warnings
from dataclasses import dataclass

warnings.filterwarnings("ignore")

import numpy as np

from multiview_diffusion.utils.config import SanaConfig


@dataclass
class SanaInferenceConfig(SanaConfig):
    ckpt_path: str | None = (
        "output/sana_mv_cond_joint_train_canonical_threedata_aug_03_07_2145_2d4a/epoch_1_step_4000.pth"
    )
    load_ema: bool = False
    work_dir: str = "output/inference"
    cfg_scale: float = 2.0
    sampler: str = "flow_dpm-solver"
    seed: int = 0
    step: int = 30
    prompt: str = "A vehicle in natural condition, white background"
    data_path: str = "exp_helper/local_test/good_samples.pkl"
    use_native_cross_attn: bool = True
    max_input_views: int = 4
    output_views: int = 16


def build_eval_cams(n_views, item, fov_scale: float = 1.0) -> list:

    elevation = 0

    dist = np.mean(item.dists)
    fov = np.mean(item.fov)
    if dist is None or fov is None:
        dist = 25
        fov = 7.5
    else:
        dist = np.mean(item.dists)
        fov = np.mean(item.fov) * fov_scale

    angle_gap = 360.0 / n_views
    cameras = []
    for azimuth in np.arange(0, 360, angle_gap):
        px = -np.cos(np.deg2rad(azimuth)) * np.cos(np.deg2rad(elevation))  # negative becuase it's camera position
        py = -np.sin(np.deg2rad(azimuth)) * np.cos(np.deg2rad(elevation))
        pz = -np.sin(np.deg2rad(elevation))
        cam_pos = np.array([px, py, pz])
        cameras.append((cam_pos, dist, fov))
    return cameras
