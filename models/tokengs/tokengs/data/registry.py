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

import json

from tokengs.data.static.assetharvest import AssetHarvest

try:
    S3_CONFIG = json.load(open("pbss_credentials_default.secret"))
except FileNotFoundError:
    S3_CONFIG = None

dataset_registry = {}

dataset_registry["assetharvest"] = {
    "cls": AssetHarvest,
    "kwargs": {
        "datadir": "/lustre/fs12/portfolios/nvr/projects/nvr_torontoai_videogen/users/kangxuey/sana_batch_inference_0615_22300",
        "realfocal": True,
        "bbox_size": 0.8,
    },
    "scene_scale": 1.0,
    "max_gap": 100000,
    "min_gap": 0,
}
