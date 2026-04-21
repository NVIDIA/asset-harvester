#!/usr/bin/env bash

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

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AH_PUBLIC_DIR="$(dirname "$SCRIPT_DIR")"

usage() {
    cat <<EOF
Usage: bash scripts/run_ncore_parser.sh --component-store <path> [options]

Parse ncore V4 clip data into multi-view object crops.

Required:
  --component-store       Comma-separated ncore V4 component-store paths
                          Supports .zarr.itar globs and clip .json manifests

Optional:
  --output-path           Output directory (default: outputs/ncore_parser/)
  --segmentation-ckpt     Mask2Former JIT checkpoint
                          (default: checkpoints/AH_object_seg_jit.pt)
  --camera-ids            Comma-separated camera sensor IDs
                          (default: camera_front_wide_120fov,
                                    camera_rear_right_70fov,
                                    camera_rear_left_70fov,
                                    camera_cross_left_120fov,
                                    camera_cross_right_120fov)
  --track-ids             Comma-separated track IDs to process (default: all)
  --help                  Show this help message

Any extra arguments are passed through to ncore-parser.
EOF
    exit "${1:-0}"
}

COMPONENT_STORE=""
OUTPUT_PATH="${AH_PUBLIC_DIR}/outputs/ncore_parser"
SEG_CKPT="${AH_PUBLIC_DIR}/checkpoints/AH_object_seg_jit.pt"
CAMERA_IDS=""
TRACK_IDS=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --component-store) COMPONENT_STORE="$2"; shift 2 ;;
        --output-path)  OUTPUT_PATH="$2"; shift 2 ;;
        --segmentation-ckpt) SEG_CKPT="$2"; shift 2 ;;
        --camera-ids)   CAMERA_IDS="$2"; shift 2 ;;
        --track-ids)    TRACK_IDS="$2"; shift 2 ;;
        --help|-h)      usage 0 ;;
        *)              EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [ -z "${COMPONENT_STORE}" ]; then
    echo "ERROR: --component-store is required"
    echo ""
    usage 1
fi

if [ ! -f "${SEG_CKPT}" ]; then
    echo "ERROR: segmentation checkpoint not found at ${SEG_CKPT}"
    exit 1
fi

echo "Component store:    ${COMPONENT_STORE}"
echo "Output path:        ${OUTPUT_PATH}"
echo "Segmentation ckpt:  ${SEG_CKPT}"
[ -n "${CAMERA_IDS}" ] && echo "Camera IDs:         ${CAMERA_IDS}"
[ -n "${TRACK_IDS}" ] && echo "Track IDs:          ${TRACK_IDS}"
echo ""

CAMERA_ARGS=()
if [ -n "${CAMERA_IDS}" ]; then
    CAMERA_ARGS+=(--camera-ids "${CAMERA_IDS}")
fi

TRACK_ARGS=()
if [ -n "${TRACK_IDS}" ]; then
    TRACK_ARGS+=(--track-ids "${TRACK_IDS}")
fi

ncore-parser \
    --component-store "${COMPONENT_STORE}" \
    --output-path "${OUTPUT_PATH}" \
    --segmentation-ckpt "${SEG_CKPT}" \
    ${CAMERA_ARGS[@]+"${CAMERA_ARGS[@]}"} \
    ${TRACK_ARGS[@]+"${TRACK_ARGS[@]}"} \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
