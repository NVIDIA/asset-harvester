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

usage() {
    cat <<EOF
Usage: bash run.sh [options]

Run multiview diffusion, optionally followed by TokenGS Gaussian lifting.
By default, lifting is enabled. Pass --skip-lifting to run multiview only.

Options:
  --data-root         Input directory with sample_paths.json
                      (default: outputs/ncore_parser/)
  --image-dir         Input directory with frame/mask pairs for direct inference
  --diffusion-ckpt    Path to multiview diffusion .safetensors checkpoint
                      (default: checkpoints/AH_multiview_diffusion.safetensors)
  --ahc-ckpt          Path to AHC .safetensors checkpoint
                      (default: checkpoints/AH_camera_estimator.safetensors)
  --lifting-ckpt      Path to TokenGS .safetensors checkpoint
                      (default: checkpoints/AH_tokengs_lifting.safetensors)
  --output-dir        Output directory (default: outputs/)
  --num-steps         Number of diffusion inference steps (default: 30)
  --cfg-scale         Classifier-free guidance scale (default: 2.0)
  --max-samples       Max samples to process, 0 = all (default: 0)
  --skip-lifting      Disable TokenGS Gaussian lifting (multiview only)
  --offload           Offload diffusion models to CPU during lifting
  --enable-image-guard
                      Enable Llama Guard image moderation on --image-dir inputs
  --image-guard-threshold
                      Unsafe score threshold for the image guard (default: 0.5)
  --help              Show this help message

Any extra arguments are passed through to run_inference.py.
EOF
    exit "${1:-0}"
}

DATA_ROOT="${SCRIPT_DIR}/outputs/ncore_parser"
IMAGE_DIR=""
DIFFUSION_CKPT=""
AHC_CKPT="${SCRIPT_DIR}/checkpoints/AH_camera_estimator.safetensors"
LIFTING_CKPT="${SCRIPT_DIR}/checkpoints/AH_tokengs_lifting.safetensors"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"
NUM_STEPS=""
CFG_SCALE=""
MAX_SAMPLES=0
SKIP_LIFTING=false
OFFLOAD=false
ENABLE_IMAGE_GUARD=false
IMAGE_GUARD_THRESHOLD=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root)        DATA_ROOT="$2"; shift 2 ;;
        --image-dir)        IMAGE_DIR="$2"; shift 2 ;;
        --diffusion-ckpt)   DIFFUSION_CKPT="$2"; shift 2 ;;
        --ahc-ckpt)         AHC_CKPT="$2"; shift 2 ;;
        --lifting-ckpt)     LIFTING_CKPT="$2"; shift 2 ;;
        --output-dir)       OUTPUT_DIR="$2"; shift 2 ;;
        --num-steps)        NUM_STEPS="$2"; shift 2 ;;
        --cfg-scale)        CFG_SCALE="$2"; shift 2 ;;
        --max-samples)      MAX_SAMPLES="$2"; shift 2 ;;
        --skip-lifting)     SKIP_LIFTING=true; shift ;;
        --offload)          OFFLOAD=true; shift ;;
        --enable-image-guard) ENABLE_IMAGE_GUARD=true; shift ;;
        --image-guard-threshold) IMAGE_GUARD_THRESHOLD="$2"; shift 2 ;;
        --help|-h)          usage 0 ;;
        *)                  EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# --- Resolve defaults ---

[ -z "${DIFFUSION_CKPT}" ] && DIFFUSION_CKPT="${SCRIPT_DIR}/checkpoints/AH_multiview_diffusion.safetensors"
[ -z "${NUM_STEPS}" ] && NUM_STEPS=30
[ -z "${CFG_SCALE}" ] && CFG_SCALE=2.0

# --- Validate inputs ---

if [ ! -f "${DIFFUSION_CKPT}" ]; then
    echo "ERROR: diffusion checkpoint not found at ${DIFFUSION_CKPT}"
    exit 1
fi

if [ "${SKIP_LIFTING}" = false ] && [ ! -f "${LIFTING_CKPT}" ]; then
    echo "ERROR: TokenGS checkpoint not found at ${LIFTING_CKPT}"
    echo "Pass --skip-lifting to run multiview only, or provide --lifting-ckpt."
    exit 1
fi

if [ -n "${IMAGE_DIR}" ]; then
    if [ ! -d "${IMAGE_DIR}" ]; then
        echo "ERROR: image dir not found at ${IMAGE_DIR}"
        exit 1
    fi
    if [ ! -f "${AHC_CKPT}" ]; then
        echo "ERROR: AHC checkpoint not found at ${AHC_CKPT}"
        exit 1
    fi
elif [ ! -f "${DATA_ROOT}/sample_paths.json" ]; then
    echo "ERROR: sample_paths.json not found in ${DATA_ROOT}"
    echo "Run ncore_parser first: bash ncore_parser/run.sh --component-store <path>"
    exit 1
fi

# --- Build command ---

echo "Diffusion checkpoint: ${DIFFUSION_CKPT}"
if [ -n "${IMAGE_DIR}" ]; then
    echo "Image dir:            ${IMAGE_DIR}"
    echo "AHC checkpoint:       ${AHC_CKPT}"
else
    echo "Data root:            ${DATA_ROOT}"
fi
echo "Output dir:           ${OUTPUT_DIR}"
echo "Steps: ${NUM_STEPS}   CFG: ${CFG_SCALE}   Max samples: ${MAX_SAMPLES}"
if [ "${SKIP_LIFTING}" = true ]; then
    echo "Lifting:              DISABLED (multiview only)"
else
    echo "Lifting checkpoint:   ${LIFTING_CKPT}"
fi
if [ "${ENABLE_IMAGE_GUARD}" = true ]; then
    echo "Image guard:          ENABLED"
    echo "Guard threshold:      ${IMAGE_GUARD_THRESHOLD:-0.5}"
fi
echo ""

LIFTING_FLAGS=()
if [ "${SKIP_LIFTING}" = true ]; then
    LIFTING_FLAGS+=(--skip_gs_lifting)
else
    LIFTING_FLAGS+=(--lifting_checkpoint "${LIFTING_CKPT}")
fi

CFG_FLAGS=()
if [ -n "${CFG_SCALE}" ]; then
    CFG_FLAGS+=(--cfg_scale "${CFG_SCALE}")
fi

OFFLOAD_FLAGS=()
if [ "${OFFLOAD}" = true ]; then
    OFFLOAD_FLAGS+=(--offload_model_to_cpu)
fi

IMAGE_GUARD_FLAGS=()
if [ "${ENABLE_IMAGE_GUARD}" = true ]; then
    IMAGE_GUARD_FLAGS+=(--enable_image_guard)
fi
if [ -n "${IMAGE_GUARD_THRESHOLD}" ]; then
    IMAGE_GUARD_FLAGS+=(--image_guard_threshold "${IMAGE_GUARD_THRESHOLD}")
fi

INPUT_FLAGS=()
if [ -n "${IMAGE_DIR}" ]; then
    INPUT_FLAGS+=(--image_dir "${IMAGE_DIR}" --ahc_checkpoint "${AHC_CKPT}")
else
    INPUT_FLAGS+=(--data_root "${DATA_ROOT}")
fi

python3 "${SCRIPT_DIR}/run_inference.py" \
    --diffusion_checkpoint "${DIFFUSION_CKPT}" \
    "${INPUT_FLAGS[@]}" \
    --num_steps "${NUM_STEPS}" \
    --max_samples "${MAX_SAMPLES}" \
    --output_dir "${OUTPUT_DIR}" \
    "${CFG_FLAGS[@]+"${CFG_FLAGS[@]}"}" \
    "${LIFTING_FLAGS[@]}" \
    "${OFFLOAD_FLAGS[@]}" \
    "${IMAGE_GUARD_FLAGS[@]+"${IMAGE_GUARD_FLAGS[@]}"}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
