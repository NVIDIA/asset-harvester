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

# ── Prerequisites ─────────────────────────────────────────────────────
# - conda (Miniconda or Miniforge)
# - NVIDIA driver >= 570 (CUDA 12.8 compatible)
# - GCC 10–13 (tested with GCC 12.3)
# - ~16 GB GPU VRAM (add --offload for lower VRAM)
# ──────────────────────────────────────────────────────────────────────

# ── Defaults ──────────────────────────────────────────────────────────
ENV_NAME="asset-harvester"
PYTHON_VERSION="3.10"
GSPLAT_COMMIT="b60e917c95afc449c5be33a634f1f457e116ff5e"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Parse args ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --env-name)  ENV_NAME="$2"; shift 2 ;;
        --python)    PYTHON_VERSION="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash setup.sh [--env-name NAME] [--python VERSION]"
            echo "  --env-name   Conda environment name (default: asset-harvester)"
            echo "  --python     Python version (default: 3.10)"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Helpers ───────────────────────────────────────────────────────────
info()  { echo -e "\n\033[1;34m==>\033[0m \033[1m$*\033[0m"; }
ok()    { echo -e "\033[1;32m  ✓\033[0m $*"; }
fail()  { echo -e "\033[1;31m  ✗\033[0m $*" >&2; exit 1; }

# ── Pre-flight checks ────────────────────────────────────────────────
command -v conda >/dev/null 2>&1 || fail "conda not found. Install Miniconda/Miniforge first."

info "Initializing conda for this shell"
# Conda activation scripts use unbound variables — suspend nounset for conda ops.
set +u
eval "$(conda shell.bash hook 2>/dev/null)"
set -u

# ── Submodules ────────────────────────────────────────────────────────
info "Initializing git submodules"
cd "$REPO_DIR"
git submodule update --init --recursive
ok "Submodules ready"

# ── Conda environment ────────────────────────────────────────────────
if conda env list | grep -qE "^${ENV_NAME}\s"; then
    info "Conda env '${ENV_NAME}' already exists — reusing"
else
    info "Creating conda env '${ENV_NAME}' (Python ${PYTHON_VERSION})"
    conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
fi
set +u
conda activate "$ENV_NAME"
set -u
ok "Active env: $(python --version) @ $(which python)"

# ── CUDA toolkit (needed for gsplat source build) ────────────────────
info "Installing CUDA toolkit via conda"
set +u
conda install -y -c nvidia cuda-toolkit=12.8
set -u
ok "CUDA toolkit installed"

# ── Pick a host compiler that works with nvcc ────────────────────────
info "Selecting host compiler for CUDA builds"

NVCC="$(command -v nvcc || echo "")"
if [[ -z "$NVCC" ]]; then
    fail "nvcc not found. Ensure CUDA toolkit is installed."
fi

try_compiler() {
    local cc="$1" cxx="$2"
    [[ -x "$cc" ]] || return 1
    echo '__host__ int main(){return 0;}' > /tmp/_nvcc_test.cu
    "$NVCC" -ccbin "$cc" -x cu /tmp/_nvcc_test.cu -o /tmp/_nvcc_test 2>/dev/null
    local rc=$?
    rm -f /tmp/_nvcc_test.cu /tmp/_nvcc_test
    return $rc
}

FOUND_CC=""
FOUND_CXX=""

candidates=(
    "/usr/bin/gcc:/usr/bin/g++"
    "$(command -v x86_64-conda-linux-gnu-cc 2>/dev/null):$(command -v x86_64-conda-linux-gnu-c++ 2>/dev/null)"
)

for pair in "${candidates[@]}"; do
    cc="${pair%%:*}"
    cxx="${pair##*:}"
    if try_compiler "$cc" "$cxx"; then
        FOUND_CC="$cc"
        FOUND_CXX="$cxx"
        break
    fi
done

if [[ -z "$FOUND_CC" ]]; then
    fail "No compatible host compiler found for nvcc. Install GCC 10-13."
fi

export CC="$FOUND_CC"
export CXX="$FOUND_CXX"
export CUDAHOSTCXX="$FOUND_CC"
ok "Using $CC ($($CC --version | head -1))"

# ── pip dependencies (ordered to respect implicit constraints) ────────
info "Installing asset-harvester dependencies"
pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
    -r "$REPO_DIR/models/multiview_diffusion/requirements.txt"
ok "asset-harvester deps"

info "Installing ncore_parser"
pip install nvidia-ncore
pip install -e "$REPO_DIR/ncore_parser"
ok "ncore_parser"

info "Installing tokenGS (skipping gsplat — will build from source)"
pip install -e "$REPO_DIR/models/tokengs" --no-deps
# Install tokengs deps *except* gsplat (avoid throwaway PyPI binary)
pip install tyro lpips packaging safetensors scikit-image tqdm roma \
            tensorboard decord 'wandb==0.17.5' matplotlib kiui
ok "tokenGS"

# ── gsplat from source (CUDA-compatible build) ───────────────────────
export CUDA_HOME="$CONDA_PREFIX"
info "Building gsplat from source (commit ${GSPLAT_COMMIT:0:10}…)"
pip install --no-cache-dir --no-build-isolation \
    "git+https://github.com/nerfstudio-project/gsplat.git@${GSPLAT_COMMIT}"
python -c "from gsplat.cuda._backend import _C; print('gsplat CUDA ready')"
ok "gsplat CUDA verified"

# ── Install ruff for code formatting ─────────────────────────────────
info "Installing ruff for code formatting"
pip install ruff
ok "ruff installed"

# ── Done ──────────────────────────────────────────────────────────────
info "Setup complete"
echo ""
echo "  Activate with:  conda activate ${ENV_NAME}"
echo ""
