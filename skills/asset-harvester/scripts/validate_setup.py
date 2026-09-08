#!/usr/bin/env python3
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
"""Validate that the host meets the prerequisites for asset-harvester.

Checks (non-network):
  - conda is on PATH
  - nvidia-smi is on PATH and reports driver >= 570
  - A GCC 10-13 compiler is on PATH
  - Python >= 3.10 for the interpreter running this script (not a PATH lookup)
  - HuggingFace credentials are reported for information only (HF_TOKEN or a
    cached `hf auth login`). They are optional: needed only for gated repos —
    the NCore dataset, DINOv3, Llama Guard and SAM 3D Body. The
    asset-harvester checkpoints are public and never require them.

Usage:
    python3 scripts/validate_setup.py [--strict]

Arguments:
    --strict      Treat warnings (e.g. a GCC outside the tested range) as
                  errors. Absent HuggingFace credentials are not a warning.

Environment variables:
    HF_TOKEN      Optional. The `nvidia/asset-harvester` checkpoints are
                  public and download without it. Needed only for gated
                  repos: the PhysicalAI-Autonomous-Vehicles-NCore dataset
                  and the optional DINOv3, Llama Guard and SAM 3D Body
                  models. DINOv3 needs manual approval and its absence
                  aborts benchmark/install.sh outright. A cached
                  `hf auth login` works instead.
                  Generate one at: https://huggingface.co/settings/tokens

Exit codes:
    0 - no hard failures; warnings (e.g. a missing or out-of-range GCC) may
        still have been reported on stderr unless --strict was passed
    1 - a required prerequisite is missing, or --strict was passed and at
        least one warning was reported (details on stderr)
    2 - unexpected error raised while running a check (a crash inside an
        invoked tool is reported as a FAIL and exits 1)
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys

WARN = "WARN"
FAIL = "FAIL"
OK = "OK"

SUBPROCESS_TIMEOUT_S = 15
MIN_PYTHON_VERSION = (3, 10)
MIN_DRIVER_MAJOR = 570
MIN_GCC_MAJOR = 10
MAX_GCC_MAJOR = 13
UNEXPECTED_ERROR_EXIT = 2


def _run(cmd: list[str]) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=SUBPROCESS_TIMEOUT_S,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError:
        return 127, "", f"{cmd[0]}: not found"
    except Exception as exc:  # pragma: no cover - defensive
        return UNEXPECTED_ERROR_EXIT, "", f"{cmd[0]}: {exc}"


def check_python() -> tuple[str, str]:
    major, minor = sys.version_info.major, sys.version_info.minor
    version = f"{major}.{minor}"
    min_major, min_minor = MIN_PYTHON_VERSION
    if (major, minor) < MIN_PYTHON_VERSION:
        return FAIL, f"Python {version} is too old (need >= {min_major}.{min_minor})."
    return OK, f"Python {version}"


def check_conda() -> tuple[str, str]:
    if shutil.which("conda") is None:
        return FAIL, "conda not on PATH. Install Miniconda/Miniforge."
    rc, out, _ = _run(["conda", "--version"])
    if rc != 0:
        return FAIL, "conda failed to invoke."
    return OK, out.strip()


def check_driver() -> tuple[str, str]:
    if shutil.which("nvidia-smi") is None:
        return FAIL, f"nvidia-smi not on PATH. Install the NVIDIA driver >= {MIN_DRIVER_MAJOR}."
    rc, out, err = _run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ]
    )
    if rc != 0:
        return FAIL, f"nvidia-smi failed: {err.strip() or 'unknown'}"
    first = out.strip().splitlines()[0] if out.strip() else ""
    match = re.match(r"^(\d+)\.(\d+)", first)
    if not match:
        return WARN, f"Could not parse driver version from '{first}'."
    major = int(match.group(1))
    if major < MIN_DRIVER_MAJOR:
        return FAIL, (f"Driver {first} is too old. CUDA 12.8 needs driver >= {MIN_DRIVER_MAJOR}.")
    return OK, f"NVIDIA driver {first}"


def check_gcc() -> tuple[str, str]:
    """Advisory only.

    setup.sh ignores a generic PATH ``gcc`` and any CC/CXX you export. It
    compiles a test .cu with ``nvcc -ccbin`` against a fixed /usr/bin/gcc,
    then against the conda cross compiler resolved by name from PATH, and
    takes the first that passes. So this check can report OK for a compiler
    setup never uses, or warn when setup would succeed anyway. Treat it as a
    hint about the host toolchain -- but note ``--strict`` promotes every
    warning to a non-zero exit, so it is advisory only in the default mode.
    """
    if shutil.which("gcc") is None:
        return WARN, "gcc not on PATH (advisory); setup.sh probes /usr/bin/gcc then the conda compiler."
    rc, out, _ = _run(["gcc", "-dumpversion"])
    if rc != 0:
        return WARN, "Unable to query gcc version."
    version = out.strip().split(".")[0]
    try:
        major = int(version)
    except ValueError:
        return WARN, f"Unexpected gcc version '{out.strip()}'."
    if major < MIN_GCC_MAJOR or major > MAX_GCC_MAJOR:
        return WARN, (
            f"gcc {out.strip()} on PATH is outside the tested "
            f"{MIN_GCC_MAJOR}-{MAX_GCC_MAJOR} range (advisory). setup.sh ignores a generic "
            "PATH gcc: it probes /usr/bin/gcc, then the conda compiler by name, and picks "
            "whichever nvcc accepts."
        )
    return OK, f"gcc {out.strip()} on PATH (advisory; setup.sh selects its own compiler)"


def _expand(path: str) -> str:
    """Expand ~ and $VARS the way huggingface_hub does before using a path."""
    return os.path.expanduser(os.path.expandvars(path))


def _hf_token_path() -> str:
    """Resolve the single effective cached-token path.

    Mirrors huggingface_hub precedence: HF_TOKEN_PATH wins outright, else
    HF_HOME/token, else XDG_CACHE_HOME/huggingface/token, else the default
    ~/.cache/huggingface/token. Only the winning path is consulted — an
    explicit override must not fall back to a stale default.
    """
    token_path = os.environ.get("HF_TOKEN_PATH")
    if token_path:
        return _expand(token_path)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return os.path.join(_expand(hf_home), "token")
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = _expand(xdg) if xdg else os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "huggingface", "token")


def _cached_hf_token_path() -> str | None:
    """Return the effective cached `hf auth login` token path, if it holds one.

    A file of only whitespace is not a credential; huggingface_hub strips the
    contents, so an empty result means unauthenticated.
    """
    path = _hf_token_path()
    try:
        with open(path, encoding="utf-8") as handle:
            if handle.read().strip():
                return path
    except OSError:
        return None
    return None


def check_hf_token() -> tuple[str, str]:
    """Report HuggingFace credentials.

    Credentials are OPTIONAL: the `nvidia/asset-harvester` checkpoints are
    public. They are needed only for gated resources — the
    PhysicalAI-Autonomous-Vehicles-NCore dataset and the optional DINOv3,
    Llama Guard and SAM 3D Body models — which also require accepting each
    licence. DINOv3 approval is manual and gates benchmark/install.sh.
    """
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(var, "").strip():
            return OK, f"{var} is set"
    cached = _cached_hf_token_path()
    if cached:
        return OK, f"cached HuggingFace login found ({cached})"
    return OK, (
        "no HuggingFace credentials found — fine for the public "
        "asset-harvester checkpoints. Run `hf auth login` (or set HF_TOKEN) "
        "only if you need gated resources: the NCore dataset, DINOv3 "
        "(manual approval; required by benchmark/install.sh), Llama Guard, "
        "or SAM 3D Body."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors.",
    )
    args = parser.parse_args()

    checks = [
        ("Python", check_python),
        ("conda", check_conda),
        ("NVIDIA driver", check_driver),
        ("GCC", check_gcc),
        ("HF_TOKEN", check_hf_token),
    ]

    failures = 0
    warnings = 0
    for name, fn in checks:
        try:
            status, detail = fn()
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[{FAIL}] {name}: unexpected error: {exc}", file=sys.stderr)
            return UNEXPECTED_ERROR_EXIT

        stream = sys.stdout if status == OK else sys.stderr
        print(f"[{status}] {name}: {detail}", file=stream)
        if status == FAIL:
            failures += 1
        elif status == WARN:
            warnings += 1

    print(file=sys.stderr)
    if failures:
        print(
            f"{failures} prerequisite(s) missing; fix before running setup.sh.",
            file=sys.stderr,
        )
        return 1
    if args.strict and warnings:
        print(
            f"{warnings} warning(s) in --strict mode — treating as failure.",
            file=sys.stderr,
        )
        return 1
    if warnings:
        print(
            f"{warnings} warning(s); host is usable but review the messages.",
            file=sys.stderr,
        )
    else:
        print("All prerequisites look good.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
