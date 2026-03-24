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

"""Generate metadata.yaml compatible with external_assets.py from pipeline output.

Scans the multiview/gaussian lifting output directory for completed samples
(those containing gaussians.ply) and produces a metadata.yaml at the root
so the directory can be used directly as --external-assets-dir.

Directory convention:
    <lifting_dir>/<category>/<track_id>/gaussians.ply
                                       /multiview/lwh.txt
"""

import argparse
import logging
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

PLY_FILENAME = "gaussians.ply"
LWH_FILENAME = "lwh.txt"


def find_completed_samples(input_dir: Path) -> list[Path]:
    """Return sample directories that contain a gaussians.ply file."""
    return sorted(p.parent for p in input_dir.rglob(PLY_FILENAME))


def read_cuboid_dims(sample_dir: Path) -> list[float]:
    """Read L/W/H from multiview/lwh.txt as [l, w, h]."""
    lwh_path = sample_dir / "multiview" / LWH_FILENAME
    if not lwh_path.exists():
        raise FileNotFoundError(f"Missing {lwh_path}")
    return [float(v) for v in lwh_path.read_text().strip().split()]


def generate_metadata(input_dir: Path) -> dict:
    """Build the metadata dict expected by external_assets.py."""
    assets = {}
    for sample_dir in find_completed_samples(input_dir):
        track_id = sample_dir.name
        label_class = sample_dir.parent.name
        cuboids_dims = read_cuboid_dims(sample_dir)
        ply_relative = (sample_dir / PLY_FILENAME).relative_to(input_dir)

        assets[track_id] = {
            "ply_file": str(ply_relative),
            "label_class": label_class,
            "cuboids_dims": cuboids_dims,
        }
        log.info(f"  track_id={track_id}  class={label_class}  dims={cuboids_dims}  ply={ply_relative}")

    return {"assets": assets}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate metadata.yaml for use with external_assets.py")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Root of the multiview/gaussian lifting output (becomes --external-assets-dir)",
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    metadata = generate_metadata(input_dir)

    if not metadata["assets"]:
        log.warning("No completed samples found (no gaussians.ply files)")
        return

    output_path = input_dir / "metadata.yaml"
    with open(output_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)

    log.info(f"Wrote {output_path} with {len(metadata['assets'])} asset(s)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
