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

"""CLI entry point for ncore_parser."""

from __future__ import annotations

import glob
import json
import logging
from pathlib import Path
from typing import Any

import click

from ncore_parser.parser import NCoreParser
from ncore_parser.schemas import NCoreParserConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def resolve_component_store_paths(component_store_arg: str) -> list[str]:
    """Expand globs and clip manifests into component-store paths."""
    raw_paths = ["".join(path.split()) for path in component_store_arg.split(",")]
    data_paths: list[str] = []
    for raw_path in raw_paths:
        path = Path(raw_path)
        if path.suffix == ".json" and path.is_file():
            with path.open() as manifest_file:
                manifest: dict[str, Any] = json.load(manifest_file)
            component_stores = manifest.get("component_stores", [])
            data_paths.extend(str((path.parent / store["path"]).resolve()) for store in component_stores)
            continue

        expanded = sorted(glob.glob(raw_path))
        data_paths.extend(expanded if expanded else [raw_path])
    return data_paths


def write_inference_format(output_dir: Path, good_samples: dict, clip_id: str) -> None:
    """Write output in run_inference.py-compatible format.

    Creates:
        output_dir/
        ├── sample_paths.json
        └── <category>/
            └── <track_id>/
                ├── input_views/
                │   ├── camera.json
                │   ├── frame_00.jpeg
                │   ├── mask_00.png
                │   └── ...
                └── metadata.json
    """
    from PIL import Image

    sample_paths = []

    for uid, mvdata in good_samples.items():
        category = mvdata.npct
        sample_name = uid
        sample_rel = f"{category}/{sample_name}"
        sample_dir = output_dir / category / sample_name
        input_views_dir = sample_dir / "input_views"
        input_views_dir.mkdir(parents=True, exist_ok=True)

        frame_filenames = []
        mask_filenames = []
        for i in range(mvdata.frames.shape[0]):
            frame_fn = f"frame_{i:02d}.jpeg"
            img = Image.fromarray(mvdata.frames[i])
            img.save(input_views_dir / frame_fn)
            frame_filenames.append(frame_fn)

            mask_fn = f"mask_{i:02d}.png"
            mask_img = Image.fromarray(mvdata.masks_instance[i] * 255)
            mask_img.save(input_views_dir / mask_fn)
            mask_filenames.append(mask_fn)

        camera_data = {
            "frame_filenames": frame_filenames,
            "mask_filenames": mask_filenames,
            "normalized_cam_positions": mvdata.cam_poses.tolist(),
            "cam_dists": mvdata.dists.tolist(),
            "cam_fovs": mvdata.fov.tolist(),
            "object_lwh": mvdata.lwh.tolist() if mvdata.lwh is not None else [1.0, 1.0, 1.0],
        }
        with open(input_views_dir / "camera.json", "w") as f:
            json.dump(camera_data, f, indent=2)

        metadata = {
            "clip_id": mvdata.clip_id,
            "obj_id": mvdata.obj_id,
            "category": category,
        }
        with open(sample_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        sample_paths.append(sample_rel)

    with open(output_dir / "sample_paths.json", "w") as f:
        json.dump({"samples": sample_paths}, f, indent=2)

    logger.info(
        "Wrote inference-compatible output (%d samples) to %s",
        len(sample_paths),
        output_dir,
    )


@click.command()
@click.option(
    "--component-store",
    type=str,
    required=True,
    help="Path to V4 component store(s). Accepts comma-separated paths, .zarr.itar globs, or a clip .json manifest.",
)
@click.option(
    "--output-path",
    type=str,
    required=True,
    help="Output directory for extracted assets.",
)
@click.option(
    "--track-ids",
    type=str,
    required=False,
    default=None,
    help="Comma-separated list of track IDs to process. If not specified, all tracks are processed.",
)
@click.option(
    "--target-resolution",
    type=int,
    default=512,
    help="Target resolution for cropped views.",
)
@click.option(
    "--num-lidar-ref-frames",
    type=int,
    default=10,
    help="Maximum number of lidar reference frames per track.",
)
@click.option(
    "--occ-rate-threshold",
    type=float,
    default=0.5,
    help="Occlusion rate threshold for filtering views.",
)
@click.option(
    "--crop-min-area-ratio",
    type=float,
    default=0.002,
    help="Minimum bounding box area ratio to process.",
)
@click.option(
    "--camera-ids",
    type=str,
    default="camera_front_wide_120fov,camera_rear_right_70fov,camera_rear_left_70fov,camera_cross_left_120fov,camera_cross_right_120fov",
    help="Comma-separated list of camera IDs.",
)
@click.option(
    "--max-threads",
    type=int,
    default=4,
    help="Maximum threads for torch operations.",
)
@click.option(
    "--cam-pose-flip",
    type=str,
    default="1,1,-1",
    help="Comma-separated pose flip factors (x,y,z).",
)
@click.option(
    "--segmentation-ckpt",
    type=str,
    required=True,
    help="Path to Mask2Former JIT checkpoint.",
)
def main(
    component_store: str,
    output_path: str,
    track_ids: str | None,
    target_resolution: int,
    num_lidar_ref_frames: int,
    occ_rate_threshold: float,
    crop_min_area_ratio: float,
    camera_ids: str | None,
    max_threads: int,
    cam_pose_flip: str,
    segmentation_ckpt: str,
) -> None:
    """Parse ncore V4 clip data into multi-view object crops."""
    data_paths = resolve_component_store_paths(component_store)
    target_track_ids: list[str] | None = None
    if track_ids:
        target_track_ids = [t.strip() for t in track_ids.split(",")]

    target_camera_ids: list[str] | None = None
    if camera_ids:
        target_camera_ids = [c.strip() for c in camera_ids.split(",")]

    flip_factors = [int(f.strip()) for f in cam_pose_flip.split(",")]

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = NCoreParserConfig(
        target_resolution=target_resolution,
        num_lidar_ref_frames=num_lidar_ref_frames,
        cam_pose_flip=flip_factors,
        max_threads=max_threads,
        occ_rate_threshold=occ_rate_threshold,
        crop_min_area_ratio=crop_min_area_ratio,
        camera_ids=target_camera_ids or [],
        segmentation_ckpt=segmentation_ckpt,
    )

    parser = NCoreParser(config=config)

    logger.info("Processing data from: %s", data_paths)
    good_samples = parser.extract(
        src_data_paths=data_paths,
        target_root_path=output_path,
        target_track_ids=target_track_ids,
        camera_ids=target_camera_ids or config.camera_ids,
    )

    logger.info("Extracted %s good samples", len(good_samples))

    # Write inference-compatible format
    clip_id = ""
    if good_samples:
        clip_id = next(iter(good_samples.values())).clip_id
    write_inference_format(output_dir, good_samples, clip_id)

    with open(output_dir / "config.json", "w") as f:
        json.dump(config.model_dump(), f, indent=2)

    parser.cleanup()

    logger.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
