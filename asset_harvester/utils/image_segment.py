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


"""
Run object centric image segmentation on all frame images under a folder and write masks.
Finds every file named --frame_name under --image_folder and writes --mask_name in the same directory.
"""

import argparse
import os
import sys
from glob import glob

import numpy as np
from PIL import Image

from asset_harvester.ncore_parser.image_segmentation import Mask2FormerSegmentationEstimator


def find_frame_images(image_folder: str, frame_name: str) -> list[str]:
    """Find all paths named frame_name under image_folder (recursive)."""
    pattern = os.path.join(image_folder, "**", frame_name)
    return sorted(glob(pattern, recursive=True))


def run_segmentation_and_save_mask(
    estimator: Mask2FormerSegmentationEstimator,
    frame_path: str,
    mask_path: str,
    input_size: tuple[int, int],
) -> bool:
    """Run segmentation on frame_path and save binary mask to mask_path. Returns True on success."""
    img = Image.open(frame_path)
    if img.mode != "RGB":
        img = img.convert("RGB")

    img_array = np.array(img)
    orig_h, orig_w = img_array.shape[:2]

    semantic_seg, instance_seg = estimator.predict(img)

    model_h, model_w = input_size
    binary_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

    if len(instance_seg["classes"]) > 0:
        instance_masks = instance_seg["instance_masks"]
        num_instances = len(instance_seg["classes"])
        unpacked_masks = np.unpackbits(instance_masks).reshape(num_instances, model_h, model_w)

        areas = unpacked_masks.sum(axis=(1, 2))
        max_idx = int(np.argmax(areas))
        mask_at_model_size = unpacked_masks[max_idx].astype(np.uint8) * 255

        mask_pil = Image.fromarray(mask_at_model_size, mode="L")
        mask_pil = mask_pil.resize((orig_w, orig_h), Image.NEAREST)
        binary_mask = np.array(mask_pil)

    mask_img = Image.fromarray(binary_mask, mode="L")
    mask_img.save(mask_path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run segmentation on frame images under a folder and write masks.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the segmentation model checkpoint (JIT .pt file)",
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        required=True,
        help="Root folder to search for frame images (recursive)",
    )
    parser.add_argument(
        "--frame_name",
        type=str,
        default="frame.jpeg",
        help="Filename of input images to segment (default: frame.jpeg)",
    )
    parser.add_argument(
        "--mask_name",
        type=str,
        default="mask.png",
        help="Filename for output masks (default: mask.png)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for inference (default: cuda:0)",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        nargs=2,
        default=[512, 512],
        metavar=("H", "W"),
        help="Model input size height width (default: 512 512)",
    )
    parser.add_argument(
        "--pred_score_thr",
        type=float,
        default=0.3,
        help="Prediction score threshold (default: 0.3)",
    )
    args = parser.parse_args()

    frame_paths = find_frame_images(args.image_folder, args.frame_name)
    if not frame_paths:
        print(f"No files named '{args.frame_name}' found under {args.image_folder}")
        sys.exit(1)

    print(f"Found {len(frame_paths)} images. Loading model from {args.checkpoint}")
    input_size = tuple(args.input_size)
    estimator = Mask2FormerSegmentationEstimator(
        model_path=args.checkpoint,
        device=args.device,
        input_size=input_size,
        mean=None,
        std=None,
        pred_score_thr=args.pred_score_thr,
    )

    processed = 0
    errors = 0
    for frame_path in frame_paths:
        dir_path = os.path.dirname(frame_path)
        mask_path = os.path.join(dir_path, args.mask_name)
        try:
            run_segmentation_and_save_mask(estimator, frame_path, mask_path, input_size)
            processed += 1
        except Exception as e:
            errors += 1
            print(f"Error processing {frame_path}: {e}", file=sys.stderr)

    estimator.cleanup()
    print(f"Done. Processed: {processed}, Errors: {errors}")


if __name__ == "__main__":
    main()
