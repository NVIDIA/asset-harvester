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

import argparse
import gc
import glob
import json
import os
import sys
import time
from functools import partial

import imageio
import numpy as np
import torch
import torchvision.transforms as T
from diffusers.schedulers import DPMSolverMultistepScheduler
from PIL import Image
from utils.image_guard import DEFAULT_IMAGE_GUARD_THRESHOLD, ImageGuard
from utils.mvd_farthest_pose import farthest_point_sampling

# Add models/ so that multiview_diffusion and tokengs (under models/) are importable
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_MODELS_PATH = os.path.join(_SCRIPT_DIR, "models")
if _MODELS_PATH not in sys.path:
    sys.path.insert(0, _MODELS_PATH)

from multiview_diffusion.data.inference_utils import build_eval_cams
from multiview_diffusion.data.nre_preproc import MVData, preproc
from multiview_diffusion.pipelines import SparseViewDiTPipeline
from multiview_diffusion.utils.model_builder import get_models

from models.camera_estimator.inference import AHCEstimator

# TokenGS Gaussian lifting (enabled by default)
try:
    from tokengs.lifting_inference import TokengsLiftingRunner

    TOKENGS_AVAILABLE = True
except ImportError:
    TOKENGS_AVAILABLE = False
    TokengsLiftingRunner = None

SAMPLE_PATHS_JSON = "sample_paths.json"



def load_sample_paths_from_json(data_root: str):
    """
    Load sample list from data_root/sample_paths.json.
    Returns list of (sample_dir_abspath, class_name) or None if file missing/invalid.
    """
    json_path = os.path.join(data_root, SAMPLE_PATHS_JSON)
    if not os.path.isfile(json_path):
        return None
    try:
        with open(json_path) as f:
            data = json.load(f)
        samples = data.get("samples")
        if not isinstance(samples, list):
            return None
        result = []
        for rel_path in samples:
            if not isinstance(rel_path, str):
                continue
            sample_dir = os.path.abspath(os.path.join(data_root, rel_path))
            parts = rel_path.replace("\\", "/").split("/")
            class_name = parts[0] if len(parts) >= 2 else os.path.basename(rel_path)
            result.append((sample_dir, class_name))
        return result
    except (json.JSONDecodeError, OSError):
        return None


def load_camera_metadata(input_dir: str) -> dict:
    """Load canonical camera metadata from input_views/camera.json."""
    camera_path = os.path.join(input_dir, "camera.json")
    if not os.path.isfile(camera_path):
        raise FileNotFoundError(f"Missing camera.json in {input_dir}")

    with open(camera_path) as f:
        return json.load(f)


def get_camera_file_paths(input_dir: str, cam_data: dict) -> tuple[list[str], list[str]]:
    """Resolve ordered frame and mask file paths from camera metadata."""
    frame_paths = [os.path.join(input_dir, filename) for filename in cam_data.get("frame_filenames", [])]
    mask_paths = [os.path.join(input_dir, filename) for filename in cam_data.get("mask_filenames", [])]
    return frame_paths, mask_paths


def load_multiview_sample(sample_dir: str, allowed_indices: list[int] | None = None) -> MVData:
    """
    Load one sample from a ncore_parser (or compatible) output directory into an MVData instance
    for use with inference preproc. Uses input_views only (frames, masks, camera.json).
    """
    input_dir = os.path.join(sample_dir, "input_views")
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Missing input_views: {input_dir}")

    cam_data = load_camera_metadata(input_dir)
    frame_filenames_all = cam_data["frame_filenames"]
    frame_paths_all, mask_paths_all = get_camera_file_paths(input_dir, cam_data)
    cam_poses_raw_all = cam_data["normalized_cam_positions"]
    dists_list_all = cam_data["cam_dists"]
    fov_list_all = cam_data["cam_fovs"]
    lwh_list = cam_data["object_lwh"]

    if allowed_indices is None:
        allowed_indices = list(range(len(frame_filenames_all)))

    frame_filenames = [frame_filenames_all[i] for i in allowed_indices]
    frame_paths = [frame_paths_all[i] for i in allowed_indices]
    mask_paths = [mask_paths_all[i] for i in allowed_indices]
    cam_poses_raw = [cam_poses_raw_all[i] for i in allowed_indices]
    dists_list = [dists_list_all[i] for i in allowed_indices]
    fov_list = [fov_list_all[i] for i in allowed_indices]

    n = len(frame_filenames)
    if n == 0:
        raise ValueError(f"No input views in {sample_dir}")

    frames = []
    masks = []
    for path in frame_paths:
        img = Image.open(path).convert("RGB")
        frames.append(np.array(img))
    for path in mask_paths:
        mask_img = Image.open(path)
        mask = np.array(mask_img)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        masks.append(mask)

    # cam_poses: (N, 3) camera positions
    cam_poses = np.array(cam_poses_raw, dtype=np.float64).reshape(n, 3)

    dists = np.array(dists_list, dtype=np.float64).reshape(n)
    fov = np.array(fov_list, dtype=np.float64).reshape(n)
    lwh = np.asarray(lwh_list, dtype=np.float64).reshape(3)

    metadata_path = os.path.join(sample_dir, "metadata.json")
    if os.path.isfile(metadata_path):
        with open(metadata_path) as f:
            meta = json.load(f)
        clip_id = meta.get("clip_id", os.path.basename(sample_dir))
    else:
        clip_id = os.path.basename(sample_dir)
    obj_id = os.path.basename(sample_dir)

    return MVData(
        clip_id=clip_id,
        obj_id=obj_id,
        frames=frames,
        cam_poses=cam_poses,
        dists=dists,
        fov=fov,
        npct="vehicle",
        lwh=lwh,
        masks=np.array(masks),
        auto_label=None,
    )


def load_image_sample(image_pairs: list[tuple[str, str]], camera_estimator: AHCEstimator) -> MVData:
    """
    Load one image sample from a path.
    """

    camera_data = camera_estimator.run(image_pairs)
    frames = []
    masks = []
    for frame_path, mask_path in image_pairs:
        img = Image.open(frame_path).convert("RGB").resize((512, 512))
        mask = Image.open(mask_path).resize((512, 512), Image.NEAREST)
        mask = np.array(mask)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        frames.append(np.array(img))
        masks.append(np.array(mask))
    track_id = os.path.basename(os.path.dirname(image_pairs[0][0]))
    raw_item = MVData(
        clip_id=track_id,
        obj_id="0",
        frames=frames,
        cam_poses=np.array(camera_data["cam_poses"], dtype=np.float32),
        dists=np.array(camera_data["dists"], dtype=np.float32),
        fov=np.array(camera_data["fov"], dtype=np.float32),
        npct="vehicle",
        lwh=np.array(camera_data["lwh"], dtype=np.float32),
        masks=masks,
        auto_label=None,
    )
    return raw_item


def discover_sample_specs(args) -> list[dict]:
    """Discover input samples before loading heavy models."""
    if args.data_root:
        assert os.path.isdir(args.data_root), "data_root must be an existing directory."
        print(f"   Loading sample paths from {args.data_root}/{SAMPLE_PATHS_JSON}")
        sample_dirs = load_sample_paths_from_json(args.data_root)
        if not sample_dirs:
            print(
                f"   Missing or invalid {SAMPLE_PATHS_JSON}. Run ncore_parser on your data first to generate this file."
            )
            return []
        if args.max_samples > 0:
            sample_dirs = sample_dirs[: args.max_samples]
        print(f"   Loaded {len(sample_dirs)} samples (max_samples={args.max_samples})")

        sample_specs = []
        for sample_dir, class_name in sample_dirs:
            try:
                input_dir = os.path.join(sample_dir, "input_views")
                cam_data = load_camera_metadata(input_dir)
                frame_paths, mask_paths = get_camera_file_paths(input_dir, cam_data)
                track_id = f"{class_name}/{os.path.basename(sample_dir)}"
                reserved_views_path = os.path.join(sample_dir, "reserved_views")
                if not os.path.isdir(reserved_views_path):
                    print(f"   Warning: no reserved_views/ in {sample_dir}")
                    reserved_views_path = None
                sample_specs.append(
                    {
                        "source": "data_root",
                        "sample_dir": sample_dir,
                        "track_id": track_id,
                        "reserved_views_path": reserved_views_path,
                        "image_pairs": list(zip(frame_paths, mask_paths)),
                        "allowed_indices": list(range(len(frame_paths))),
                    }
                )
            except Exception as e:
                print(f"   Skip {sample_dir}: {e}")
        return sample_specs

    assert os.path.isdir(args.image_dir), "image_dir must be an existing directory."
    print(f"   Loading image paths from {args.image_dir}")
    image_dirs = sorted(
        [
            os.path.join(args.image_dir, d)
            for d in os.listdir(args.image_dir)
            if os.path.isdir(os.path.join(args.image_dir, d))
        ]
    )

    sample_specs = []
    for image_dir in image_dirs:
        candidates = sorted(glob.glob(os.path.join(image_dir, "*.jpg")) + glob.glob(os.path.join(image_dir, "*.jpeg")))
        if not candidates:
            print(f"   No images found in {image_dir}")
            continue
        img_pairs = []
        for candidate in candidates:
            base = os.path.basename(candidate)
            mask_name = base.replace("frame", "mask")
            mask_name = os.path.splitext(mask_name)[0] + ".png"
            mask_path = os.path.join(image_dir, mask_name)
            if os.path.isfile(mask_path):
                img_pairs.append((candidate, mask_path))
        if not img_pairs:
            print(f"   No image pairs found in {image_dir}")
            continue
        track_id = os.path.basename(image_dir)
        sample_specs.append(
            {
                "source": "image_dir",
                "sample_dir": None,
                "track_id": track_id,
                "reserved_views_path": None,
                "image_pairs": img_pairs,
                "allowed_indices": None,
            }
        )

    if not sample_specs:
        print(f"   No images found in {args.image_dir}")
        return []

    if args.max_samples > 0:
        sample_specs = sample_specs[: args.max_samples]

    print(f"   Loaded {len(sample_specs)} image samples")
    return sample_specs


def main(args):
    """Main inference function."""
    print("\n Loading data...")
    if not args.data_root and not args.image_dir:
        print("   --data_root or --image_dir is required.")
        return

    sample_specs = discover_sample_specs(args)
    if not sample_specs:
        return

    if args.enable_image_guard and args.image_dir and not args.data_root:
        print(
            f"\n Running image guard on {len(sample_specs)} samples "
            f"(threshold={args.image_guard_threshold:.2f})..."
        )
        guard = ImageGuard(threshold=args.image_guard_threshold)
        filtered_specs = []
        total_guard_images = 0
        total_guard_time = 0.0
        try:
            guard.load()
            for spec in sample_specs:
                result = guard.moderate_sample(
                    image_pairs=spec["image_pairs"],
                    track_id=spec["track_id"],
                    output_dir=args.output_dir,
                    allowed_indices=spec["allowed_indices"] if spec["source"] == "data_root" else None,
                )
                total_guard_images += result.report["total_images"]
                total_guard_time += result.report["elapsed_seconds"]
                for image_report in result.report["images"]:
                    if image_report.get("passed") is False:
                        message = image_report.get("error")
                        if message is None:
                            message = f"label={image_report.get('label')}, score={image_report.get('score'):.3f}"
                        print(f"   Image guard skipped {image_report['image']} in {spec['track_id']} ({message})")

                if not result.kept_pairs:
                    print(f"   Skip sample {spec['track_id']}: no images passed image guard")
                    continue

                filtered_spec = dict(spec)
                filtered_spec["image_pairs"] = result.kept_pairs
                if spec["source"] == "data_root":
                    filtered_spec["allowed_indices"] = result.kept_indices
                filtered_specs.append(filtered_spec)
        finally:
            guard.unload()

        print(f"   Image guard kept {len(filtered_specs)}/{len(sample_specs)} samples")
        if total_guard_images > 0:
            print(
                f"   Image guard stats: processed {total_guard_images} images in "
                f"{total_guard_time:.2f}s total ({total_guard_time / total_guard_images:.2f}s/image)"
            )
        sample_specs = filtered_specs
        if not sample_specs:
            print("\nNo samples remaining after image guard.")
            return
    elif args.enable_image_guard:
        print("\n Image guard is only applied for --image_dir inputs; skipping for data_root.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
    }[args.precision]

    print(f"Using device: {device}, dtype: {dtype}")

    image_size = 512
    vae, cradio_model, cradio_image_processor, transformer = get_models(
        args.diffusion_checkpoint,
        device=device,
        dtype=dtype,
    )
    ahc_estimator = None
    if args.image_dir and not args.data_root:
        assert args.ahc_checkpoint, "AHC checkpoint is required when using image_dir."

        print(f"\n Loading AHC camera estimator from {args.ahc_checkpoint}...")
        ahc_estimator = AHCEstimator(
            checkpoint_path=args.ahc_checkpoint,
            device=device,
            cradio_model=cradio_model,
            cradio_image_processor=cradio_image_processor,
        )
        print("   AHC estimator loaded (backbone shared with c-radio)")

    print("\n Creating pipeline...")
    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_schedule="scaled_linear",
        prediction_type="flow_prediction",
        flow_shift=1.0,
        use_flow_sigmas=True,
    )
    pipeline = SparseViewDiTPipeline(
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        scheduler=scheduler,
        transformer=transformer,
        image_encoder=cradio_model,
        image_processor=cradio_image_processor,
    ).to(dtype)
    print("   Pipeline created (image conditioning via c-radio)")

    print("\n Setting up data preprocessing...")
    transform = T.Compose(
        [
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ]
    )

    inference_preproc = partial(
        preproc,
        image_transform=transform,
        resolution=image_size,
        conditioning_mode="n",
        eval_mode=True,
        eval_cam_sampler=build_eval_cams,
    )

    print("   Preprocessing configured")

    lifting_runner = None
    if not args.skip_gs_lifting:
        print("\n Loading TokenGS (Gaussian lifting) model...")
        if not TOKENGS_AVAILABLE:
            raise ImportError(
                "TokenGS is required for GS lifting. Install the tokengs package "
                "(e.g. pip install -e ./models/tokengs) and ensure safetensors is available."
            )
        lifting_runner = TokengsLiftingRunner(
            args.lifting_checkpoint,
            bbox_size=0.8,
            dtype=dtype,
            render_img_size=512,
        )
        print(
            f"   TokenGS loaded from {args.lifting_checkpoint} (precision: {args.precision}, "
            f"lifting {lifting_runner.img_size}x{lifting_runner.img_size}, "
            f"num_gs_tokens={lifting_runner.opt.num_gs_tokens}, render: 512x512)"
        )

    # Load and process data
    samples = []
    for spec in sample_specs:
        try:
            if spec["source"] == "data_root":
                item = load_multiview_sample(spec["sample_dir"], allowed_indices=spec["allowed_indices"])
            else:
                item = load_image_sample(spec["image_pairs"], ahc_estimator)
            samples.append((item, spec["track_id"], spec["reserved_views_path"]))
        except Exception as e:
            print(f"   Skip {spec['track_id']}: {e}")
            continue

    print(f"   Loaded {len(samples)} samples (max_samples={args.max_samples})")
    if not samples:
        print("\nNo samples remain after preprocessing.")
        return
    for sample_idx, (item, track_id, reserved_views_path) in enumerate(samples):
        print(f"\n--- Processing sample {sample_idx + 1}/{len(samples)}: {track_id} ---")
        shuffle_inds = None
        poses = np.asarray(item.cam_poses, dtype=np.float64)
        n_pose = poses.shape[0]
        if n_pose > 1:
            idx = farthest_point_sampling(poses, num_samples=min(n_pose, 4), dist_threshold=0.1)
            shuffle_inds = [int(i) for i in idx]
            print(f"   MVD conditioning: farthest-point view indices {shuffle_inds} (n={len(shuffle_inds)})")
        data_dict = inference_preproc(item, shuffle_inds=shuffle_inds)
        print(f"   Track ID: {track_id}")
        print(f"   Total views: {len(data_dict.x)}")
        print(f"   Target views: {data_dict.n_target}")
        print(f"   Conditioning views: {len(data_dict.x) - data_dict.n_target}")
        max_length = data_dict.n_target + min(4, len(data_dict.x) - data_dict.n_target)
        data_dict.x = data_dict.x[:max_length]
        data_dict.c2w_relatives = data_dict.c2w_relatives[:max_length]
        data_dict.x_white_background = data_dict.x_white_background[:max_length]
        data_dict.dists = data_dict.dists[:max_length]
        data_dict.fovs = data_dict.fovs[:max_length]
        data_dict.plucker_image = data_dict.plucker_image[:max_length]
        data_dict.relative_brightness = data_dict.relative_brightness[:max_length]
        if hasattr(data_dict, "intrinsics") and data_dict.intrinsics.shape[0] > max_length:
            data_dict.intrinsics = data_dict.intrinsics[:max_length]
        run_generation(
            pipeline=pipeline,
            data_dict=data_dict,
            args=args,
            device=device,
            dtype=dtype,
            output_subdir=track_id,  # output_dir/class_name/sample_00000/
            lifting_runner=lifting_runner,
            reserved_views_path=reserved_views_path,
        )

    print("\nDone!")


def run_generation(
    pipeline,
    data_dict,
    args,
    device,
    dtype,
    output_subdir="output",
    lifting_runner=None,
    reserved_views_path=None,
):
    """Run the generation pipeline using data_dict, with optional TokenGS lifting."""

    print(f"\n Generating {data_dict.n_target} views...")
    print(f"   Conditioning: {len(data_dict.x) - data_dict.n_target} image views (c-radio)")
    print(f"   Target views: {data_dict.n_target} to generate")
    print(f"   Inference steps: {args.num_steps}")
    print(f"   CFG scale: {args.cfg_scale}")

    with torch.no_grad():
        output = pipeline(
            data_dict=data_dict,
            num_inference_steps=args.num_steps,
            guidance_scale=args.cfg_scale,
            flow_shift=1.0,
            output_type="pil",
        )

    images = output["images"]
    print(f"   Generated {len(images)} images")

    # Save outputs
    output_dir = os.path.join(args.output_dir, output_subdir)
    print(f"\n Saving outputs to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    if reserved_views_path is not None:
        with open(os.path.join(output_dir, "reserved_views.json"), "w") as f:
            json.dump({"reserved_views": reserved_views_path}, f, indent=2)

    # Save as video
    images_np = [np.array(img) for img in images]
    imageio.v2.mimwrite(f"{output_dir}/multiview.mp4", images_np, fps=5, macro_block_size=1)
    print(f"   Saved video to {output_dir}/multiview.mp4")

    # Save individual generated views
    recon_dir = os.path.join(output_dir, "multiview")
    os.makedirs(recon_dir, exist_ok=True)
    for i, img in enumerate(images):
        img.save(os.path.join(recon_dir, f"{i}.png"))
    print(f"   Saved {len(images)} generated views to {recon_dir}")

    # Save FOV and distance info
    with open(os.path.join(recon_dir, "fov.txt"), "w") as f:
        f.write(str(data_dict.fovs[0].item()))
    with open(os.path.join(recon_dir, "dist.txt"), "w") as f:
        f.write(str(data_dict.dists[0].item()))

    if hasattr(data_dict, "lwh") and data_dict.lwh is not None:
        with open(os.path.join(recon_dir, "lwh.txt"), "w") as f:
            f.write(f"{data_dict.lwh[0]} {data_dict.lwh[1]} {data_dict.lwh[2]}")

    # save input views
    cond_view_dir = os.path.join(output_dir, "input")
    os.makedirs(cond_view_dir, exist_ok=True)

    cond_x = data_dict.x_original[data_dict.n_target:]
    cond_images = torch.clamp(127.5 * cond_x + 128.0, 0, 255).to("cpu", dtype=torch.uint8)
    cond_images = [Image.fromarray(im.permute((1, 2, 0)).cpu().numpy()) for im in cond_images]

    msk_x = data_dict.x_msk[data_dict.n_target:]
    msk_x = torch.clamp(127.5 * msk_x + 128.0, 0, 255).to("cpu", dtype=torch.uint8)
    msk_images = [Image.fromarray(im.permute((1, 2, 0)).cpu().numpy()) for im in msk_x]

    for id, (cond_image, msk_image) in enumerate(zip(cond_images, msk_images)):
        cond_image.save(os.path.join(cond_view_dir, f"frame_{id}.jpeg"))
        msk_image.save(os.path.join(cond_view_dir, f"mask_{id}.png"))

    print(f"   Saved {len(cond_images)} conditioning views and masks to {cond_view_dir}")



    # ---- Object-TokenGS Gaussian lifting ----
    if lifting_runner is not None:
        print("\n Preparing for lifting...")
        if args.offload_model_to_cpu and torch.cuda.is_available():
            print("   Offloading diffusion models to CPU (can take 1–2 min)...")
            sys.stdout.flush()
            for name in ("vae", "transformer", "image_encoder"):
                if hasattr(pipeline, name):
                    m = getattr(pipeline, name)
                    if m is not None:
                        m.to("cpu")
            pipeline.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        fov = data_dict.fovs[0].item()
        dist = data_dict.dists[0].item()
        lwh = data_dict.lwh if hasattr(data_dict, "lwh") and data_dict.lwh is not None else [1.0, 1.0, 1.0]

        print("   Running lifting...")
        t_lift_start = torch.cuda.Event(enable_timing=True)
        t_lift_end = torch.cuda.Event(enable_timing=True)
        t_lift_start.record()
        with torch.no_grad():
            gaussians = lifting_runner.run_lifting(images_np, fov, dist, lwh)
        t_lift_end.record()
        torch.cuda.synchronize()
        lift_ms = t_lift_start.elapsed_time(t_lift_end)
        print(f"   Lifting: {lift_ms / 1000:.2f}s")

        print("   Rendering out Gaussians for visualization...")
        t_render_start = torch.cuda.Event(enable_timing=True)
        t_render_end = torch.cuda.Event(enable_timing=True)
        t_render_start.record()
        rendered_images = lifting_runner.render_orbit_views(gaussians, fov, dist, lwh)
        t_render_end.record()
        torch.cuda.synchronize()
        render_ms = t_render_start.elapsed_time(t_render_end)
        print(f"   Orbit render: {render_ms / 1000:.2f}s")
        rendered_ims = [im.permute(1, 2, 0).numpy() for im in rendered_images]
        rendered_pil = [Image.fromarray(im) for im in rendered_ims]

        # Save rendered views
        lifting_dir = os.path.join(output_dir, "3d_lifted")
        os.makedirs(lifting_dir, exist_ok=True)
        for i, img in enumerate(rendered_pil):
            img.save(os.path.join(lifting_dir, f"{i}.png"))
        print(f"   Saved {len(rendered_pil)} TokenGS-rendered views to {lifting_dir}")

        # Save rendered video
        imageio.v2.mimwrite(f"{output_dir}/3d_lifted.mp4", rendered_ims, fps=5, macro_block_size=1)
        print(f"   Saved TokenGS video to {output_dir}/3d_lifted.mp4")

        # Save Gaussian PLY
        ply_path = os.path.join(output_dir, "gaussians.ply")
        lifting_runner.save_ply(gaussians, ply_path)
        print(f"   Saved Gaussian PLY to {ply_path}")

        if args.offload_model_to_cpu:
            print("   Moving diffusion models back to GPU for next sample...")
            sys.stdout.flush()
            for name in ("vae", "transformer", "image_encoder"):
                if hasattr(pipeline, name):
                    m = getattr(pipeline, name)
                    if m is not None:
                        m.to(device)
            pipeline.to(device)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Asset Harvester inference with c-radio image conditioning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:

  python run_inference.py \\
      --diffusion_checkpoint /path/to/checkpoint.safetensors \\
      --data_root /path/to/sample_data\\
      --lifting_checkpoint /path/to/model.safetensors \\
      --output_dir outputs

To disable GS lifting: add --skip_gs_lifting to the command above.

Note: This uses c-radio IMAGE conditioning (default for SparseViewDiT).
The checkpoint is automatically converted from training format to native
diffusers format at load time.
        """,
    )
    parser.add_argument(
        "--diffusion_checkpoint",
        type=str,
        default=None,
        help="Path to multiview diffusion model checkpoint (.safetensors file).",
    )
    parser.add_argument(
        "--ahc_checkpoint",
        type=str,
        default=None,
        help="Path to AHC camera estimator checkpoint (.safetensors file). Required when using image_dir.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Path to data root (class_name/sample_id/input_views with camera.json). Either data_root or image_dir is required.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Path to root dir of segmented object centric images (frame.jpeg, mask.png) to build assets from. Either data_root or image_dir is required.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="Maximum number of samples to process (0 = all, default: 0)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=30,
        help="Number of SparseViewDiT denoising steps (default: 30)",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=2.0,
        help="Classifier-free guidance scale (default: 2.0)",
    )
    parser.add_argument(
        "--precision",
        choices=["fp16", "fp32", "bf16"],
        default="bf16",
        help="Precision (default: bf16)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory (default: ./outputs)",
    )

    parser.add_argument(
        "--skip_gs_lifting",
        action="store_true",
        help="Disable TokenGS Gaussian reconstruction on generated views.",
    )
    parser.add_argument(
        "--lifting_checkpoint",
        type=str,
        default=None,
        help="Path to the TokenGS checkpoint (.safetensors). Lifting input resolution and "
        "num_gs_tokens are read from safetensors header metadata (input_res, num_gs_tokens or num_gs_token).",
    )
    parser.add_argument(
        "--offload_model_to_cpu",
        action="store_true",
        help="Offload diffusion models to CPU before TokenGS lifting to free GPU VRAM, then reload after (default: disabled)",
    )
    parser.add_argument(
        "--enable_image_guard",
        action="store_true",
        help="Enable Llama Guard image moderation on input frames before loading the main inference pipeline.",
    )
    parser.add_argument(
        "--image_guard_threshold",
        type=float,
        default=DEFAULT_IMAGE_GUARD_THRESHOLD,
        help="Unsafe score threshold for image moderation (default: 0.5).",
    )

    args = parser.parse_args()

    if not args.diffusion_checkpoint:
        parser.error("--diffusion_checkpoint is required")

    main(args)
