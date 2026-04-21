# Full End-to-End Example

This guide walks through the complete Asset Harvester pipeline: from raw NCore V4 driving logs to simulation-ready 3D Gaussian splat assets.

## Prerequisites

Make sure you have completed the [setup and checkpoint download](../README.md#user-guide) steps described in the main README.

## Download Sample Data (Optional)

Download a sample NCore V4 clip to try the pipeline:

```bash
hf download nvidia/PhysicalAI-Autonomous-Vehicles-NCore \
    --repo-type dataset \
    --local-dir ./ncore-clips \
    --include 'clips/2a6f330-5ab0-4e92-99d4-d19e406952f4/*'
```
or manually from [Hugging Face](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NCore).

## Step 0: NCore V4 Data Format

Asset Harvester consumes [NCore V4](https://nvidia.github.io/ncore/index.html) driving logs, which bundle multi-camera images, lidar point clouds, and 3D cuboid track annotations into a single clip format.

To inspect a clip before running the pipeline, use [ncore_vis](https://nvidia.github.io/ncore/tools/ncore_vis.html) — a browser-based visualizer for NCore V4 data that renders camera feeds, lidar, and cuboid overlays interactively.

## Step 1: NCore Parsing

Parse NCore V4 clip data (cameras, lidar, cuboid tracks) into multi-view object crops.

```bash
bash scripts/run_ncore_parser.sh --component-store "path/to/clip.json"
```

Using the sample data:

```bash
bash scripts/run_ncore_parser.sh \
    --component-store "ncore-clips/clips/2a6f330-5ab0-4e92-99d4-d19e406952f4/pai_02a6f330-5ab0-4e92-99d4-d19e406952f4.json"
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--component-store` | *(required)* | Clip `.json` manifest, comma-separated NCore V4 component-store paths, or `.zarr.itar` globs |
| `--output-path` | `outputs/ncore_parser/` | Output directory |
| `--segmentation-ckpt` | `checkpoints/AH_object_seg_jit.pt` | Mask2Former JIT checkpoint |
| `--camera-ids` | all 5 default cameras | Comma-separated camera sensor IDs |
| `--track-ids` | all tracks | Comma-separated track IDs to process |

## Step 2: Multiview Diffusion + Gaussian Lifting

Generate consistent multi-view images and lift them to 3D Gaussian splats via TokenGS.

```bash
bash run.sh --data-root ./outputs/ncore_parser --output-dir ./outputs/ncore_harvest
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-root` | `outputs/ncore_parser/` | Input directory with `sample_paths.json` |
| `--diffusion-ckpt` | `checkpoints/AH_multiview_diffusion.safetensors` | Diffusion model checkpoint |
| `--lifting-ckpt` | `checkpoints/AH_tokengs_lifting.safetensors` | TokenGS checkpoint |
| `--output-dir` | `outputs/` | Output directory |
| `--num-steps` | 30 | Number of diffusion inference steps |
| `--cfg-scale` | 2.0 | Classifier-free guidance scale |
| `--max-samples` | 0 (all) | Max samples to process |
| `--skip-lifting` | off | Disable TokenGS Gaussian lifting (multiview only) |
| `--offload` | off | Offload diffusion models to CPU during lifting |

Outputs per sample: `multiview/` (generated views), `3d_lifted/` (TokenGS-rendered views), `gaussians.ply`, `multiview.mp4`, `3d_lifted.mp4`.

## Step 3: Generate External Assets Metadata to use with NVIDIA Omniverse NuRec (Optional)

To use Asset Harvester with a NuRec reconstruction, generate a `metadata.yaml` file using the script below. The Step 2 output directory can then be used as input to the NuRec workflow for asset replacement and insertion, described [here](https://sw-docs.gitlab-master-pages.nvidia.com/av-sim/early-access/nurec/use-ah-assets.html).

```bash
python asset_harvester/utils/generate_external_assets_metadata.py --input-dir ./outputs/ncore_harvest
```

If you need to regenerate masks for direct image inputs, prefer the module entry point:

```bash
python -m asset_harvester.utils.image_segment --help
```

Direct file execution also works after the editable install:

```bash
python asset_harvester/utils/image_segment.py --help
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--input-dir` | *(required)* | Root of the input directory (lifting output) |

