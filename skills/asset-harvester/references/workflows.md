# Asset Harvester — Inference workflows

Three runnable workflows. The single-line summary lives in
`SKILL.md`; the full commands live here.

## Workflow Q — Quick start on bundled samples

`data_samples/rectified_AV_objects/` ships with the repo and
exercises the multiview-diffusion + lifting path (no parsing, no
camera estimation — intrinsics are pre-rectified).

```bash
export DATA_ROOT=data_samples/rectified_AV_objects/
export CHECKPOINT_MV=checkpoints/AH_multiview_diffusion.safetensors
export CHECKPOINT_GS=checkpoints/AH_tokengs_lifting.safetensors
export OUTPUT_DIR=outputs/harvesting

python3 run_inference.py \
    --diffusion_checkpoint "${CHECKPOINT_MV}" \
    --data_root "${DATA_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --lifting_checkpoint "${CHECKPOINT_GS}"
```

Add `--offload_model_to_cpu` to `run_inference.py` if you OOM at 16 GB.

## Workflow S — Single image (or a folder of single images)

Layout each object as one folder with a 512×512 `frame.jpeg` and
`mask.png`:

```text
YOUR_IMAGE_ROOT/
├── object_0/
│   ├── frame.jpeg
│   └── mask.png
└── object_1/
    └── ...
```

If you only have `frame.jpeg`, generate `mask.png` with the bundled
AV-object segmentation model. It always selects the **largest** detected
instance and offers no target selector, so give it an object-centric crop
where the intended object dominates the frame, and inspect the resulting
`mask.png` before running inference:

```bash
export CHECKPOINT_SEG=checkpoints/AH_object_seg_jit.pt
export IMAGE_ROOT=data_samples/OOD_images

python -m asset_harvester.utils.image_segment \
    --checkpoint "${CHECKPOINT_SEG}" \
    --image_folder "${IMAGE_ROOT}" \
    --frame_name frame.jpeg \
    --mask_name mask.png
```

Then run inference with the built-in camera estimator (no calibration
needed):

```bash
export YOUR_IMAGE_ROOT=data_samples/OOD_images
export CHECKPOINT_MV=checkpoints/AH_multiview_diffusion.safetensors
export CHECKPOINT_GS=checkpoints/AH_tokengs_lifting.safetensors
export CHECKPOINT_CAM=checkpoints/AH_camera_estimator.safetensors
export OUTPUT_DIR=outputs/harvesting_with_camera_estimate

python3 run_inference.py \
    --diffusion_checkpoint "${CHECKPOINT_MV}" \
    --ahc_checkpoint "${CHECKPOINT_CAM}" \
    --image_dir "${YOUR_IMAGE_ROOT}" \
    --output_dir "${OUTPUT_DIR}" \
    --lifting_checkpoint "${CHECKPOINT_GS}"
```

## Workflow N — End-to-end on NCore V4 driving logs

Two steps: parsing → diffusion+lifting. See
[`end-to-end-ncore.md`](end-to-end-ncore.md) for the full
walkthrough (sample-data download, argument matrices, optional
benchmark, NuRec handoff). Condensed form:

```bash
# (optional) pull a sample clip
hf download nvidia/PhysicalAI-Autonomous-Vehicles-NCore \
    --repo-type dataset \
    --local-dir ./ncore-clips \
    --include 'clips/2a6f330-5ab0-4e92-99d4-d19e406952f4/*'

# 1. NCore V4 clip → per-track object crops
bash scripts/run_ncore_parser.sh \
    --component-store "ncore-clips/clips/2a6f330-5ab0-4e92-99d4-d19e406952f4/pai_02a6f330-5ab0-4e92-99d4-d19e406952f4.json"

# 2. Multiview diffusion + Gaussian lifting
bash run.sh \
    --data-root ./outputs/ncore_parser \
    --output-dir ./outputs/ncore_harvest
```

## Configuration matrix (most common flags)

| Entry point | Flag | Default | Purpose |
|-------------|------|---------|---------|
| `run_inference.py` | `--diffusion_checkpoint` | *(required)* | SparseViewDiT `.safetensors` path |
| `run_inference.py` | `--lifting_checkpoint` | *(required unless `--skip_gs_lifting`)* | TokenGS `.safetensors`; omitting it does not skip lifting |
| `run_inference.py` | `--ahc_checkpoint` | *(optional)* | Camera estimator; required for single-image mode |
| `run_inference.py` | `--data_root` | — | Directory with `sample_paths.json` (rectified samples) |
| `run_inference.py` | `--image_dir` | — | Directory of `frame.jpeg`/`mask.png` folders (single-view mode) |
| `run_inference.py` | `--output_dir` | `outputs/` | Where to write per-sample outputs |
| `run_inference.py` | `--offload_model_to_cpu` | off | Reduce VRAM (<16 GB) |
| `run.sh` (wraps step 2) | `--num-steps` | 30 | Diffusion inference steps |
| `run.sh` | `--cfg-scale` | 2.0 | Classifier-free guidance |
| `run.sh` | `--skip-lifting` | off | Multiview only (no `.ply`) |
| `run.sh` | `--offload` | off | Offload diffusion to CPU during lifting |
| `run_ncore_parser.sh` | `--component-store` | *(required)* | NCore V4 clip manifest / component stores / zarr glob |
| `run_ncore_parser.sh` | `--camera-ids` | 5 default cameras | Comma-separated sensor IDs |
| `run_ncore_parser.sh` | `--track-ids` | all | Comma-separated track IDs |

Further flags for each wrapper are in
[`cli-reference.md`](cli-reference.md), which covers the documented
workflows rather than every option — run any entry point with `--help`
for its full surface.
