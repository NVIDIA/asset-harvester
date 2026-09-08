# End-to-end NCore V4 → 3D assets (+ benchmark + NuRec)

Full walkthrough for converting a real autonomous-vehicle NCore V4
clip into `gaussians.ply` assets, optionally evaluating them, and
handing them to NuRec. Commands assume you are at the repo root with
the `asset-harvester` conda env active.

## 0. Prep

- Complete the install steps from `SKILL.md`. The checkpoints are public;
  `hf auth login` is needed only for the gated resources used below.
- Verify `checkpoints/` contains the four `AH_*` files.

## 1. (Optional) Pull a sample clip

The `PhysicalAI-Autonomous-Vehicles-NCore` dataset is **gated**: accept the
AV dataset licence on its HuggingFace page, then `hf auth login` (or set
`HF_TOKEN`). Without that this download returns 401.

```bash
hf download nvidia/PhysicalAI-Autonomous-Vehicles-NCore \
    --repo-type dataset \
    --local-dir ./ncore-clips \
    --include 'clips/2a6f330-5ab0-4e92-99d4-d19e406952f4/*'
```

Each clip is a standalone NCore V4 bundle: multi-camera images, lidar,
and 3D cuboid tracks. To browse interactively before running the
pipeline, install and run
[`ncore_vis`](https://nvidia.github.io/ncore/tools/ncore_vis.html).

## 2. NCore parsing — clip → per-track object crops

```bash
bash scripts/run_ncore_parser.sh \
    --component-store "ncore-clips/clips/2a6f330-5ab0-4e92-99d4-d19e406952f4/pai_02a6f330-5ab0-4e92-99d4-d19e406952f4.json"
```

Writes into `outputs/ncore_parser/` (overridable via `--output-path`),
including a `sample_paths.json` manifest used by step 3. To focus on
specific cameras or tracks:

```bash
bash scripts/run_ncore_parser.sh \
    --component-store "ncore-clips/clips/.../pai_....json" \
    --camera-ids camera_front_wide_120fov,camera_cross_left_120fov \
    --track-ids 42,87,103
```

## 3. Multiview diffusion + Gaussian lifting

```bash
bash run.sh \
    --data-root ./outputs/ncore_parser \
    --output-dir ./outputs/ncore_harvest
```

Tuning:

- `--num-steps 30` (default) is the sweet spot; increase for slightly
  cleaner views at proportional cost.
- `--cfg-scale 2.0` is the default; higher values tighten fidelity to
  the input view but reduce novel-view diversity.
- `--max-samples 5` cuts runs for smoke testing.
- `--skip-lifting` emits only `multiview/` + `multiview.mp4` (no
  `gaussians.ply`).
- `--offload` moves the diffusion model to CPU during lifting, cutting
  residency in the lifting stage at a mild latency cost. It does not halve
  the overall peak: TokenGS is loaded while diffusion is still resident.

Per-sample output layout is documented in `SKILL.md` and
`references/cli-reference.md`.

## 4. (Optional) Benchmark evaluation

Because `benchmark/install.sh` requires `transformers>=4.56.0` while
the main env pins `transformers==4.48.3`, clone the env:

```bash
conda create --name av-object-benchmark --clone asset-harvester
conda activate av-object-benchmark
bash benchmark/install.sh
```

`benchmark/install.sh` installs `pytorch-lightning`, `yacs`, `ninja`,
`termcolor`, and `transformers>=4.56.0`, clones
https://github.com/facebookresearch/sam-3d-body, and downloads:

- `facebook/dinov3-vith16plus-pretrain-lvd1689m` (DINOv3 ViT-H/16+) —
  **gated, manual approval** at
  https://huggingface.co/facebook/dinov3-vith16plus-pretrain-lvd1689m.
  `install.sh` runs under `set -euo pipefail` and downloads this
  unconditionally, so without access the whole benchmark install **aborts
  here** — before the SAM step below. Request access first.
- `facebook/sam-3d-body-dinov3` (gated — request access at
  https://huggingface.co/facebook/sam-3d-body-dinov3; `eval.py` falls back
  to rendering metrics if this one is unavailable).

`benchmark/eval.py` scores renders against **held-out reserved views**:
it expects `<output_dir>/<class_name>/<sample_id>/gaussians.ply` alongside
a `reserved_views.json` pointing at the ground-truth `camera.json` and
frames. `run_inference.py` writes that `reserved_views.json` only when the
input sample directory already contains a `reserved_views/` folder; the
NCore parser does **not** emit one, so the step-3 output above cannot be
scored as-is — those samples are skipped.

Run the benchmark against data that ships reserved views, such as the
released benchmark dataset below:

```bash
python benchmark/eval.py \
    --output_dir <benchmark-run-output> \
    --eval_output_dir benchmark/eval
```

Metrics (per sample and averaged): PSNR, SSIM, LPIPS, and (when
available) DINOv3 patch-embedding distance. SAM 3D Body is used as a
body-part keypoint detector to gate those embedding metrics on person
samples, not as a separate distance. See `rendering_metrics_summary.txt` under
`--eval_output_dir`.

Use the publicly released benchmark dataset
[`nvidia/NuRec-AV-Object-Benchmark`](https://huggingface.co/datasets/nvidia/NuRec-AV-Object-Benchmark)
for held-out evaluation.

## 5. (Optional) NuRec object-insertion handoff

### 5a. Orient Gaussians

```bash
python -m asset_harvester.utils.orient_gaussians_for_nurec \
    --input-dir ./outputs/ncore_harvest \
    --output-dir ./outputs/ncore_harvest_nurec
```

Use `--in-place` to mutate the existing directory instead of copying.
`--degrees` controls the Y-axis rotation applied (default 90°).

### 5b. Emit `metadata.yaml`

```bash
python asset_harvester/utils/generate_external_assets_metadata.py \
    --input-dir ./outputs/ncore_harvest_nurec
```

### 5c. Feed into NuRec

Follow the NuRec external-assets guide:
https://docs.nvidia.com/nurec/nurec/use-ah-assets.html

**Checklist:**

- [ ] PPISP **disabled** when reconstructing the 3D scene in NuRec
      (otherwise inserted assets look over-saturated).
- [ ] Source-clip cuboid dimensions reviewed — Asset Harvester does
      not predict scale; insertion reads scale from the clip.
- [ ] `metadata.yaml` sits at the root of the directory handed to
      NuRec.

## 6. Troubleshooting

See the main `SKILL.md` troubleshooting table. The most common
step-specific errors:

| Stage | Symptom | Fix |
|-------|---------|-----|
| Parsing (step 2) | Empty `sample_paths.json` | Check `--camera-ids` / `--track-ids` against the clip manifest; defaults expect all 5 canonical cameras present. |
| Diffusion (step 3) | OOM | Add `--offload` to `run.sh`; optionally drop `--num-steps` to 20. |
| Benchmark (step 4) | `transformers` import error | Activate `av-object-benchmark`, not `asset-harvester`. |
| NuRec (step 5) | Asset inserted at wrong orientation | Re-run `orient_gaussians_for_nurec` with the correct `--degrees`. |
