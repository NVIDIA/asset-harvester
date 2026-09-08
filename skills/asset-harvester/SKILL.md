---
name: asset-harvester
description: >-
  Use to install and run NVIDIA Asset Harvester (Apache-2.0) to
  extract per-object 3D Gaussian Splat assets (`gaussians.ply`) from
  AV NCore V4 clips or masked single images via SparseViewDiT +
  TokenGS, optionally producing `metadata.yaml` for NuRec object
  insertion. Do NOT use for full-scene reconstruction (use the `nurec`
  skills), or for maskless inputs outside the bundled segmentation
  model's classes (masks for AV objects can be generated with
  `image_segment`).
version: "0.1.1"
tools:
  - Shell
  - Read
  - Write
license: CC-BY-4.0 AND Apache-2.0
compatibility: >-
  Linux + conda (Miniconda/Miniforge), NVIDIA driver >= 570 (CUDA
  12.8), GCC 10-13, CUDA toolkit 12.8 (installed by setup.sh), ~16
  GB GPU VRAM (use `--offload_model_to_cpu` for less). The
  `nvidia/asset-harvester` checkpoints are public; HF_TOKEN plus
  license acceptance is needed only for gated resources: the NCore
  dataset, and the optional DINOv3, Llama Guard and SAM 3D Body models. Egress to
  github.com, huggingface.co, pypi.org, download.pytorch.org, and
  the configured conda channels.
dependencies:
  - bash
  - conda
  - git
  - python3
metadata:
  author: NVIDIA NRS <nurec-skills@nvidia.com>
  tags:
    - asset-harvester
    - autonomous-vehicles
    - 3d-reconstruction
    - gaussian-splatting
    - simulation
  upstream: https://github.com/NVIDIA/asset-harvester
  project_page: https://research.nvidia.com/labs/sil/projects/asset-harvester/
  paper: https://arxiv.org/abs/2604.18468
  hf_model: https://huggingface.co/nvidia/asset-harvester
  hf_demo: https://huggingface.co/spaces/nvidia/asset-harvester
  hf_dataset: https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NCore
  hf_benchmark: https://huggingface.co/datasets/nvidia/NuRec-AV-Object-Benchmark
  time-estimate: "45min (20min setup + 15min inference + evaluation)"
---

# Asset Harvester

## Purpose

Install and drive NVIDIA Asset Harvester to extract per-object 3D
Gaussian Splat assets from sparse autonomous-vehicle (AV) object
observations — either a multi-view crop pulled from an NCore V4
driving log or a single masked image. The output is a
simulation-ready `gaussians.ply` plus optional `metadata.yaml` that
NVIDIA Omniverse NuRec can ingest as an external asset. Apache-2.0
upstream code lives at <https://github.com/NVIDIA/asset-harvester>.

## When to Use / When NOT to Use

**Use this skill when:**

- The user has AV clips or masked single images and wants per-object
  3D assets via the SparseViewDiT + TokenGS pipeline.
- The user has NCore V4 driving-log clips and wants per-track 3D
  assets for simulation.
- The user asks about `SparseViewDiT`, `TokenGS`, or wants to
  reproduce the Asset Harvester paper / HF Space demo locally.
- The user wants `.ply` Gaussians + `metadata.yaml` suitable for
  NVIDIA Omniverse **NuRec** object insertion.

**Do NOT use this skill when:**

- The user wants a full-scene reconstruction (use the `nurec` skills).
- The user has neither per-object masks nor AV-style object crops, and
  the inputs are outside the bundled segmentation model's classes
  (vehicles, VRUs, cyclists, road objects) — masks can otherwise be
  generated with `image_segment`, see Workflow S.
- The user wants text-to-3D, indoor scans, or non-AV imagery —
  out of distribution.
- The user wants to ingest raw sensor data into NCore V4 (use the
  `ncore` skill first).
- The user wants to re-train SparseViewDiT or TokenGS — this skill
  is install + inference only.
- The user just wants the no-install demo: point them at
  <https://huggingface.co/spaces/nvidia/asset-harvester>.

## Background

Open-source (Apache-2.0) image-to-3D pipeline pairing
**SparseViewDiT** (multiview diffusion, 16 consistent views) with
**TokenGS** (feed-forward Gaussian lifting):

```text
NCore V4 clip ──► NCore parsing ──► SparseViewDiT (16-view diffusion)
              ──► TokenGS lifting ──► gaussians.ply
              ──► (optional) metadata.yaml for NuRec object insertion
```

Single HF repo `nvidia/asset-harvester` ships four checkpoints:
`AH_object_seg_jit.pt` (AV-object Mask2Former),
`AH_multiview_diffusion.safetensors` (SparseViewDiT),
`AH_camera_estimator.safetensors` (camera pose, used when calibration
is absent), and `AH_tokengs_lifting.safetensors` (TokenGS).

## Inputs

- **image_root** — directory of per-object folders, each with
  `frame.jpeg` (512×512) and `mask.png`. In `--image_dir` mode a folder
  without a paired mask is skipped, so generate masks first with
  `image_segment` (Workflow S). `component_store` is a separate,
  parser-side input and does not remove this requirement.
- **component_store** — path to NCore V4 clip `.json` manifest,
  comma-separated component-store paths, or `.zarr.itar` glob
  (required when running the NCore parsing path).
- **output_dir** — where per-sample outputs (`gaussians.ply`,
  `multiview/`, `*.mp4`) are written (default
  `outputs/`).
- **offload_flag** — enable CPU offload (`--offload_model_to_cpu` /
  `--offload`) when VRAM < ~16 GB.
- **HF_TOKEN** — HuggingFace access token. **Not** required for the
  `nvidia/asset-harvester` checkpoints, which are public. Required,
  together with accepting the dataset licence, for the gated
  `PhysicalAI-Autonomous-Vehicles-NCore` dataset, and for the optional
  DINOv3, Llama Guard and SAM 3D Body models — DINOv3 needs manual
  approval and its absence aborts `benchmark/install.sh` outright (obtain at
  <https://huggingface.co/settings/tokens>). A cached `hf auth login`
  works in place of the environment variable.

## Instructions

1. **Validate the host.** Have the agent execute this skill's
   `scripts/validate_setup.py` via its standard script runner —
   e.g. `run_script("scripts/validate_setup.py")` or, from this
   skill's own directory, `python scripts/validate_setup.py`. Paths
   beginning `scripts/` in this document are relative to the skill
   directory; commands like `run.sh` and `scripts/run_ncore_parser.sh`
   are relative to the Asset Harvester checkout. It checks conda, the
   NVIDIA driver, GCC and `HF_TOKEN`, and exits non-zero when Python,
   conda or the driver fail. A missing GCC is a warning that still exits
   zero; absent HuggingFace credentials are reported as OK, since they are
   only needed for gated resources. Pass `--strict` to make warnings fail
   too. Do **not** print `$HF_TOKEN` directly;
   see [`references/installation.md`](references/installation.md).
2. **Install.** Use the one-shot `bash setup.sh` path unless the
   user asks for a manual install. Full commands and the pinned
   `gsplat` step are in
   [`references/installation.md`](references/installation.md).
3. **Download checkpoints.** `hf download nvidia/asset-harvester
   --local-dir checkpoints` — this repo is public, no login needed.
   Authenticate only for gated resources (see
   [`references/installation.md`](references/installation.md)).
4. **Pick the inference path:**
   - Bundled demo → Workflow Q in
     [`references/workflows.md`](references/workflows.md).
   - Single user image (+/- mask) → Workflow S in the same file.
   - NCore V4 driving log → Workflow N (full walkthrough in
     [`references/end-to-end-ncore.md`](references/end-to-end-ncore.md)).
5. **Execute with appropriate VRAM flag.** If `< 16 GB` VRAM, add
   `--offload_model_to_cpu` (direct `run_inference.py`) or
   `--offload` (`run.sh`).
6. **Validate outputs.** Confirm `multiview/` and `multiview.mp4` exist
   under the per-sample output directory. With lifting enabled (the
   default) also expect `gaussians.ply` and `3d_lifted.mp4`; with
   `--skip_gs_lifting` neither is produced. Every `--data_root` run,
   including the bundled Workflow Q, nests samples as
   `<output_dir>/<class_name>/<sample_id>/`; only `--image_dir` is flat.
7. **(Optional) Benchmark.** Clone the env to `av-object-benchmark`
   and run `benchmark/eval.py` for PSNR / LPIPS / SSIM and DINOv3
   embedding metrics. See
   [`references/end-to-end-ncore.md`](references/end-to-end-ncore.md).
8. **(Optional) Hand off to NuRec.** Rotate Gaussians with
   `orient_gaussians_for_nurec`, emit `metadata.yaml`, then follow
   the [NuRec external-assets docs](https://docs.nvidia.com/nurec/nurec/use-ah-assets.html).
   Metadata generation expects the nested `<class_name>/<sample_id>/`
   layout a `--data_root` run produces — it reads `label_class` from the
   parent directory. On flat `--image_dir` output it records the output
   folder name instead, so re-parent those samples under a real class
   directory first.

## Examples

Three concrete entry points. Each one points at the workflow file
with the full command; nothing here is meant to be copy-pasted in
isolation.

### Example 1 — Smoke-test the install with bundled samples

```bash
python scripts/validate_setup.py          # from the skill dir; then `bash setup.sh` in the checkout
python3 run_inference.py \
    --diffusion_checkpoint checkpoints/AH_multiview_diffusion.safetensors \
    --lifting_checkpoint   checkpoints/AH_tokengs_lifting.safetensors \
    --data_root            data_samples/rectified_AV_objects/ \
    --output_dir           outputs/harvesting
```

See Workflow Q in
[`references/workflows.md`](references/workflows.md).

### Example 2 — One masked single image → 3D asset

```bash
python -m asset_harvester.utils.image_segment \
    --checkpoint checkpoints/AH_object_seg_jit.pt \
    --image_folder data_samples/OOD_images
python3 run_inference.py \
    --diffusion_checkpoint checkpoints/AH_multiview_diffusion.safetensors \
    --ahc_checkpoint       checkpoints/AH_camera_estimator.safetensors \
    --lifting_checkpoint   checkpoints/AH_tokengs_lifting.safetensors \
    --image_dir            data_samples/OOD_images \
    --output_dir           outputs/single
```

See Workflow S in
[`references/workflows.md`](references/workflows.md).

### Example 3 — NCore V4 clip → NuRec-ready external assets

```bash
bash scripts/run_ncore_parser.sh --component-store <clip.json>
bash run.sh --data-root ./outputs/ncore_parser --output-dir ./outputs/ncore_harvest
python -m asset_harvester.utils.orient_gaussians_for_nurec \
    --input-dir ./outputs/ncore_harvest \
    --output-dir ./outputs/ncore_harvest_nurec
python asset_harvester/utils/generate_external_assets_metadata.py \
    --input-dir ./outputs/ncore_harvest_nurec
```

Full walkthrough including sample-clip download, the benchmark
flow, and the NuRec PPISP caveat lives in
[`references/end-to-end-ncore.md`](references/end-to-end-ncore.md).

## Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/validate_setup.py` | Verify host meets Asset Harvester prerequisites (conda, driver, GCC, and whether `HF_TOKEN` is set — only needed for gated repos). No network access. | Invoke via the agent's `run_script` helper, or `python scripts/validate_setup.py`. |

## Output Format

Per input sample (image or NCore track) the pipeline writes:

```text
${OUTPUT_DIR}/<sample_id>/      # --image_dir is flat; every --data_root
                               # run nests as <class_name>/<sample_id>/
├── multiview/                  # 16 RGB views generated by SparseViewDiT
├── multiview.mp4
├── gaussians.ply               # lifting only — absent with --skip_gs_lifting
└── 3d_lifted.mp4               # lifting only — TokenGS-rendered orbit views
```

When the NuRec handoff runs, `metadata.yaml` is additionally written
at the root of the oriented output directory.

## Prerequisites

Linux (Ubuntu 22.04 tested), conda, NVIDIA driver `>= 570` (CUDA
12.8), GCC 10–13, ~16 GB GPU VRAM, ~60 GB free disk (the four
checkpoints alone are ~12.8 GB, plus two conda envs, caches, any
downloaded clips and per-sample outputs), and egress to
`github.com`, `huggingface.co`, `pypi.org`,
`download.pytorch.org` and the configured conda channels. The
prerequisite check is
`scripts/validate_setup.py` (add `--strict` to fail on warnings);
secret-handling
guidance lives in
[`references/installation.md`](references/installation.md).

## References

- [`references/installation.md`](references/installation.md) —
  one-shot + manual install, checkpoint download, safe `HF_TOKEN`
  handling.
- [`references/workflows.md`](references/workflows.md) — Workflows
  Q (bundled), S (single image), N (NCore) plus a configuration
  matrix.
- [`references/end-to-end-ncore.md`](references/end-to-end-ncore.md)
  — full NCore V4 walkthrough including benchmark eval in the cloned
  `av-object-benchmark` env, and the NuRec handoff checklist.
- [`references/cli-reference.md`](references/cli-reference.md) —
  selected flag matrix for `run_inference.py`, `run.sh`,
  `run_ncore_parser.sh`, `image_segment`,
  `orient_gaussians_for_nurec`,
  `generate_external_assets_metadata.py`.
- [`references/troubleshooting.md`](references/troubleshooting.md)
  — extended error matrix and full teardown / disk-cleanup recipe.
- Sibling skills: `ncore` (NCore V4 ingest), the `nurec` skills
  (NuRec scene reconstruction, `export-external-assets` packaging,
  sample NCore clips, benchmark dataset).
- Upstream README: <https://github.com/NVIDIA/asset-harvester>
- Project page: <https://research.nvidia.com/labs/sil/projects/asset-harvester/>
- HF model: <https://huggingface.co/nvidia/asset-harvester>
- Live demo: <https://huggingface.co/spaces/nvidia/asset-harvester>
- NuRec external-assets guide: <https://docs.nvidia.com/nurec/nurec/use-ah-assets.html>

## Limitations

- AV-only domain. Non-road / non-AV objects are out of distribution.
- `AH_object_seg_jit.pt` is class-restricted (vehicles, VRUs,
  cyclists, road objects). Supply your own `mask.png` for arbitrary
  objects.
- Scale is **not predicted from NCore clips** — NuRec insertion reads it
  from the source clip's cuboid tracks, and
  `generate_external_assets_metadata` is meant for that nested
  `<class_name>/<sample_id>/` layout — it reads `label_class` from the
  parent directory, so running it on flat `--image_dir` output records the
  output folder name (e.g. `single`) instead of an object class.
  In `--image_dir` mode the camera
  estimator does predict object dimensions, written to
  `multiview/lwh.txt` and consumed as `cuboid_dims` by
  `generate_external_assets_metadata`.
- 16 GB VRAM is the practical floor; lower-VRAM users must offload
  to CPU (slower).
- Inputs must be 512×512 square crops.
- `benchmark/eval.py` needs a separately cloned conda env
  (`av-object-benchmark`) because `transformers>=4.56.0` conflicts
  with the main env's pinned `transformers==4.48.3`.
- Linux-only install path (tested on Ubuntu 22.04 + CUDA 12.8).
- `benchmark/install.sh` runs under `set -euo pipefail` and downloads
  `facebook/dinov3-vith16plus-pretrain-lvd1689m` unconditionally. That
  repo is gated (manual approval), so without accepted access the
  benchmark install aborts there — before the SAM 3D Body step, which
  does have a fallback. Request access first, or skip the benchmark.
  `eval.py` itself degrades to PSNR/LPIPS/SSIM when the SAM 3D Body
  detector is unavailable.

## Troubleshooting (top 4)

| Error | Cause | Solution |
|-------|-------|----------|
| `gsplat` import / CUDA ABI mismatch | Installed `gsplat` from PyPI wheel instead of the pinned commit | Reinstall from the pinned source commit; see [`references/installation.md`](references/installation.md). |
| `nvcc` "unsupported GNU version" | GCC outside 10–13 on PATH | Install GCC 12 and export `CC`/`CXX`/`CUDAHOSTCXX` before `setup.sh`. |
| `CUDA error: out of memory` | GPU VRAM < ~16 GB | Add `--offload_model_to_cpu` (direct) or `--offload` (`run.sh`). |
| `401 Unauthorized` from `hf download` | Hitting a gated repo (the NCore dataset, DINOv3, Llama Guard, or SAM 3D Body) without an accepted licence or token. The `nvidia/asset-harvester` checkpoints are public and need neither. | Accept that repo's licence on its HF page, then `hf auth login`. |

Full matrix + teardown live in
[`references/troubleshooting.md`](references/troubleshooting.md).
