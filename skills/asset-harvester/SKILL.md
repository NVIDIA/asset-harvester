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
  Linux + conda (Miniconda/Miniforge), host Python >= 3.10 to run
  `scripts/validate_setup.py` (setup.sh builds its own 3.10 env),
  NVIDIA driver >= 570 (CUDA
  12.8), GCC 10-13 (advisory; setup.sh selects its own compiler),
  CUDA toolkit 12.8 (installed by setup.sh), ~16
  GB GPU VRAM (`--offload_model_to_cpu` lowers the lifting-stage peak only). The
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
  the inputs are outside the bundled segmentation model's AV domain
  (its exact classes are not enumerated here — see Limitations) — masks
  can otherwise be
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
`AH_camera_estimator.safetensors` (camera pose, used when calibration is
absent — constructed **only** for `--image_dir`; `--data_root` instead
requires `input_views/camera.json` per sample and skips samples lacking it),
and `AH_tokengs_lifting.safetensors` (TokenGS).

## Inputs

- **image_root** — directory of per-object folders, each with
  `frame.jpeg` (square recommended; `--image_dir` stretches any aspect to
  512×512) and `mask.png`. In `--image_dir` mode a folder
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
  `--offload`) when VRAM < ~16 GB. It offloads the diffusion models while
  lifting runs, so it lowers the lifting peak only; it is a no-op with
  `--skip_gs_lifting`.
- **HF_TOKEN** — HuggingFace access token. **Not** required for the
  `nvidia/asset-harvester` checkpoints, which are public. Required,
  together with accepting the dataset licence, for the gated
  `PhysicalAI-Autonomous-Vehicles-NCore` dataset, and for the optional
  DINOv3, Llama Guard and SAM 3D Body models — DINOv3 needs manual
  approval and its absence aborts `benchmark/install.sh` outright (obtain at
  <https://huggingface.co/settings/tokens>). A cached `hf auth login`
  works in place of the environment variable.

## Instructions

1. **Validate the host.** From this skill's own directory, run
   `python3 scripts/validate_setup.py`. Paths beginning `scripts/` in this
   document are relative to the skill directory; commands like `run.sh` and
   `scripts/run_ncore_parser.sh` are relative to the Asset Harvester
   checkout.

   It fails (exit 1) on host Python < 3.10 — the interpreter *running the
   validator*, since `setup.sh` builds its own 3.10 env — on a conda
   failure, and on the NVIDIA driver: missing or failing `nvidia-smi`, or a
   driver older than 570. GCC problems and an
   unparseable driver string are warnings that still exit 0; a missing
   `HF_TOKEN` is always reported OK, since it is only needed for gated
   resources. Pass `--strict` to make warnings fail too. Do **not** print
   `$HF_TOKEN` directly; see
   [`references/installation.md`](references/installation.md).
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
   - Single user image **with a paired mask** → Workflow S in the same
     file. Discovery skips any frame without its mask, so generate one
     first (see Workflow S) rather than expecting mask-less input to work.
   - NCore V4 driving log → Workflow N (full walkthrough in
     [`references/end-to-end-ncore.md`](references/end-to-end-ncore.md)).
5. **Execute.** If you hit OOM **during lifting** on a small card, add
   `--offload_model_to_cpu` (direct `run_inference.py`) or `--offload`
   (`run.sh`). It moves the diffusion models to CPU while TokenGS runs, so
   it lowers the lifting-stage peak only — it cannot make the diffusion
   stage fit, and is a no-op with `--skip_gs_lifting`.
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
# from skills/asset-harvester/
python3 scripts/validate_setup.py
```

Then, in the Asset Harvester checkout, after `bash setup.sh` and
`conda activate asset-harvester`:

```bash
python3 run_inference.py \
    --diffusion_checkpoint checkpoints/AH_multiview_diffusion.safetensors \
    --lifting_checkpoint   checkpoints/AH_tokengs_lifting.safetensors \
    --data_root            data_samples/rectified_AV_objects/ \
    --output_dir           outputs/harvesting
```

See Workflow Q in
[`references/workflows.md`](references/workflows.md).

### Example 2 — One masked single image → 3D asset

The bundled `data_samples/OOD_images/` already contains tracked `mask.png`
files, and `image_segment` overwrites masks in place. Segment into a copy:

```bash
STAGING=                                 # FILL IN: a staging dir, not data_samples/
: "${STAGING:?set STAGING}"
cp -r data_samples/OOD_images/. "$STAGING"/

python -m asset_harvester.utils.image_segment \
    --checkpoint checkpoints/AH_object_seg_jit.pt \
    --image_folder "$STAGING"
python3 run_inference.py \
    --diffusion_checkpoint checkpoints/AH_multiview_diffusion.safetensors \
    --ahc_checkpoint       checkpoints/AH_camera_estimator.safetensors \
    --lifting_checkpoint   checkpoints/AH_tokengs_lifting.safetensors \
    --image_dir            "$STAGING" \
    --output_dir           outputs/single
```

See Workflow S in
[`references/workflows.md`](references/workflows.md).

### Example 3 — NCore V4 clip → NuRec-ready external assets

```bash
# Assign real values first. Never paste angle-bracket placeholders into a
# shell: bash reads them as redirections, so depending on position they either
# fail to parse outright or silently redirect instead of passing an argument.
CLIP_JSON=                               # FILL IN: path to the clip .json
: "${CLIP_JSON:?set CLIP_JSON}"

bash scripts/run_ncore_parser.sh --component-store "$CLIP_JSON"
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
| `scripts/validate_setup.py` | Verify host meets Asset Harvester prerequisites (conda, driver, GCC, and whether `HF_TOKEN` is set — only needed for gated repos). No network access. | `python3 scripts/validate_setup.py` from `skills/asset-harvester/`. |

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

Linux, conda, NVIDIA driver `>= 570` (CUDA
12.8), a GCC that `nvcc` accepts (10–13 is the tested range, but
`setup.sh` never checks the version — it selects `/usr/bin/gcc` or the
conda compiler by nvcc smoke test), ~16 GB GPU VRAM, ~60 GB free disk (the four
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
- `AH_object_seg_jit.pt` is class-restricted, but the exact class names are
  not enumerated in this repository: the JIT wrapper exposes numeric labels
  only, and standalone segmentation ignores labels entirely and returns the
  largest instance. Treat the AV domain (vehicles, road users, road objects)
  as guidance, not a proven class map. Supply your own `mask.png` for arbitrary
  objects.
- Scale is **not predicted from NCore clips**. The clip's cuboid `dim` is
  carried through `object_lwh` into `multiview/lwh.txt`, and
  `generate_external_assets_metadata` writes it into `metadata.yaml` under
  the key `cuboids_dims` — that metadata file is what NuRec insertion reads,
  not the live clip. So a wrong size is traced through
  clip cuboid → `lwh.txt` → `cuboids_dims`. Note
  `generate_external_assets_metadata` is meant for that nested
  `<class_name>/<sample_id>/` layout — it reads `label_class` from the
  parent directory, so running it on flat `--image_dir` output records the
  output folder name (e.g. `single`) instead of an object class.
  In `--image_dir` mode the camera
  estimator does predict object dimensions, written to
  `multiview/lwh.txt` and written to metadata as `cuboids_dims` by
  `generate_external_assets_metadata`.
- 16 GB VRAM is the practical floor. `--offload_model_to_cpu` helps only
  during **lifting** — it moves the diffusion models to CPU while TokenGS
  runs — so it does not relieve OOM in the diffusion stage and is a no-op
  with `--skip_gs_lifting`.
- Square crops are recommended, and the exact pixel size is not fixed.
  `--image_dir` accepts any aspect ratio but **force-stretches** each frame
  and mask to 512×512, so a non-square source is distorted. `--data_root`
  instead resizes the shorter side and preserves aspect, which is where a
  non-square input can fail downstream on tensor shape.
- `benchmark/eval.py` needs a separately cloned conda env
  (`av-object-benchmark`) because `transformers>=4.56.0` conflicts
  with the main env's pinned `transformers==4.48.3`.
- Linux-only install path (CUDA 12.8).
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
| `nvcc` "unsupported GNU version" | No candidate compiler passed `setup.sh`'s nvcc probe | Install a GCC `nvcc` accepts (10–13) at `/usr/bin/gcc`. Exporting `CC`/`CXX`/`CUDAHOSTCXX` beforehand does **not** work — `setup.sh` ignores a generic PATH `gcc` and your `CC`/`CXX`, probes `/usr/bin/gcc` then the conda compiler by name, and overwrites them. |
| `CUDA error: out of memory` **during lifting** | GPU VRAM < ~16 GB | Add `--offload_model_to_cpu` (direct) or `--offload` (`run.sh`). |
| `401`/`403` when Llama Guard loads | `--enable-image-guard` pulls `meta-llama/Llama-Guard-3-11B-Vision` lazily through transformers at inference time, not via `hf download` | Accept that model's licence on HuggingFace and authenticate before enabling the flag. |
| `401 Unauthorized` from `hf download` | Hitting a gated repo (the NCore dataset, DINOv3, or SAM 3D Body) without an accepted licence or token. The `nvidia/asset-harvester` checkpoints are public and need neither. | Accept that repo's licence on its HF page, then `hf auth login`. |

Full matrix + teardown live in
[`references/troubleshooting.md`](references/troubleshooting.md).
