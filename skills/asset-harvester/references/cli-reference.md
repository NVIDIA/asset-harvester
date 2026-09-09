# Asset Harvester — CLI reference

The flags you need for the documented workflows, per wrapper and entry
point. See the top of `SKILL.md` for the common-case cookbook. This is a
selected set, not the complete surface — `run.sh`, `image_segment`, the
NCore parser passthrough, `benchmark/eval.py` and the installed console
scripts accept further options; run any of them with `--help` for the
authoritative list.

## `run_inference.py` (direct image-to-3D entry)

| Flag | Default | Purpose |
|------|---------|---------|
| `--diffusion_checkpoint` | *(required)* | Path to `AH_multiview_diffusion.safetensors`. |
| `--lifting_checkpoint` | *(argparse allows omission; required in practice unless `--skip_gs_lifting`)* | Path to `AH_tokengs_lifting.safetensors`. Omitting it does **not** skip lifting: TokenGS is still constructed, with randomly initialised parameters and no checkpoint loaded, so the output is not meaningful. Use `--skip_gs_lifting` for multiview-only output. |
| `--ahc_checkpoint` | *(optional)* | Path to `AH_camera_estimator.safetensors`. **Required** when using `--image_dir` (single-view mode) — an AV camera pose is estimated from the single input. |
| `--data_root` | — | Root of a rectified-samples directory containing `sample_paths.json` (e.g. `data_samples/rectified_AV_objects/` or `outputs/ncore_parser/`). Pass exactly one of this or `--image_dir`; it is not enforced, and `--data_root` takes precedence if both are given. |
| `--image_dir` | — | Root of a single-view directory with `<object_id>/{frame.jpeg,mask.png}` sub-folders. Pass exactly one of this or `--data_root`; `--data_root` wins if both are set. |
| `--output_dir` | `outputs/` | Per-sample outputs (`multiview/`, `gaussians.ply`, `*.mp4`). |
| `--skip_gs_lifting` | off | Disable TokenGS Gaussian reconstruction; emits multiview outputs only. |
| `--num_steps` | see `--help` | Diffusion inference steps. |
| `--cfg_scale` | see `--help` | Classifier-free guidance scale. |
| `--max_samples` | all | Cap the number of samples processed. |
| `--precision` | see `--help` | Inference precision for the diffusion / lifting models. |
| `--enable_image_guard` | off | Run the input image-safety guard. **`--image_dir` mode only** — skipped for `--data_root` runs, with a message on stdout. |
| `--image_guard_threshold` | see `--help` | Threshold used when `--enable_image_guard` is set. |
| `--offload_model_to_cpu` | off | Offload the diffusion models to CPU **while lifting runs**; lowers the lifting-stage peak, not diffusion's, so it does not by itself make a < 16 GB card sufficient. |

## `run.sh` — step-2 wrapper (diffusion + lifting)

| Flag | Default | Purpose |
|------|---------|---------|
| `--data-root` | `outputs/ncore_parser/` | Input directory containing `sample_paths.json`. |
| `--diffusion-ckpt` | `checkpoints/AH_multiview_diffusion.safetensors` | SparseViewDiT checkpoint. |
| `--lifting-ckpt` | `checkpoints/AH_tokengs_lifting.safetensors` | TokenGS checkpoint. |
| `--output-dir` | `outputs/` | Where per-sample outputs are written. |
| `--num-steps` | 30 | Diffusion inference steps. |
| `--cfg-scale` | 2.0 | Classifier-free guidance scale. |
| `--max-samples` | 0 | Maximum samples to process; `0` = all. |
| `--skip-lifting` | off | Disable TokenGS lifting — emit multiview outputs only. |
| `--offload` | off | Offload diffusion models to CPU while lifting runs. |

Per-sample outputs (`--image_dir` is flat; every `--data_root` run nests as
`<class_name>/<sample_id>/`;
`gaussians.ply` and `3d_lifted.mp4` are produced only when lifting runs):

```text
${OUTPUT_DIR}/<sample_id>/
├── multiview/
├── multiview.mp4
├── gaussians.ply
└── 3d_lifted.mp4
```

## `scripts/run_ncore_parser.sh` — step-1 wrapper

| Flag | Default | Purpose |
|------|---------|---------|
| `--component-store` | *(required)* | Clip `.json` manifest, comma-separated NCore V4 component-store paths, or `.zarr.itar` glob(s). |
| `--output-path` | `outputs/ncore_parser/` | Output directory for per-track object crops and `sample_paths.json`. |
| `--segmentation-ckpt` | `checkpoints/AH_object_seg_jit.pt` | Mask2Former JIT checkpoint for foreground masking. |
| `--camera-ids` | `camera_front_wide_120fov,camera_rear_right_70fov,camera_rear_left_70fov,camera_cross_left_120fov,camera_cross_right_120fov` | Comma-separated camera sensor IDs, passed through verbatim — use full sensor keys (e.g. `camera_front_wide_120fov,camera_cross_left_120fov`). |
| `--track-ids` | all tracks | Comma-separated track IDs to process (filter to specific objects). |

The flags above are the **wrapper**'s (`scripts/run_ncore_parser.sh`), which
supplies defaults such as `outputs/ncore_parser`.

The module can also be invoked directly via the installed console script
(`ncore-parser`) or `python -m asset_harvester.ncore_parser` — but the direct
CLI has **no output default** and requires its own mandatory flags (including
the output path and the segmentation checkpoint). Run it with `--help` rather
than copying the wrapper's flags.

## `asset_harvester.utils.image_segment` — standalone segmentation

Generate `mask.png` from `frame.jpeg` for each sub-folder using the
AV-object Mask2Former.

| Flag | Default | Purpose |
|------|---------|---------|
| `--checkpoint` | *(required)* | Path to `AH_object_seg_jit.pt`. |
| `--image_folder` | *(required)* | Directory of `<object_id>/` sub-folders. |
| `--frame_name` | `frame.jpeg` | Input filename to segment. |
| `--mask_name` | `mask.png` | Output mask filename. |

Run via:

```bash
python -m asset_harvester.utils.image_segment --help
```

## `asset_harvester.utils.orient_gaussians_for_nurec`

Rotate Gaussian PLY files into the orientation NuRec external-assets
insertion expects.

| Flag | Default | Purpose |
|------|---------|---------|
| `--input-dir` | *(required)* | Step-2 output directory (contains per-sample `gaussians.ply`). |
| `--output-dir` | — | If given, writes a transformed copy; otherwise requires `--in-place`. |
| `--in-place` | off | Overwrite `gaussians.ply` under `--input-dir`. |
| `--degrees` | 90 | Y-axis rotation applied (degrees). |

## `asset_harvester.utils.generate_external_assets_metadata`

Emit `metadata.yaml` describing each `gaussians.ply` for NuRec
insertion. `label_class` is read from each sample's **parent directory**,
so point this at nested `<class_name>/<sample_id>/` output from a
`--data_root` run. Flat `--image_dir` output records the output folder
name as the class unless you re-parent it first.

| Flag | Default | Purpose |
|------|---------|---------|
| `--input-dir` | *(required)* | Root of the (oriented) lifting output. |

## Installed console scripts (from `pyproject.toml`)

| Script | Target |
|--------|--------|
| `ncore-parser` | `asset_harvester.ncore_parser.__main__:main` |
| `tokengs-train` | `asset_harvester.tokengs.main:main` (see `docs/tokengs.md` in the upstream repo) |

## `benchmark/eval.py` — reconstruction metrics

| Flag | Default | Purpose |
|------|---------|---------|
| `--output_dir` | *(required)* | Root output directory from `run_inference.py` / `run.sh`. |
| `--eval_output_dir` | `<output_dir>/eval` | Where eval results are written. |
| `--output_size` | 512 | Render resolution for comparisons. |
| `--no_comparisons` | off | Skip saving side-by-side comparison images. |

Summary lands in `rendering_metrics_summary.txt`. Must be run inside
the `av-object-benchmark` conda env (clone of `asset-harvester` +
`bash benchmark/install.sh`).
