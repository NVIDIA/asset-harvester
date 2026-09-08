# Asset Harvester — Troubleshooting and teardown

Detail moved out of `SKILL.md`.

## Troubleshooting matrix

| Error | Cause | Solution |
|-------|-------|----------|
| `gsplat` import error or CUDA ABI mismatch | Installed `gsplat` from PyPI wheel instead of the pinned commit | Reinstall from source: `pip install --no-cache-dir --no-build-isolation "git+https://github.com/nerfstudio-project/gsplat.git@b60e917c95afc449c5be33a634f1f457e116ff5e"` then redo the editable install. |
| `nvcc` fails — "unsupported GNU version" | No candidate compiler passed `setup.sh`'s nvcc probe | `setup.sh` ignores your `CC`/`CXX`: it compiles a test `.cu` with `nvcc -ccbin` against `/usr/bin/gcc`, then the conda cross-compiler, takes the first that passes, and **overwrites** `CC`/`CXX`/`CUDAHOSTCXX`. Exporting them beforehand has no effect. Install a GCC that `nvcc` accepts (10–13) at `/usr/bin/gcc`, or let the conda compiler be selected. |
| `CUDA error: out of memory` **during lifting** | GPU VRAM < ~16 GB | Add `--offload_model_to_cpu` to `run_inference.py` or `--offload` to `run.sh`. |
| `401 Unauthorized` when Llama Guard loads | `--enable-image-guard` lazily pulls `meta-llama/Llama-Guard-3-11B-Vision` through transformers at inference time, not via `hf download` | Accept that model's licence on HuggingFace and authenticate before enabling the flag. |
| `401 Unauthorized` from `hf download` | A gated repo — the NCore dataset, DINOv3, or SAM 3D Body. `nvidia/asset-harvester` itself is public and downloads anonymously. | Accept that repo's licence on its HF page, then `hf auth login` with a token from https://huggingface.co/settings/tokens. |
| `sample_paths.json` not found (step 2) | Step 1 (NCore parsing) wasn't run or `--data-root` points elsewhere | Point `run.sh --data-root` at the `--output-path` of `run_ncore_parser.sh` (default `outputs/ncore_parser/`). |
| Over-saturated / wrong-color assets in NuRec | PPISP enabled during scene reconstruction | Disable PPISP for the NuRec scene reconstruction used with inserted assets. |
| Inserted asset is wrong size in NuRec | Wrong cuboid dimensions anywhere along the scale handoff | Scale is not predicted on the NCore path. Trace it: source-clip cuboid `dim` → `object_lwh` → `multiview/lwh.txt` → the `cuboids_dims` key in `metadata.yaml`, which is what NuRec reads. (`--image_dir` mode does predict dimensions into `multiview/lwh.txt`.) |
| `benchmark/eval.py` crashes on `transformers` import | Wrong env (main env pins 4.48.3; benchmark needs ≥ 4.56.0) | `conda activate av-object-benchmark`. Create it with the sequence in `end-to-end-ncore.md`: clone the env, **activate it**, and only then run `bash benchmark/install.sh`. The activation step is not optional — `install.sh` pip-installs `transformers>=4.56.0` into whatever env is active, so running it before activating upgrades the main `asset-harvester` env and breaks its pinned `transformers==4.48.3`. |
| SAM 3D Body checkpoint 403 | Gated repo access not granted | Request access at https://huggingface.co/facebook/sam-3d-body-dinov3; eval continues without embedding metrics. |

## Teardown

An Asset Harvester install leaves ~60 GB or more on the host. The steps
below remove what is exclusive to it; models it pulls into the **shared**
HuggingFace cache (C-RADIO, the DC-AE VAE, optionally Llama Guard) are
left in place unless you opt in at step 6, because other projects may use
them. Breakdown:
the four AH checkpoints (~12.8 GB), the source clone, two conda envs
(~10–15 GB each), benchmark checkpoints if installed, per-sample outputs
(multiview MP4s + Gaussian PLYs), and any downloaded NCore sample clips.

Every path below is anchored to `$AH_CHECKOUT`, the Asset Harvester clone,
because `checkpoints/`, `ncore-clips/` and `outputs/` are created inside it.
Resolve it to an **absolute** path and verify it before deleting anything —
the install steps leave you *inside* the checkout, so a relative
`./asset-harvester` will not resolve from there.

```bash
# 0. Resolve and sanity-check the checkout, then step outside it
AH_CHECKOUT=$(cd /path/to/asset-harvester && pwd)   # adjust to your clone
test -f "$AH_CHECKOUT/run_inference.py" || { echo "not an Asset Harvester checkout: $AH_CHECKOUT"; return 2>/dev/null || exit 1; }
cd "$AH_CHECKOUT/.."

# 1. Only if a container or another user left root-owned files behind, take
#    ownership of exactly those entries. Do not chown -R the whole tree: that
#    would also seize files belonging to collaborators.
if sudo find "$AH_CHECKOUT" -xdev -user root -print -quit | grep -q .; then
    sudo find "$AH_CHECKOUT" -xdev -user root -exec chown -h -- "$(id -u):$(id -g)" {} +
fi

# 2. Conda environments. OPT-IN: this list starts EMPTY on purpose. setup.sh
#    accepts --env-name and REUSES a pre-existing env rather than creating one,
#    so removing the defaults blindly can destroy an environment you already
#    had. Add only environments you know this install created.
AH_ENVS=()                             # e.g. AH_ENVS=(asset-harvester av-object-benchmark)
conda deactivate 2>/dev/null || true
for env in "${AH_ENVS[@]:-}"; do
    [ -n "$env" ] || continue
    conda env remove --name "$env" --yes 2>/dev/null || true
done

# 3. HuggingFace hub cache. This covers the common overrides; it is NOT the
#    full precedence chain — Transformers has its own cache/module overrides,
#    and custom paths using ~ or $VARS are not expanded here. If you set any
#    of those, check `hf cache scan` (or your own config) before assuming
#    this removed everything.
HF_CACHE="${HF_HUB_CACHE:-${HUGGINGFACE_HUB_CACHE:-${HF_HOME:+$HF_HOME/hub}}}"
HF_CACHE="${HF_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface/hub}"
rm -rf "$AH_CHECKOUT/checkpoints"
rm -rf "$HF_CACHE/models--nvidia--asset-harvester"

# 4. NCore sample clips downloaded for the demo (size depends on clips)
rm -rf "$AH_CHECKOUT/ncore-clips"
rm -rf "$HF_CACHE/datasets--nvidia--PhysicalAI-Autonomous-Vehicles-NCore"

# 5. Per-sample outputs (gaussians.ply, multiview.mp4, 3d_lifted.mp4, …)
rm -rf "$AH_CHECKOUT/outputs"

# 6. OPT-IN: models pulled into the SHARED HuggingFace cache by inference.
#    These are not exclusive to Asset Harvester — other projects may rely on
#    them — so removal is deliberate, not automatic. Uncomment what you want.
# rm -rf "$HF_CACHE/models--nvidia--C-RADIO"
# rm -rf "$HF_CACHE/models--mit-han-lab--dc-ae-f32c32-sana-1.0-diffusers"
# rm -rf "$HF_CACHE/models--meta-llama--Llama-Guard-3-11B-Vision"   # only if --enable_image_guard was used
# HF_MODULES="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/modules/transformers_modules/nvidia"
# rm -rf "$HF_MODULES/C-RADIO" "$HF_MODULES/C_hyphen_RADIO"

# 7. The clone itself, last (and the pinned-commit gsplat build cache it pulled)
rm -rf "$AH_CHECKOUT"

# 8. pip + conda caches (only if disk is tight; safe to keep)
pip cache purge 2>/dev/null || true
conda clean --all --yes 2>/dev/null || true
```

Verify:

```bash
for env in "${AH_ENVS[@]:-}"; do
    [ -n "$env" ] || continue
    conda env list | grep -qE "^${env}\\s" && echo "still present: $env"
done || true
test -e "$AH_CHECKOUT" && echo "still present: $AH_CHECKOUT" || echo "files: clean"
```

Do **not** revoke `HF_TOKEN` as part of teardown unless you suspect
it has been leaked (see `installation.md` "Verifying secrets
safely"); the token is per-user and shared across HuggingFace
workflows.
