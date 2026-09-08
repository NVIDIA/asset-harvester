# Asset Harvester — Installation, checkpoints, and secret handling

Reference detail moved out of `SKILL.md`. The headline steps live in
the parent skill's `## Instructions`; everything below is the long
form.

## One-shot install (recommended, ~20 min)

```bash
git clone https://github.com/NVIDIA/asset-harvester.git
cd asset-harvester
bash setup.sh                 # creates the `asset-harvester` conda env
conda activate asset-harvester
```

Optional flags: `bash setup.sh --env-name asset-harvester --python 3.10`.

`setup.sh` handles: git submodules, conda env creation,
`cuda-toolkit=12.8` install, nvcc host-compiler probing, PyTorch
2.10.0 CUDA wheels, a **source build of `gsplat` at the pinned
commit `b60e917c95afc449c5be33a634f1f457e116ff5e`**, editable
install of `asset-harvester` with all extras, and `ruff`.

## Manual install (when `setup.sh` is not usable)

Preinstall `gsplat` at the pinned commit **before** the editable
install — otherwise pip will resolve a wheel with the wrong CUDA ABI:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
    torch==2.10.0 torchvision
pip install --no-cache-dir --no-build-isolation \
    "git+https://github.com/nerfstudio-project/gsplat.git@b60e917c95afc449c5be33a634f1f457e116ff5e"
pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
    -e ".[ncore-parser,multiview_diffusion,tokengs,camera-estimator]"
```

Sanity check the CUDA extension:

```bash
python -c "from gsplat.cuda._backend import _C; print('gsplat CUDA ready')"
```

## Checkpoints

The `nvidia/asset-harvester` repo is public — this downloads anonymously,
no login required:

```bash
pip install "huggingface_hub[cli]"
hf download nvidia/asset-harvester --local-dir checkpoints
```

Authenticate only for gated resources — the
`PhysicalAI-Autonomous-Vehicles-NCore` dataset, DINOv3, Llama Guard and
SAM 3D Body — each of which also needs its licence accepted on its own
HuggingFace page first:

```bash
hf auth login                 # paste token from https://huggingface.co/settings/tokens
```

Result:

```text
checkpoints/
├── AH_multiview_diffusion.safetensors
├── AH_tokengs_lifting.safetensors
├── AH_camera_estimator.safetensors
└── AH_object_seg_jit.pt
```

## Verifying secrets safely

**Always verify prerequisites by running
[`scripts/validate_setup.py`](../scripts/validate_setup.py); never by
writing ad-hoc bash that interpolates `HF_TOKEN` values.** The common
one-liner

```bash
# BAD — leaks the secret to the terminal when the variable is set
echo "HF_TOKEN: ${HF_TOKEN:+yes}${HF_TOKEN:-no}"
```

prints `yes<token-value>` whenever `HF_TOKEN` is set, because
`${VAR:-no}` only falls back to "no" when `VAR` is empty — when set
it expands to `$VAR`. Use a length-only check, which never echoes
the value:

```bash
# OK — prints "set (N chars)" or "missing", never the value
test -n "$HF_TOKEN" && echo "HF_TOKEN: set (${#HF_TOKEN} chars)" || echo "HF_TOKEN: missing"
```

Rotate any token you suspect was echoed at
<https://huggingface.co/settings/tokens>.
