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

"""Patch nvidia/C-RADIO for transformers >=4.48 compatibility.

transformers 4.48+ loads weights per-parameter with assign=True, passing a
single-key state_dict (e.g. {"weight": tensor}) to _load_from_state_dict.
C-RADIO's custom overrides expect the full prefixed key (e.g. "embedder.weight"),
causing a KeyError.

Two patch strategies:
  1. patch_cradio_cache() — patches vit_patch_generator.py on disk before import.
  2. patch_cradio_modules() — patches already-loaded classes in sys.modules
     (for the case where cache was empty and from_pretrained downloaded + imported
     the code before we could patch the file).

Usage:
    from asset_harvester.patches.cradio_compat import patch_cradio_cache, patch_cradio_modules

    patch_cradio_cache()  # patch files on disk (no-op if cache empty)
    try:
        model = AutoModel.from_pretrained("nvidia/C-RADIO", trust_remote_code=True)
    except KeyError:
        patch_cradio_modules()  # patch in-memory classes after first download
        model = AutoModel.from_pretrained("nvidia/C-RADIO", trust_remote_code=True)
"""

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Sentinel comment added to patched files so we can detect prior application
_PATCH_SENTINEL = "# PATCHED by asset_harvester/patches/cradio_compat.py for transformers >=4.48 compat"

# Original code snippets to replace (on-disk patching)
_GENERATOR_ORIGINAL = """\
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if self.abs_pos:
            self._load_embed(state_dict[f'{prefix}pos_embed'], self.pos_embed)"""

_GENERATOR_PATCHED = f"""\
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        {_PATCH_SENTINEL}
        if self.abs_pos:
            _key = f'{{prefix}}pos_embed'
            _val = state_dict[_key] if _key in state_dict else state_dict.get('pos_embed')
            if _val is not None:
                self._load_embed(_val, self.pos_embed)"""

_LINEAR_ORIGINAL = """\
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if self.bias is not None:
            self.bias.data.copy_(state_dict[f'{prefix}bias'])

        chk_weight = state_dict[f'{prefix}weight']"""

_LINEAR_PATCHED = f"""\
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        {_PATCH_SENTINEL}
        if self.bias is not None:
            _bkey = f'{{prefix}}bias'
            self.bias.data.copy_(state_dict[_bkey] if _bkey in state_dict else state_dict['bias'])

        _wkey = f'{{prefix}}weight'
        chk_weight = state_dict[_wkey] if _wkey in state_dict else state_dict['weight']"""


# --- In-memory patch functions (for classes already loaded into sys.modules) ---


def _get_key(state_dict, prefixed_key, bare_key):
    """Look up prefixed key first, fall back to bare key for per-parameter loading."""
    if prefixed_key in state_dict:
        return state_dict[prefixed_key]
    if bare_key in state_dict:
        return state_dict[bare_key]
    raise KeyError(f"Neither '{prefixed_key}' nor '{bare_key}' found in state_dict (keys: {list(state_dict.keys())})")


def _patched_generator_load(
    self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
):
    if self.abs_pos:
        self._load_embed(_get_key(state_dict, f"{prefix}pos_embed", "pos_embed"), self.pos_embed)


def _patched_linear_load(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    import math

    import torch.nn.functional as F
    from einops import rearrange

    if self.bias is not None:
        self.bias.data.copy_(_get_key(state_dict, f"{prefix}bias", "bias"))

    chk_weight = _get_key(state_dict, f"{prefix}weight", "weight")
    if chk_weight.shape != self.weight.shape:
        src_patch_size = int(math.sqrt(chk_weight.shape[1] // 3))
        assert (src_patch_size**2) * 3 == chk_weight.shape[1], "Unable to interpolate non-square patch size"
        chk_weight = rearrange(chk_weight, "b (c h w) -> b c h w", c=3, h=src_patch_size, w=src_patch_size)
        chk_weight = F.interpolate(
            chk_weight,
            size=(self.patch_size, self.patch_size),
            mode="bicubic",
            align_corners=True,
            antialias=False,
        )
        chk_weight = rearrange(chk_weight, "b c h w -> b (c h w)")
    self.weight.data.copy_(chk_weight)


# --- On-disk patch ---


def _find_cached_files():
    """Find all vit_patch_generator.py files in the HF modules cache."""
    base = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules" / "nvidia"
    results = []
    for dirname in ("C-RADIO", "C_hyphen_RADIO"):
        hf_cache = base / dirname
        if hf_cache.is_dir():
            results.extend(hf_cache.rglob("vit_patch_generator.py"))
    return results


def patch_cradio_cache():
    """Patch C-RADIO's vit_patch_generator.py on disk in the HF modules cache.

    Idempotent — skips files already patched. Call before from_pretrained.
    Returns True if at least one file was patched.
    """
    files = _find_cached_files()
    if not files:
        logger.info(
            "No C-RADIO vit_patch_generator.py found in HF cache. Will patch in-memory after download if needed."
        )
        return False

    any_patched = False
    for fpath in files:
        content = fpath.read_text()

        if _PATCH_SENTINEL in content:
            logger.info("C-RADIO patch already applied to %s", fpath)
            continue

        patched = False
        if _GENERATOR_ORIGINAL in content:
            content = content.replace(_GENERATOR_ORIGINAL, _GENERATOR_PATCHED)
            patched = True
        if _LINEAR_ORIGINAL in content:
            content = content.replace(_LINEAR_ORIGINAL, _LINEAR_PATCHED)
            patched = True

        if patched:
            fpath.write_text(content)
            logger.info("C-RADIO compat patch applied to %s", fpath)
            any_patched = True
        else:
            logger.warning(
                "C-RADIO vit_patch_generator.py at %s did not match expected code. Manual review may be needed.",
                fpath,
            )
    return any_patched


def patch_cradio_modules():
    """Patch C-RADIO classes already loaded in sys.modules.

    Call this after a failed from_pretrained when the cache was empty on the
    first call. Also patches on disk and flushes __pycache__ so that the retry
    re-imports the patched code.
    """
    # Patch on disk first (for the retry import)
    patch_cradio_cache()

    # Patch in-memory classes
    patched = []
    for mod_name, mod in list(sys.modules.items()):
        if "RADIO" not in mod_name and "radio" not in mod_name:
            continue
        if not hasattr(mod, "__file__"):
            continue

        if hasattr(mod, "ViTPatchGenerator"):
            cls = mod.ViTPatchGenerator
            if not getattr(cls, "_cradio_patched", False):
                cls._load_from_state_dict = _patched_generator_load
                cls._cradio_patched = True
                patched.append("ViTPatchGenerator")

        if hasattr(mod, "ViTPatchLinear"):
            cls = mod.ViTPatchLinear
            if not getattr(cls, "_cradio_patched", False):
                cls._load_from_state_dict = _patched_linear_load
                cls._cradio_patched = True
                patched.append("ViTPatchLinear")

    if patched:
        logger.info("C-RADIO in-memory patch applied to: %s", ", ".join(patched))
    else:
        logger.warning("No C-RADIO classes found in sys.modules to patch")
