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

"""
Model builder utilities for multiview diffusion inference.

Provides:
- VAE encode/decode functions (dc-ae, AutoencoderDC)
- C-RADIO image encoder loading
- Null image prompt generation
"""

import numpy as np
import torch
from diffusers import AutoencoderDC
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoModel, CLIPImageProcessor

from ..models.sparseviewdit import SparseViewDiTTransformer2DModelNative
from .convert_checkpoint import convert_sana_ms_to_diffusers


def get_c_radio(device="cuda"):
    """Load the C-RADIO image encoder and processor from HuggingFace."""
    from asset_harvester.patches.cradio_compat import patch_cradio_cache, patch_cradio_modules

    hf_repo = "nvidia/C-RADIO"
    patch_cradio_cache()
    image_processor = CLIPImageProcessor.from_pretrained(hf_repo)
    try:
        model = AutoModel.from_pretrained(hf_repo, trust_remote_code=True).to(device)
    except KeyError:
        patch_cradio_modules()
        model = AutoModel.from_pretrained(hf_repo, trust_remote_code=True).to(device)
    model.eval()
    return model, image_processor


def get_null_image_prompt(cradio_model, cradio_image_processor, device):
    """Generate null (grey) image prompt embeddings for classifier-free guidance."""
    imgs = [Image.fromarray(np.ones((512, 512, 3), dtype=np.uint8) * 128)]
    with torch.no_grad():
        pixel_values = cradio_image_processor(images=imgs, return_tensors="pt", do_resize=True).pixel_values
        pixel_values = pixel_values.to(device)
        summary, features = cradio_model(pixel_values)

    features = torch.amax(features, dim=0, keepdim=True)
    y = features.unsqueeze(1)
    y_mask = torch.ones((y.shape[0], 1, 1, y.shape[2]), dtype=torch.int64, device=y.device)
    return y[0], y_mask[0][0]


def vae_encode(name, vae, images, sample_posterior=False, device="cuda"):
    """Encode images through VAE to latent space."""
    if name == "sdxl" or name == "sd3":
        posterior = vae.encode(images.to(device)).latent_dist
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        z = (z - vae.config.shift_factor) * vae.config.scaling_factor
    elif "dc-ae" in name:
        ae = vae
        scaling_factor = ae.cfg.scaling_factor if ae.cfg.scaling_factor else 0.41407
        z = ae.encode(images.to(device))
        z = z * scaling_factor
    elif "AutoencoderDC" in name:
        ae = vae
        scaling_factor = ae.config.scaling_factor if ae.config.scaling_factor else 0.41407
        z = ae.encode(images.to(device))
        z = z * scaling_factor
    else:
        raise ValueError(f"Unknown VAE type: {name}")
    return z


def vae_decode(name, vae, latent):
    """Decode latent representations through VAE to images."""
    if name == "sdxl" or name == "sd3":
        latent = (latent.detach() / vae.config.scaling_factor) + vae.config.shift_factor
        samples = vae.decode(latent).sample
    elif "dc-ae" in name:
        ae = vae
        vae_scale_factor = (
            2 ** (len(ae.config.encoder_block_out_channels) - 1)
            if hasattr(ae, "config") and ae.config is not None
            else 32
        )
        scaling_factor = ae.cfg.scaling_factor if ae.cfg.scaling_factor else 0.41407
        if latent.shape[-1] * vae_scale_factor > 4000 or latent.shape[-2] * vae_scale_factor > 4000:
            try:
                from patch_conv import convert_model

                ae = convert_model(ae, splits=4)
            except ImportError:
                pass
        samples = ae.decode(latent.detach() / scaling_factor)
    elif "AutoencoderDC" in name:
        ae = vae
        scaling_factor = ae.config.scaling_factor if ae.config.scaling_factor else 0.41407
        try:
            samples = ae.decode(latent / scaling_factor, return_dict=False)[0]
        except torch.cuda.OutOfMemoryError:
            print("Warning: OOM during VAE decoding, retrying with tiled VAE decoding.")
            ae.enable_tiling(tile_sample_min_height=1024, tile_sample_min_width=1024)
            samples = ae.decode(latent / scaling_factor, return_dict=False)[0]
    else:
        raise ValueError(f"Unknown VAE type: {name}")
    return samples


_TRANSFORMER_CONFIG = dict(
    in_channels=32,
    out_channels=32,
    num_attention_heads=70,  # 70 * 32 = 2240
    attention_head_dim=32,
    num_layers=20,
    num_cross_attention_heads=20,
    cross_attention_head_dim=112,  # 20 * 112 = 2240
    cross_attention_dim=2240,
    caption_channels=1280,
    mlp_ratio=2.5,
    patch_size=1,
    sample_size=16,  # 16 * 32 = 512
    camera_emb=True,
    camera_emb_dim=17,
    brightness_emb=True,
    cond_on_rays=True,
    cond_on_mask=True,
)


def get_models(checkpoint_path, device, dtype):
    # 1. Load VAE
    print("\n Loading VAE...")
    print("   Loading standard VAE...")
    vae = AutoencoderDC.from_pretrained("mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers").to(device).to(dtype).eval()
    print("   VAE loaded")

    # 2. Load image encoder (c-radio)
    print("\n Loading image encoder (c-radio)...")
    cradio_model, cradio_image_processor = get_c_radio(device=device)
    print("   c-radio loaded")

    # 3. Load transformer
    print(f"\n Loading transformer from {checkpoint_path}...")
    transformer = SparseViewDiTTransformer2DModelNative(**_TRANSFORMER_CONFIG)
    state_dict = load_file(checkpoint_path, device="cpu")
    state_dict = convert_sana_ms_to_diffusers(state_dict, hidden_size=2240)
    missing, unexpected = transformer.load_state_dict(state_dict, strict=False)
    print(f"   Checkpoint converted and loaded (missing: {len(missing)}, unexpected: {len(unexpected)})")

    transformer = transformer.to(device).to(dtype)
    transformer.eval()
    return vae, cradio_model, cradio_image_processor, transformer
