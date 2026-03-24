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
Native diffusers implementation of SparseViewDiTTransformer2DModel.

This model inherits from SanaTransformer2DModel and adds multiview-specific
conditioning (camera, brightness, rays, mask, x_seq_len). It uses standard
diffusers building blocks (Attention, SanaTransformerBlock, etc.) so that
all diffusers features (JVP, LoRA, attention processors, etc.) work out of
the box.

Architecture mapping (original training format -> this):
  blocks           -> transformer_blocks  (SanaTransformerBlock)
  x_embedder       -> patch_embed         (PatchEmbed)
  t_embedder       -> time_embed          (AdaLayerNormSingle / SanaCombinedTimestepGuidanceEmbeddings)
  t_block          -> (part of time_embed)
  y_embedder       -> caption_projection + caption_norm
  final_layer      -> norm_out + proj_out + scale_shift_table
  cam_emb/block    -> cam_emb + cam_emb_block      (new)
  aug_emb/block    -> aug_emb + aug_emb_block      (new)
  brightness_emb   -> brightness_emb + brightness_emb_block  (new)
  skeletal_emb     -> skeletal_emb + skeletal_emb_block      (new)
"""

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention_processor import Attention
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.sana_transformer import SanaTransformer2DModel, SanaTransformerBlock
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers

logger = logging.get_logger(__name__)


# ---------------------------------------------------------------------------
# Linear attention processor matching original LiteLA numerics exactly
# ---------------------------------------------------------------------------
def _unaggregate(x: torch.Tensor, seq_len: list[int]):
    """Pad variable-length sequences into a fixed-size batch.

    Args:
        x: (BV, N, C) — packed tokens from all views of all batch items
        seq_len: list of ints, number of views per batch item (sums to BV)
    Returns:
        x_new: (B, max_views*N, C) — padded batch
        mask: (B, max_views*N, 1) — 1 for real tokens, 0 for padding
    """
    import einops

    x_new = []
    mask_list = []
    _, N, C = x.shape
    max_n = max(seq_len) * N
    start = 0
    for n in seq_len:
        x_v = x[start : start + n]
        x_v = einops.rearrange(x_v, "V N C -> 1 (V N) C")
        pad_len = max_n - N * n
        x_new.append(torch.cat([x_v, x.new_zeros((1, pad_len, C))], dim=1))
        # Create mask with same dtype as x for compatibility with JVP and mixed precision
        mask_list.append(
            torch.cat(
                [
                    torch.ones((1, n * N, 1), device=x.device),
                    torch.zeros((1, pad_len, 1), device=x.device),
                ],
                dim=1,
            ).to(x.dtype)
        )
        start += n
    return torch.cat(x_new, dim=0), torch.cat(mask_list, dim=0)


def _aggregate(x: torch.Tensor, mask: torch.Tensor, seq_len: list[int]):
    """Unpad back to the original packed format.

    Args:
        x: (B, max_views*N, C) — padded batch
        mask: (B, max_views*N, 1) — mask (may be a dual tensor inside JVP)
        seq_len: list of ints — MUST be a plain Python list (not tensors)
    Returns:
        x: (BV, N, C)
    """
    import einops

    B, max_n, C = x.shape
    # Compute N (tokens per view) from the known seq_len and max_n
    max_views = max(seq_len)
    N = max_n // max_views

    # Instead of using mask.sum().tolist() (fails in JVP), compute token counts
    # directly from seq_len which is a plain Python list
    new_x = []
    for i, n in enumerate(seq_len):
        n_tokens = n * N
        # Slice the valid tokens (first n_tokens out of max_n)
        new_x.append(einops.rearrange(x[i, :n_tokens], "(V N) C -> V N C", V=n))
    return torch.cat(new_x, dim=0)


class LiteLinearAttnProcessor:
    """
    Linear attention processor that matches the original LiteLA implementation
    numerically, including ``x_seq_len`` support for multiview batching.

    When ``x_seq_len`` is provided (a list of ints giving the number of views
    per batch item), the processor:
    1. Unpacks the (BV, N, C) tensor into a padded (B, max_V*N, C) batch
    2. Applies a mask so that padding tokens don't contribute to attention
    3. Re-packs the output back to (BV, N, C)

    This exactly replicates the ``unaggregate`` / ``attn_matmul`` / ``aggregate``
    flow in the original ``LiteLA.forward()``.
    """

    PAD_VAL = 1.0
    EPS = 1e-15

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        # multiview extras (forwarded from SparseViewDiTTransformerBlock)
        x_seq_len: list[int] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        C = hidden_states.shape[-1]
        heads = attn.heads
        head_dim = C // heads

        # --- QKV projection ---
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Fuse QKV for unaggregate (same as original: qkv then unaggregate)
        qkv = torch.cat([query, key, value], dim=-1)  # (BV, N, 3*C)

        # --- Unaggregate for x_seq_len ---
        mask = None
        if x_seq_len is not None:
            qkv, mask = _unaggregate(qkv, x_seq_len)

        B, N_padded, _ = qkv.shape
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, N_padded, C)

        # Reshape to LiteLA layout: (B, heads, head_dim, N)
        q = q.transpose(1, 2).reshape(B, heads, head_dim, N_padded)
        k = k.transpose(1, 2).reshape(B, heads, head_dim, N_padded).transpose(-1, -2)  # (B,h,N,d)
        v = v.transpose(1, 2).reshape(B, heads, head_dim, N_padded)

        # ReLU kernel
        q = F.relu(q)
        k = F.relu(k)

        # Note: We do NOT cast to float32 here (unlike the original LiteLA)
        # because torch.func.jvp requires consistent dtypes throughout.
        # The float32 cast in LiteLA was for numerical stability but the
        # JVP tangent propagation breaks if we change dtype mid-computation.
        # For training, the model runs in bf16/fp16 which is sufficient.

        # Pad value channel
        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=self.PAD_VAL)  # (B, h, d+1, N)

        # Apply mask (zero out padding tokens)
        if mask is not None:
            # mask: (B, N_padded, 1) -> broadcast to (B, 1, 1, N_padded)
            m = mask.view(B, 1, 1, N_padded)
            v = v * m
            k = k * mask.view(B, 1, N_padded, 1)
            q = q * m

        # Linear attention matmul
        vk = torch.matmul(v, k)  # (B, h, d+1, d)
        out = torch.matmul(vk, q)  # (B, h, d+1, N)

        out = out[:, :, :-1] / (out[:, :, -1:] + self.EPS)

        # Reshape back: (B, h, d, N) -> (B, C, N) -> (B, N, C)
        out = out.reshape(B, C, N_padded).permute(0, 2, 1)

        # --- Aggregate back ---
        if x_seq_len is not None:
            out = _aggregate(out, mask, x_seq_len)

        # Output projection
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)

        return out


# ---------------------------------------------------------------------------
# Camera / auxiliary embedder (matches original BasicCameraEmbedder)
# ---------------------------------------------------------------------------
class BasicCameraEmbedder(nn.Module):
    """MLP embedder for camera matrices, augmentation sigmas, brightness, or skeleton poses."""

    def __init__(self, hidden_size: int, camera_emb_size: int = 17):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(camera_emb_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(t)


# ---------------------------------------------------------------------------
# Custom cross-attention processor that handles x_seq_len via
# xformers BlockDiagonalMask (same semantics as original MultiHeadCrossAttention)
# ---------------------------------------------------------------------------
class SparseViewDiTCrossAttnProcessor:
    """
    Cross-attention processor that supports variable-length sequences via
    ``x_seq_len`` / ``y_lens`` for multiview batching.

    When ``x_seq_len`` is provided, a block-diagonal mask is built so each
    batch item only attends to its own key/value tokens.

    Args:
        use_manual_sdpa: If True, use a manual scaled dot-product attention
            implementation that is compatible with ``torch.func.jvp`` (required
            for distillation training). If False (default), use
            ``xformers.memory_efficient_attention`` for best inference speed.
    """

    def __init__(self, use_manual_sdpa: bool = False):
        self.use_manual_sdpa = use_manual_sdpa

    @staticmethod
    def _manual_sdpa(query, key, value, attn_mask=None):
        """Manual scaled dot-product attention (JVP-compatible)."""
        scale_factor = 1.0 / math.sqrt(query.size(-1))
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        if attn_mask is not None:
            attn_weight = attn_weight + attn_mask
        attn_weight = torch.softmax(attn_weight, dim=-1)
        return attn_weight @ value

    def _block_diag_manual(self, q, k, v, token_seq_len, y_lens, heads, head_dim, inner_dim):
        """Block-diagonal attention using manual SDPA (JVP-safe)."""
        total_q = sum(token_seq_len)
        total_kv = sum(y_lens)

        q = q.reshape(1, total_q, heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(1, total_kv, heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(1, total_kv, heads, head_dim).permute(0, 2, 1, 3)

        # Build block-diagonal mask
        mask = torch.full(
            (total_q, total_kv),
            float("-inf"),
            device=q.device,
            dtype=q.dtype,
        )
        q_off, kv_off = 0, 0
        for ql, kvl in zip(token_seq_len, y_lens):
            mask[q_off : q_off + ql, kv_off : kv_off + kvl] = 0.0
            q_off += ql
            kv_off += kvl
        mask = mask.unsqueeze(0).unsqueeze(0)

        out = self._manual_sdpa(q, k, v, attn_mask=mask)
        return out.permute(0, 2, 1, 3).reshape(1, total_q, inner_dim)

    def _block_diag_xformers(self, q, k, v, token_seq_len, y_lens, heads, head_dim, inner_dim):
        """Block-diagonal attention using xformers (fast, not JVP-safe)."""
        total_q = sum(token_seq_len)
        total_kv = sum(y_lens)

        q = q.reshape(1, total_q, heads, head_dim)
        k = k.reshape(1, total_kv, heads, head_dim)
        v = v.reshape(1, total_kv, heads, head_dim)

        import xformers.ops

        attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens(token_seq_len, y_lens)
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        return out.reshape(1, total_q, inner_dim)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        x_seq_len: list[int] | None = None,
        y_lens: list[int] | None = None,
    ) -> torch.Tensor:
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        batch_size, seq_len_q, _ = hidden_states.shape
        _, seq_len_kv, _ = encoder_hidden_states.shape

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        if x_seq_len is not None and y_lens is not None:
            N = seq_len_q
            token_seq_len = [N * xsl for xsl in x_seq_len]

            if self.use_manual_sdpa:
                out = self._block_diag_manual(
                    query,
                    key,
                    value,
                    token_seq_len,
                    y_lens,
                    attn.heads,
                    head_dim,
                    inner_dim,
                )
            else:
                out = self._block_diag_xformers(
                    query,
                    key,
                    value,
                    token_seq_len,
                    y_lens,
                    attn.heads,
                    head_dim,
                    inner_dim,
                )
            hidden_states = out.reshape(batch_size, seq_len_q, inner_dim)
        else:
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attention_mask is not None:
                attention_mask = attn.prepare_attention_mask(attention_mask, seq_len_kv, batch_size)
                attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

            if self.use_manual_sdpa:
                hidden_states = self._manual_sdpa(query, key, value, attn_mask=attention_mask)
            else:
                hidden_states = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=False,
                )
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, inner_dim)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


# ---------------------------------------------------------------------------
# Multiview Transformer Block (thin wrapper adding x_seq_len forwarding)
# ---------------------------------------------------------------------------
class SparseViewDiTTransformerBlock(SanaTransformerBlock):
    """SanaTransformerBlock extended with ``x_seq_len`` / ``y_lens`` support.

    - Self-attention (attn1) uses ``LiteLinearAttnProcessor`` for exact numerical
      match with the original LiteLA implementation.
    - Cross-attention (attn2) uses ``SparseViewDiTCrossAttnProcessor`` so
      that variable-length multiview sequences are handled correctly.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace self-attention processor with LiteLA-compatible one
        self.attn1.set_processor(LiteLinearAttnProcessor())
        # Replace the cross-attention processor with our multiview-aware one
        if hasattr(self, "attn2") and self.attn2 is not None:
            self.attn2.set_processor(SparseViewDiTCrossAttnProcessor())

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        timestep: torch.LongTensor | None = None,
        height: int = None,
        width: int = None,
        # Multiview extras
        x_seq_len: list[int] | None = None,
        y_lens: list[int] | None = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        # 1. Modulation (same as parent)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
        ).chunk(6, dim=1)

        # 2. Self Attention — pass x_seq_len for multiview masking
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        norm_hidden_states = norm_hidden_states.to(hidden_states.dtype)

        attn_output = self.attn1(norm_hidden_states, x_seq_len=x_seq_len)
        hidden_states = hidden_states + gate_msa * attn_output

        # 3. Cross Attention — pass x_seq_len / y_lens through attention_kwargs
        if self.attn2 is not None:
            attn_output = self.attn2(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                x_seq_len=x_seq_len,
                y_lens=y_lens,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        # Cast to weight dtype for Conv2d compatibility (JVP may produce float32)
        ff_dtype = self.ff.conv_inverted.weight.dtype
        norm_hidden_states = norm_hidden_states.unflatten(1, (height, width)).permute(0, 3, 1, 2)
        norm_hidden_states = norm_hidden_states.to(ff_dtype)
        ff_output = self.ff(norm_hidden_states)
        ff_output = ff_output.flatten(2, 3).permute(0, 2, 1)
        hidden_states = hidden_states + gate_mlp * ff_output

        return hidden_states


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class SparseViewDiTTransformer2DModelNative(SanaTransformer2DModel):
    """
    Native diffusers SanaTransformer2DModel extended with multiview conditioning.

    Extra config params (on top of base SanaTransformer2DModel):
        camera_emb, camera_emb_dim, aug_emb, aug_emb_dim,
        brightness_emb, skeletal_emb, skeletal_emb_dim,
        cond_on_rays, cond_on_mask
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["SparseViewDiTTransformerBlock", "PatchEmbed"]

    @register_to_config
    def __init__(
        self,
        # --- base SparseViewDiTTransformer2DModel params ---
        in_channels: int = 32,
        out_channels: int | None = 32,
        num_attention_heads: int = 70,
        attention_head_dim: int = 32,
        num_layers: int = 20,
        num_cross_attention_heads: int | None = 20,
        cross_attention_head_dim: int | None = 112,
        cross_attention_dim: int | None = 2240,
        caption_channels: int = 1280,
        mlp_ratio: float = 2.5,
        dropout: float = 0.0,
        attention_bias: bool = False,
        sample_size: int = 32,
        patch_size: int = 1,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        interpolation_scale: int | None = None,
        guidance_embeds: bool = False,
        guidance_embeds_scale: float = 0.1,
        qk_norm: str | None = None,
        timestep_scale: float = 1.0,
        # --- multiview params ---
        camera_emb: bool = False,
        camera_emb_dim: int = 17,
        aug_emb: bool = False,
        aug_emb_dim: int = 1,
        brightness_emb: bool = False,
        skeletal_emb: bool = False,
        skeletal_emb_dim: int = 63,
        cond_on_rays: bool = False,
        cond_on_mask: bool = False,
    ):
        # Adjust in_channels for rays / mask before calling super
        actual_in_channels = in_channels
        if cond_on_rays:
            actual_in_channels += 6
        if cond_on_mask:
            actual_in_channels += 1

        super().__init__(
            in_channels=actual_in_channels,
            out_channels=out_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            num_cross_attention_heads=num_cross_attention_heads,
            cross_attention_head_dim=cross_attention_head_dim,
            cross_attention_dim=cross_attention_dim,
            caption_channels=caption_channels,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_bias=attention_bias,
            sample_size=sample_size,
            patch_size=patch_size,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            interpolation_scale=interpolation_scale,
            guidance_embeds=guidance_embeds,
            guidance_embeds_scale=guidance_embeds_scale,
            qk_norm=qk_norm,
            timestep_scale=timestep_scale,
        )

        inner_dim = num_attention_heads * attention_head_dim

        # Replace transformer_blocks with multiview-aware blocks
        self.transformer_blocks = nn.ModuleList(
            [
                SparseViewDiTTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    num_cross_attention_heads=num_cross_attention_heads,
                    cross_attention_head_dim=cross_attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    mlp_ratio=mlp_ratio,
                    qk_norm=qk_norm,
                )
                for _ in range(num_layers)
            ]
        )

        # --- Multiview conditioning embedders ---
        if camera_emb:
            self.cam_emb = BasicCameraEmbedder(inner_dim, camera_emb_dim)
            self.cam_emb_block = nn.Sequential(nn.SiLU(), nn.Linear(inner_dim, 6 * inner_dim, bias=True))

        if aug_emb:
            self.aug_emb_module = BasicCameraEmbedder(inner_dim, aug_emb_dim)
            self.aug_emb_block = nn.Sequential(nn.SiLU(), nn.Linear(inner_dim, 6 * inner_dim, bias=True))

        if brightness_emb:
            self.brightness_emb_module = BasicCameraEmbedder(inner_dim, camera_emb_size=1)
            self.brightness_emb_block = nn.Sequential(nn.SiLU(), nn.Linear(inner_dim, 6 * inner_dim, bias=True))

        if skeletal_emb:
            self.skeletal_emb_module = BasicCameraEmbedder(inner_dim, skeletal_emb_dim)
            self.skeletal_emb_block = nn.Sequential(nn.SiLU(), nn.Linear(inner_dim, 6 * inner_dim, bias=True))

        # Convenience aliases used by distillation / inference
        self.depth = num_layers
        self.hidden_size = inner_dim

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        controlnet_block_samples: tuple[torch.Tensor] | None = None,
        return_dict: bool = True,
        # --- multiview extras ---
        camera_emb: torch.Tensor | None = None,
        aug_sigmas: torch.Tensor | None = None,
        relative_brightness: torch.Tensor | None = None,
        skeletal_emb: torch.Tensor | None = None,
        rays: torch.Tensor | None = None,
        cond_mask: torch.Tensor | None = None,
        clean_images: torch.Tensor | None = None,
        x_seq_len: list[int] | None = None,
    ) -> tuple[torch.Tensor, ...] | Transformer2DModelOutput:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        # --- 0. Multiview input conditioning ---
        if cond_mask is not None and clean_images is not None:
            hidden_states = cond_mask * clean_images + (1 - cond_mask) * hidden_states

        if self.config.cond_on_rays and rays is not None:
            hidden_states = torch.cat([hidden_states, rays], dim=1)
        if self.config.cond_on_mask and cond_mask is not None:
            hidden_states = torch.cat([hidden_states, cond_mask[:, :1, :, :]], dim=1)

        # --- 1. Patch embed ---
        batch_size, num_channels, height, width = hidden_states.shape
        p = self.config.patch_size
        post_patch_height, post_patch_width = height // p, width // p

        hidden_states = self.patch_embed(hidden_states)

        # --- 2. Time embedding ---
        if self.config.guidance_embeds:
            # Model has SanaCombinedTimestepGuidanceEmbeddings — requires guidance=
            if guidance is None:
                guidance = torch.zeros(batch_size, device=hidden_states.device, dtype=hidden_states.dtype)
            timestep_out, embedded_timestep = self.time_embed(
                timestep,
                guidance=guidance,
                hidden_dtype=hidden_states.dtype,
            )
        else:
            # Model has AdaLayerNormSingle — requires batch_size=
            timestep_out, embedded_timestep = self.time_embed(
                timestep,
                batch_size=batch_size,
                hidden_dtype=hidden_states.dtype,
            )

        # --- 3. Add multiview conditioning to timestep embedding ---
        # timestep_out is (B, 6*D) — the adaLN conditioning signal
        if self.config.camera_emb and camera_emb is not None:
            c_emb = self.cam_emb(camera_emb)
            timestep_out = timestep_out + self.cam_emb_block(c_emb)

        if self.config.aug_emb and aug_sigmas is not None:
            a_emb = self.aug_emb_module(aug_sigmas)
            timestep_out = timestep_out + self.aug_emb_block(a_emb)

        if self.config.brightness_emb and relative_brightness is not None:
            b_emb = self.brightness_emb_module(relative_brightness)
            timestep_out = timestep_out + self.brightness_emb_block(b_emb)

        if self.config.skeletal_emb and skeletal_emb is not None:
            s_emb = self.skeletal_emb_module(skeletal_emb)
            timestep_out = timestep_out + self.skeletal_emb_block(s_emb)

        # --- 4. Caption projection ---
        # encoder_hidden_states may come in different forms:
        # (a) Raw captions: (B_prompt, seq_len, caption_channels) -> needs projection
        # (b) Already packed by pipeline: (1, total_tokens, inner_dim) -> skip projection
        inner_dim = hidden_states.shape[-1]

        if encoder_hidden_states.shape[-1] != inner_dim:
            # Needs projection (raw c-radio or text embeddings)
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)

        # Reshape to (B_prompt, seq_len, inner_dim)
        if encoder_hidden_states.ndim == 2:
            encoder_hidden_states = encoder_hidden_states.unsqueeze(0)

        # If encoder_hidden_states batch doesn't match hidden_states batch,
        # it's because it's per-prompt (not per-view). We need to handle this
        # by expanding or keeping as-is for the packing step below.
        enc_batch = encoder_hidden_states.shape[0]

        encoder_hidden_states = self.caption_norm(encoder_hidden_states)

        # Prepare y_lens for multiview cross-attention
        # The original code packs all captions into (1, total_tokens, D)
        # and tracks per-item lengths via y_lens.
        y_lens: list[int] | None = None

        # If x_seq_len is provided but no mask, always construct y_lens from encoder_hidden_states shape
        if x_seq_len is not None and encoder_attention_mask is None:
            seq_len = encoder_hidden_states.shape[1]
            y_lens = [seq_len] * enc_batch
            encoder_hidden_states = encoder_hidden_states.reshape(1, -1, inner_dim)
        elif encoder_attention_mask is not None and encoder_attention_mask.ndim >= 2:
            # encoder_attention_mask: (B, 1, 1, S) or (B, S) — 1=keep, 0=discard
            mask_2d = encoder_attention_mask.squeeze()
            if mask_2d.ndim == 1:
                mask_2d = mask_2d.unsqueeze(0)
            # Expand mask to match encoder_hidden_states batch if needed
            if mask_2d.shape[0] != enc_batch:
                mask_2d = mask_2d[:enc_batch]
            # Compute y_lens — use try/except for JVP compatibility
            # (.tolist() fails inside torch.func.jvp because tensors are dual)
            try:
                y_lens = mask_2d.sum(dim=1).int().tolist()
            except RuntimeError:
                # Inside JVP: assume all tokens are valid (mask is all 1s)
                seq_len_val = encoder_hidden_states.shape[1]
                y_lens = [seq_len_val] * enc_batch
            # Pack into (1, total_tokens, D)
            try:
                encoder_hidden_states = encoder_hidden_states.reshape(-1, inner_dim)
                keep = mask_2d.reshape(-1).bool()
                encoder_hidden_states = encoder_hidden_states[keep].unsqueeze(0)
            except RuntimeError:
                # Inside JVP: just reshape without masking
                encoder_hidden_states = encoder_hidden_states.reshape(1, -1, inner_dim)
            # Clear the mask since we handle it via y_lens
            encoder_attention_mask = None
        elif enc_batch != batch_size:
            # No mask but batch mismatch: pack all captions into single sequence
            seq_len = encoder_hidden_states.shape[1]
            y_lens = [seq_len] * enc_batch
            encoder_hidden_states = encoder_hidden_states.reshape(1, -1, inner_dim)

        # Convert standard attention_mask
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # --- 5. Transformer blocks ---
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for index_block, block in enumerate(self.transformer_blocks):
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep_out,
                    post_patch_height,
                    post_patch_width,
                    x_seq_len,
                    y_lens,
                )
                if controlnet_block_samples is not None and 0 < index_block <= len(controlnet_block_samples):
                    hidden_states = hidden_states + controlnet_block_samples[index_block - 1]
        else:
            for index_block, block in enumerate(self.transformer_blocks):
                hidden_states = block(
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep_out,
                    post_patch_height,
                    post_patch_width,
                    x_seq_len=x_seq_len,
                    y_lens=y_lens,
                )
                if controlnet_block_samples is not None and 0 < index_block <= len(controlnet_block_samples):
                    hidden_states = hidden_states + controlnet_block_samples[index_block - 1]

        # --- 6. Output norm + projection ---
        hidden_states = self.norm_out(hidden_states, embedded_timestep, self.scale_shift_table)
        hidden_states = self.proj_out(hidden_states)

        # --- 7. Unpatchify ---
        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_height,
            post_patch_width,
            p,
            p,
            -1,
        )
        hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)
        output = hidden_states.reshape(batch_size, -1, post_patch_height * p, post_patch_width * p)

        # --- 8. Output masking ---
        if cond_mask is not None and clean_images is not None:
            output = cond_mask * clean_images + (1 - cond_mask) * output

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    def forward_with_dpmsolver(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass compatible with DPM-Solver sampler (no variance prediction).

        The DPM-Solver calls ``model(x, t, cond, **model_kwargs)`` where
        ``cond`` is the text/image embedding. This thin wrapper adapts the
        call signature.
        """
        kwargs.pop("data_info", None)  # Not used by native model
        mask = kwargs.pop("mask", encoder_attention_mask)

        # Cast inputs to model dtype (c-radio outputs float32, model may be bf16)
        model_dtype = next(self.parameters()).dtype
        hidden_states = hidden_states.to(model_dtype)
        encoder_hidden_states = encoder_hidden_states.to(model_dtype)

        output = self.forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            encoder_attention_mask=mask,
            return_dict=False,
            **kwargs,
        )[0]
        return output
