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

from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask
from torch.nn.attention.flex_attention import flex_attention as flex_attn_func

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000
# Disable DDP optimizer to avoid issues with higher order ops
torch._dynamo.config.optimize_ddp = False
flex_attn_func_compiled = torch.compile(flex_attn_func)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
        flex_attn_block_mask: BlockMask | None = None,
        flex_attn_score_mod: Callable[[int, int, int, int, Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
        self.flex_attn_block_mask = flex_attn_block_mask
        self.flex_attn_score_mod = flex_attn_score_mod

    def forward(self, x: Tensor, pos=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # [3, B, H, N, C]
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if self.flex_attn_block_mask is not None or self.flex_attn_score_mod is not None:
            x = flex_attn_func_compiled(
                q.to(v.dtype),
                k.to(v.dtype),
                v,
                block_mask=self.flex_attn_block_mask,
                score_mod=self.flex_attn_score_mod,
            )
        elif self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float | Tensor = 1e-5, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor.div_(keep_prob)
    output = x * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
        flex_attn_block_mask: BlockMask | None = None,
        flex_attn_score_mod: Callable[[int, int, int, int, Tensor], Tensor] | None = None,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            fused_attn=fused_attn,
            rope=rope,
            flex_attn_block_mask=flex_attn_block_mask,
            flex_attn_score_mod=flex_attn_score_mod,
        )

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, bias=ffn_bias
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

    def forward(self, x: Tensor, pos=None) -> Tensor:
        def attn_residual_func(x: Tensor, pos=None) -> Tensor:
            return self.ls1(self.attn(self.norm1(x), pos=pos))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        if self.training and self.sample_drop_ratio > 0.1:
            # the overhead is compensated only for a drop path rate larger than 0.1
            x = drop_add_residual_stochastic_depth(
                x, pos=pos, residual_func=attn_residual_func, sample_drop_ratio=self.sample_drop_ratio
            )
            x = drop_add_residual_stochastic_depth(
                x, residual_func=ffn_residual_func, sample_drop_ratio=self.sample_drop_ratio
            )
        elif self.training and self.sample_drop_ratio > 0.0:
            x = x + self.drop_path1(attn_residual_func(x, pos=pos))
            x = x + self.drop_path2(ffn_residual_func(x))
        else:
            x = x + attn_residual_func(x, pos=pos)
            x = x + ffn_residual_func(x)
        return x


def drop_add_residual_stochastic_depth(
    x: Tensor, residual_func: Callable[[Tensor], Tensor], sample_drop_ratio: float = 0.0, pos=None
) -> Tensor:
    # 1) extract subset using permutation
    b, n, d = x.shape
    sample_subset_size = max(int(b * (1 - sample_drop_ratio)), 1)
    brange = (torch.randperm(b, device=x.device))[:sample_subset_size]
    x_subset = x[brange]

    # 2) apply residual_func to get residual
    if pos is not None:
        # if necessary, apply rope to the subset
        pos = pos[brange]
        residual = residual_func(x_subset, pos=pos)
    else:
        residual = residual_func(x_subset)

    x_flat = x.flatten(1)
    residual = residual.flatten(1)

    residual_scale_factor = b / sample_subset_size

    # 3) add the residual
    x_plus_residual = torch.index_add(x_flat, 0, brange, residual.to(dtype=x.dtype), alpha=residual_scale_factor)
    return x_plus_residual.view_as(x)


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Callable | None = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (image_HW[0] // patch_HW[0], image_HW[1] // patch_HW[1])

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x


# ========== Encoder-Decoder Components ==========


class DecoderBlock(nn.Module):
    """
    Decoder block for encoder-decoder architecture.
    Works like a transformer decoder layer except the keys/values are provided by the encoder (already normalized).
    """

    class SelfAttnBlock(nn.Module):
        def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool,
            qk_norm: bool,
            flex_attn_score_mod=None,
        ):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.gs_self_attn = Attention(
                dim,
                num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                fused_attn=True,
                flex_attn_score_mod=flex_attn_score_mod,
            )

        def forward(self, gs_tokens: torch.Tensor) -> torch.Tensor:
            queries_normed = self.norm(gs_tokens)
            return self.gs_self_attn(queries_normed)

    class CrossAttnBlock(nn.Module):
        def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool,
            q_norm: bool,
            safe_mode: bool = False,
        ):
            super().__init__()
            self.num_heads = num_heads
            self.gs_token_norm = nn.LayerNorm(dim)
            self.q_norm = nn.LayerNorm(dim // num_heads) if q_norm else nn.Identity()
            self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.out_proj = nn.Linear(dim, dim)
            self.safe_mode = safe_mode

        def forward(self, gs_tokens: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
            gs_tokens_normed = self.gs_token_norm(gs_tokens)
            q = rearrange(self.q_proj(gs_tokens_normed), "b n (h d) -> b h n d", h=self.num_heads)
            q = self.q_norm(q)

            if self.safe_mode:
                # Add numerical stability check for bf16
                # Clamp extreme values before attention to prevent overflow
                q = torch.clamp(q, min=-50, max=50)

            cross_attn_output = nn.functional.scaled_dot_product_attention(q, keys, values)
            cross_attn_output = rearrange(cross_attn_output, "b h n d -> b n (h d)")

            if self.safe_mode:
                # Clamp output to prevent explosion in next layer
                cross_attn_output = torch.clamp(cross_attn_output, min=-100, max=100)

            return self.out_proj(cross_attn_output)

    class MlpBlock(nn.Module):
        def __init__(self, dim: int, mlp_ratio: float, ffn_bias: bool):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.mlp = Mlp(dim, int(dim * mlp_ratio), dim, bias=ffn_bias)

        def forward(self, gs_tokens: torch.Tensor) -> torch.Tensor:
            return self.mlp(self.norm(gs_tokens))

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        proj_bias: bool,
        ffn_bias: bool,
        qk_norm: bool,
        init_values: float | None = None,
        attn_score_mod=None,
        attn_order: Literal["self_cross", "cross_self"] = "self_cross",
        safe_mode: bool = False,
    ):
        super().__init__()

        self.attn_order = attn_order

        def make_scale() -> nn.Module:
            return LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        self.gs_self_attn = DecoderBlock.SelfAttnBlock(
            dim,
            num_heads,
            qkv_bias,
            qk_norm=qk_norm,
            flex_attn_score_mod=attn_score_mod,
        )
        self.gs_self_attn_scale = make_scale()

        self.gs_cross_attn = DecoderBlock.CrossAttnBlock(
            dim,
            num_heads,
            qkv_bias,
            q_norm=qk_norm,
            safe_mode=safe_mode,
        )
        self.gs_cross_attn_scale = make_scale()

        self.mlp = DecoderBlock.MlpBlock(dim, mlp_ratio, ffn_bias)
        self.mlp_scale = make_scale()

    def forward(self, gs_tokens: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        if self.attn_order == "cross_self":
            gs_tokens = gs_tokens + self.gs_cross_attn_scale(self.gs_cross_attn(gs_tokens, keys, values))
            gs_tokens = gs_tokens + self.gs_self_attn_scale(self.gs_self_attn(gs_tokens))
        elif self.attn_order == "self_cross":
            gs_tokens = gs_tokens + self.gs_self_attn_scale(self.gs_self_attn(gs_tokens))
            gs_tokens = gs_tokens + self.gs_cross_attn_scale(self.gs_cross_attn(gs_tokens, keys, values))
        else:
            raise ValueError(f"Invalid attn_order: {self.attn_order}")

        gs_tokens = gs_tokens + self.mlp_scale(self.mlp(gs_tokens))

        return gs_tokens


class EncDecBackbone(nn.Module):
    """
    An encoder-decoder backbone for the EncDec architecture.

    Encoder: a stack of ViT blocks which produce a latent representation. This is followed by a key-value projection to produce a key and value for the decoder.

    Decoder: a stack of transformer decoder layers which attend from GS tokens to the encoder output and among themselves.
    """

    def __init__(self, opt):
        super().__init__()

        self.opt = opt

        if self.opt.decoder_attn_order == "self_cross":
            print(
                f"[WARNING] legacy decoder_attn_order={self.opt.decoder_attn_order}, use 'cross_self' for new experiments"
            )

        self.encoder = nn.Sequential(
            *[
                Block(
                    self.opt.enc_embed_dim,
                    self.opt.enc_num_heads,
                    self.opt.mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    init_values=0.01 if self.opt.use_layerscale else None,
                    qk_norm=True,
                    rope=None,
                    flex_attn_block_mask=None,
                )
                for _ in range(self.opt.enc_depth)
            ]
        )

        self.encoder_norm = nn.LayerNorm(self.opt.enc_embed_dim)

        self.kv_proj = nn.Linear(self.opt.enc_embed_dim, self.opt.enc_embed_dim * 2, bias=True)

        # normalize the k projection, q are normalized in the attention block
        self.k_proj_norm = nn.LayerNorm(self.opt.enc_embed_dim // self.opt.enc_num_heads)

        if self.opt.num_dynamic_gs_tokens > 0:
            block_id = [0 for _ in range(self.opt.num_gs_tokens)] + [1 for _ in range(self.opt.num_dynamic_gs_tokens)]
            block_id = torch.tensor(block_id).cuda()

            def block_causal_score_mod(score, b, h, q_idx, kv_idx):
                same_block_mask = block_id[q_idx] == block_id[kv_idx]
                causal_mask = q_idx >= kv_idx
                return torch.where(same_block_mask | causal_mask, score, float("-inf"))

            score_mod = block_causal_score_mod
        elif self.opt.use_causal_mask:

            def causal_score_mod(score, b, h, q_idx, kv_idx):
                causal_mask = q_idx >= kv_idx
                return torch.where(causal_mask, score, float("-inf"))

            score_mod = causal_score_mod
        else:
            score_mod = None

        dec_depth = self.opt.dec_depth if self.opt.dec_depth is not None else self.opt.enc_depth
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    self.opt.enc_embed_dim,
                    self.opt.enc_num_heads,
                    self.opt.mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    init_values=5e-3 * self.opt.gs_token_std if self.opt.use_layerscale else None,
                    qk_norm=True,
                    attn_score_mod=score_mod,
                    attn_order=self.opt.decoder_attn_order,
                    safe_mode=self.opt.safe_mode,
                )
                for _ in range(dec_depth)
            ]
        )

    def _encode_to_kv(self, image_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image_features = self.encoder(image_features)
        image_features = self.encoder_norm(image_features)
        image_feature_keys, image_feature_values = rearrange(
            self.kv_proj(image_features),
            "b n (kv h c) -> kv b h n c",
            kv=2,
            h=self.opt.enc_num_heads,
        )
        image_feature_keys = self.k_proj_norm(image_feature_keys)
        return image_feature_keys, image_feature_values

    def forward(self, image_features: torch.Tensor, gs_tokens: torch.Tensor) -> torch.Tensor:
        image_feature_keys, image_feature_values = self._encode_to_kv(image_features)
        if self.opt.safe_mode:
            # Clamp keys and values for numerical stability with bf16
            image_feature_keys = torch.clamp(image_feature_keys, min=-50, max=50)
            image_feature_values = torch.clamp(image_feature_values, min=-50, max=50)

        for layer in self.decoder_blocks:
            gs_tokens = layer(
                gs_tokens=gs_tokens,
                keys=image_feature_keys,
                values=image_feature_values,
            )
        return gs_tokens
