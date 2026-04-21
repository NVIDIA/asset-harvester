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
Convert SanaMS checkpoints to diffusers SparseViewDiTTransformer2DModelNative format.

Key mappings:
  Original SanaMS                          -> Diffusers Native
  -----------------------------------------------
  blocks.N.attn.qkv.weight                -> transformer_blocks.N.attn1.to_{q,k,v}.weight  (split 3-way)
  blocks.N.attn.proj.{w,b}                -> transformer_blocks.N.attn1.to_out.0.{w,b}
  blocks.N.cross_attn.q_linear.{w,b}      -> transformer_blocks.N.attn2.to_q.{w,b}
  blocks.N.cross_attn.kv_linear.{w,b}     -> transformer_blocks.N.attn2.to_{k,v}.{w,b}  (split 2-way)
  blocks.N.cross_attn.proj.{w,b}          -> transformer_blocks.N.attn2.to_out.0.{w,b}
  blocks.N.mlp.inverted_conv.conv.{w,b}   -> transformer_blocks.N.ff.conv_inverted.{w,b}
  blocks.N.mlp.depth_conv.conv.{w,b}      -> transformer_blocks.N.ff.conv_depth.{w,b}
  blocks.N.mlp.point_conv.conv.weight      -> transformer_blocks.N.ff.conv_point.weight
  blocks.N.scale_shift_table              -> transformer_blocks.N.scale_shift_table
  x_embedder.proj.{w,b}                   -> patch_embed.proj.{w,b}
  t_embedder.mlp.0.{w,b}                  -> time_embed.emb.timestep_embedder.linear_1.{w,b}
  t_embedder.mlp.2.{w,b}                  -> time_embed.emb.timestep_embedder.linear_2.{w,b}
  t_block.1.{w,b}                         -> time_embed.linear.{w,b}
  y_embedder.y_proj.fc1.{w,b}             -> caption_projection.linear_1.{w,b}
  y_embedder.y_proj.fc2.{w,b}             -> caption_projection.linear_2.{w,b}
  attention_y_norm.weight                  -> caption_norm.weight
  final_layer.linear.{w,b}                -> proj_out.{w,b}
  final_layer.scale_shift_table           -> scale_shift_table
  pos_embed                               -> (dropped — diffusers PatchEmbed recomputes it)
  y_embedder.y_embedding                  -> (dropped — not used in diffusers)
  cam_emb.*                               -> cam_emb.*              (same)
  cam_emb_block.*                          -> cam_emb_block.*        (same)
  brightness_emb.mlp.*                    -> brightness_emb_module.mlp.*  (renamed)
  brightness_emb_block.*                   -> brightness_emb_block.*      (same)
"""

import argparse
import os

import torch


def convert_sana_ms_to_diffusers(
    src_state_dict: dict[str, torch.Tensor],
    hidden_size: int = 2240,
) -> dict[str, torch.Tensor]:
    """
    Convert a SanaMS state dict to the diffusers native format.

    Args:
        src_state_dict: Original SanaMS state dict.
        hidden_size: Hidden dimension (needed to split fused QKV).

    Returns:
        New state dict compatible with SparseViewDiTTransformer2DModelNative.
    """
    dst: dict[str, torch.Tensor] = {}
    consumed = set()

    def _copy(src_key: str, dst_key: str):
        if src_key in src_state_dict:
            dst[dst_key] = src_state_dict[src_key]
            consumed.add(src_key)

    # --- Patch embedding ---
    _copy("x_embedder.proj.weight", "patch_embed.proj.weight")
    _copy("x_embedder.proj.bias", "patch_embed.proj.bias")

    # --- Timestep embedding ---
    # Original: t_embedder.mlp = Sequential(Linear, SiLU, Linear)
    # Diffusers: time_embed = AdaLayerNormSingle containing:
    #   .emb = PixArtAlphaCombinedTimestepSizeEmbeddings
    #     .timestep_embedder = TimestepEmbedding(linear_1, act, linear_2)
    #   .silu + .linear  (the t_block equivalent)
    _copy("t_embedder.mlp.0.weight", "time_embed.emb.timestep_embedder.linear_1.weight")
    _copy("t_embedder.mlp.0.bias", "time_embed.emb.timestep_embedder.linear_1.bias")
    _copy("t_embedder.mlp.2.weight", "time_embed.emb.timestep_embedder.linear_2.weight")
    _copy("t_embedder.mlp.2.bias", "time_embed.emb.timestep_embedder.linear_2.bias")

    # t_block = Sequential(SiLU, Linear) -> time_embed.linear
    _copy("t_block.1.weight", "time_embed.linear.weight")
    _copy("t_block.1.bias", "time_embed.linear.bias")

    # --- Caption projection ---
    _copy("y_embedder.y_proj.fc1.weight", "caption_projection.linear_1.weight")
    _copy("y_embedder.y_proj.fc1.bias", "caption_projection.linear_1.bias")
    _copy("y_embedder.y_proj.fc2.weight", "caption_projection.linear_2.weight")
    _copy("y_embedder.y_proj.fc2.bias", "caption_projection.linear_2.bias")

    # Caption norm
    _copy("attention_y_norm.weight", "caption_norm.weight")

    # --- Output layer ---
    _copy("final_layer.linear.weight", "proj_out.weight")
    _copy("final_layer.linear.bias", "proj_out.bias")
    _copy("final_layer.scale_shift_table", "scale_shift_table")

    # --- Camera / brightness / aug / skeletal embedders ---
    # These use the same internal structure, just slightly different names
    for src_prefix, dst_prefix in [
        ("cam_emb.", "cam_emb."),
        ("cam_emb_block.", "cam_emb_block."),
        ("aug_emb.", "aug_emb_module."),
        ("aug_emb_block.", "aug_emb_block."),
        ("brightness_emb.", "brightness_emb_module."),
        ("brightness_emb_block.", "brightness_emb_block."),
        ("skeletal_emb.", "skeletal_emb_module."),
        ("skeletal_emb_block.", "skeletal_emb_block."),
    ]:
        for k in list(src_state_dict.keys()):
            if k.startswith(src_prefix):
                suffix = k[len(src_prefix) :]
                dst[dst_prefix + suffix] = src_state_dict[k]
                consumed.add(k)

    # --- Transformer blocks ---
    # Detect block indices
    block_indices = set()
    for k in src_state_dict.keys():
        if k.startswith("blocks."):
            idx = int(k.split(".")[1])
            block_indices.add(idx)

    for idx in sorted(block_indices):
        src_pre = f"blocks.{idx}"
        dst_pre = f"transformer_blocks.{idx}"

        # Self-attention: split fused QKV
        qkv_key = f"{src_pre}.attn.qkv.weight"
        if qkv_key in src_state_dict:
            qkv_w = src_state_dict[qkv_key]  # [3*D, D]
            q_w, k_w, v_w = qkv_w.chunk(3, dim=0)
            dst[f"{dst_pre}.attn1.to_q.weight"] = q_w
            dst[f"{dst_pre}.attn1.to_k.weight"] = k_w
            dst[f"{dst_pre}.attn1.to_v.weight"] = v_w
            consumed.add(qkv_key)

        # Self-attention output projection
        _copy(f"{src_pre}.attn.proj.weight", f"{dst_pre}.attn1.to_out.0.weight")
        _copy(f"{src_pre}.attn.proj.bias", f"{dst_pre}.attn1.to_out.0.bias")

        # Cross-attention Q
        _copy(f"{src_pre}.cross_attn.q_linear.weight", f"{dst_pre}.attn2.to_q.weight")
        _copy(f"{src_pre}.cross_attn.q_linear.bias", f"{dst_pre}.attn2.to_q.bias")

        # Cross-attention KV: split fused KV
        kv_w_key = f"{src_pre}.cross_attn.kv_linear.weight"
        kv_b_key = f"{src_pre}.cross_attn.kv_linear.bias"
        if kv_w_key in src_state_dict:
            kv_w = src_state_dict[kv_w_key]  # [2*D, D]
            k_w, v_w = kv_w.chunk(2, dim=0)
            dst[f"{dst_pre}.attn2.to_k.weight"] = k_w
            dst[f"{dst_pre}.attn2.to_v.weight"] = v_w
            consumed.add(kv_w_key)
        if kv_b_key in src_state_dict:
            kv_b = src_state_dict[kv_b_key]  # [2*D]
            k_b, v_b = kv_b.chunk(2, dim=0)
            dst[f"{dst_pre}.attn2.to_k.bias"] = k_b
            dst[f"{dst_pre}.attn2.to_v.bias"] = v_b
            consumed.add(kv_b_key)

        # Cross-attention output projection
        _copy(f"{src_pre}.cross_attn.proj.weight", f"{dst_pre}.attn2.to_out.0.weight")
        _copy(f"{src_pre}.cross_attn.proj.bias", f"{dst_pre}.attn2.to_out.0.bias")

        # FFN (GLUMBConv)
        # Original: mlp.inverted_conv.conv -> ff.conv_inverted
        _copy(f"{src_pre}.mlp.inverted_conv.conv.weight", f"{dst_pre}.ff.conv_inverted.weight")
        _copy(f"{src_pre}.mlp.inverted_conv.conv.bias", f"{dst_pre}.ff.conv_inverted.bias")
        _copy(f"{src_pre}.mlp.depth_conv.conv.weight", f"{dst_pre}.ff.conv_depth.weight")
        _copy(f"{src_pre}.mlp.depth_conv.conv.bias", f"{dst_pre}.ff.conv_depth.bias")
        _copy(f"{src_pre}.mlp.point_conv.conv.weight", f"{dst_pre}.ff.conv_point.weight")

        # Scale-shift table
        _copy(f"{src_pre}.scale_shift_table", f"{dst_pre}.scale_shift_table")

    # --- Keys intentionally skipped ---
    skipped_prefixes = ("pos_embed", "y_embedder.y_embedding")
    for k in src_state_dict.keys():
        if k not in consumed:
            is_skipped = any(k.startswith(p) for p in skipped_prefixes)
            if not is_skipped:
                print(f"WARNING: unconsumed key: {k} (shape={list(src_state_dict[k].shape)})")

    return dst


def main():
    parser = argparse.ArgumentParser(description="Convert SanaMS checkpoint to diffusers native format")
    parser.add_argument("--src", type=str, required=True, help="Path to source .pth checkpoint")
    parser.add_argument("--dst", type=str, required=True, help="Path to output .pth checkpoint")
    parser.add_argument("--hidden_size", type=int, default=2240, help="Hidden dimension")
    args = parser.parse_args()

    print(f"Loading source checkpoint: {args.src}")
    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    src_sd = ckpt.get("state_dict", ckpt)

    print(f"Converting {len(src_sd)} keys (hidden_size={args.hidden_size})...")
    dst_sd = convert_sana_ms_to_diffusers(src_sd, args.hidden_size)

    print(f"Saving converted checkpoint ({len(dst_sd)} keys) to: {args.dst}")
    os.makedirs(os.path.dirname(args.dst) or ".", exist_ok=True)
    torch.save(dst_sd, args.dst)
    print("Done!")


if __name__ == "__main__":
    main()
