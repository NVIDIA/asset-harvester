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
TokenGS: Encoder-decoder model for 3D scene reconstruction from sparse views.
"""

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from lpips import LPIPS

from ..attention import EncDecBackbone, PatchEmbed
from ..gs import GaussianRenderer
from ..options import Options
from .activations import ClipActivationHead, DirectClipHead, ObjaverseActivationHead


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


@contextmanager
def freeze_model_parameters(model: nn.Module) -> Generator[None, None, None]:
    """
    Sets all model parameters to requires_grad=False for the duration and resets to original afterwards.

    This works correctly for models which may already have some parameters frozen - it will *not* unfreeze them.
    """
    original_state: dict[str, bool] = {}
    for name, param in model.named_parameters():
        original_state[name] = param.requires_grad
        param.requires_grad = False
    try:
        yield
    finally:
        for name, param in model.named_parameters():
            param.requires_grad = original_state[name]


@dataclass
class ModelInputEncoder:
    """Input data for the encoder (input views)"""

    # Image features split by type (input views only)
    images_rgb: torch.Tensor  # [B, V_in, 3, H, W] - RGB channels (normalized for encoder)
    plucker: torch.Tensor  # [B, V_in, 6, H, W] - Plucker coordinates

    # Ray information (input views only)
    rays_os: torch.Tensor  # [B, V_in, 3, H, W] - ray origins
    rays_ds: torch.Tensor  # [B, V_in, 3, H, W] - ray directions

    # Camera parameters for input views
    intrinsics_input: torch.Tensor  # [B, V_in, 4]
    cam_to_world_input: torch.Tensor  # [B, V_in, 4, 4]

    # Optional time embeddings for input views
    time_embedding_input: torch.Tensor | None = None  # [B, V_in, T_dim, H, W]
    time_embedding_target: torch.Tensor | None = None  # [B, V_out, T_dim, H, W]

    # Unnormalized RGB images for supervision (e.g., TTT)
    images_rgb_unnormalized: torch.Tensor | None = None  # [B, V_in, 3, H, W]

    def __add__(self, other: "ModelInputEncoder") -> "ModelInputEncoder":
        """
        Merge two ModelInputEncoder instances by concatenating along the view dimension (dim=1).

        Args:
            other: Another ModelInputEncoder instance to merge with

        Returns:
            A new ModelInputEncoder with concatenated views
        """
        # Verify batch sizes match
        assert self.images_rgb.shape[0] == other.images_rgb.shape[0], (
            f"Batch sizes must match: {self.images_rgb.shape[0]} vs {other.images_rgb.shape[0]}"
        )

        # Concatenate along view dimension (dim=1)
        merged = ModelInputEncoder(
            images_rgb=torch.cat([self.images_rgb, other.images_rgb], dim=1),
            plucker=torch.cat([self.plucker, other.plucker], dim=1),
            rays_os=torch.cat([self.rays_os, other.rays_os], dim=1),
            rays_ds=torch.cat([self.rays_ds, other.rays_ds], dim=1),
            intrinsics_input=torch.cat([self.intrinsics_input, other.intrinsics_input], dim=1),
            cam_to_world_input=torch.cat([self.cam_to_world_input, other.cam_to_world_input], dim=1),
        )

        # Handle optional time embeddings
        if self.time_embedding_input is not None and other.time_embedding_input is not None:
            merged.time_embedding_input = torch.cat([self.time_embedding_input, other.time_embedding_input], dim=1)
        elif self.time_embedding_input is not None or other.time_embedding_input is not None:
            raise ValueError("Both ModelInputEncoder instances must have time_embedding_input or neither should")

        if self.time_embedding_target is not None and other.time_embedding_target is not None:
            merged.time_embedding_target = torch.cat([self.time_embedding_target, other.time_embedding_target], dim=1)
        elif self.time_embedding_target is not None or other.time_embedding_target is not None:
            raise ValueError("Both ModelInputEncoder instances must have time_embedding_target or neither should")

        # Handle optional unnormalized images
        if self.images_rgb_unnormalized is not None and other.images_rgb_unnormalized is not None:
            merged.images_rgb_unnormalized = torch.cat(
                [self.images_rgb_unnormalized, other.images_rgb_unnormalized], dim=1
            )
        elif self.images_rgb_unnormalized is not None or other.images_rgb_unnormalized is not None:
            raise ValueError("Both ModelInputEncoder instances must have images_rgb_unnormalized or neither should")

        return merged

    @classmethod
    def merge(cls, encoders: list["ModelInputEncoder"]) -> "ModelInputEncoder":
        """
        Merge multiple ModelInputEncoder instances by concatenating along the view dimension.

        Args:
            encoders: List of ModelInputEncoder instances to merge

        Returns:
            A new ModelInputEncoder with concatenated views
        """
        if len(encoders) == 0:
            raise ValueError("Cannot merge empty list of encoders")
        if len(encoders) == 1:
            return encoders[0]

        from functools import reduce

        return reduce(lambda a, b: a + b, encoders)


@dataclass
class EncoderLatent:
    """Encoder output containing keys and values for cross-attention"""

    keys: torch.Tensor  # [B, H, N, C//H] - attention keys
    values: torch.Tensor  # [B, H, N, C//H] - attention values


@dataclass
class ModelInputDecoder:
    """Input data for the decoder (target views)"""

    # Target time embedding (used to condition GS tokens)
    time_embedding_target: torch.Tensor | None = None  # [B, 1, T_dim, H, W]

    # Camera parameters for rendering (output views)
    cam_view: torch.Tensor = None  # [B, V_out, 4, 4] - Camera view matrices for rendering
    intrinsics: torch.Tensor = None  # [B, V_out, 4] - Intrinsics for rendering

    def __add__(self, other: "ModelInputDecoder") -> "ModelInputDecoder":
        """
        Merge two ModelInputDecoder instances by concatenating along the view dimension (dim=1).

        Args:
            other: Another ModelInputDecoder instance to merge with

        Returns:
            A new ModelInputDecoder with concatenated views
        """
        # Verify batch sizes match
        if self.cam_view is not None and other.cam_view is not None:
            assert self.cam_view.shape[0] == other.cam_view.shape[0], (
                f"Batch sizes must match: {self.cam_view.shape[0]} vs {other.cam_view.shape[0]}"
            )

        # Concatenate camera parameters along view dimension (dim=1)
        merged_cam_view = None
        merged_intrinsics = None

        if self.cam_view is not None and other.cam_view is not None:
            merged_cam_view = torch.cat([self.cam_view, other.cam_view], dim=1)
        elif self.cam_view is not None or other.cam_view is not None:
            raise ValueError("Both ModelInputDecoder instances must have cam_view or neither should")

        if self.intrinsics is not None and other.intrinsics is not None:
            merged_intrinsics = torch.cat([self.intrinsics, other.intrinsics], dim=1)
        elif self.intrinsics is not None or other.intrinsics is not None:
            raise ValueError("Both ModelInputDecoder instances must have intrinsics or neither should")

        # Handle optional time embeddings - for decoder, time_embedding_target is typically [B, 1, T_dim, H, W]
        # When merging, we keep the first one (or verify they're the same)
        merged_time_embedding_target = None
        if self.time_embedding_target is not None and other.time_embedding_target is not None:
            # For time_embedding_target, we concatenate along view dimension as well
            merged_time_embedding_target = torch.cat([self.time_embedding_target, other.time_embedding_target], dim=1)
        elif self.time_embedding_target is not None:
            merged_time_embedding_target = self.time_embedding_target
        elif other.time_embedding_target is not None:
            merged_time_embedding_target = other.time_embedding_target

        return ModelInputDecoder(
            time_embedding_target=merged_time_embedding_target,
            cam_view=merged_cam_view,
            intrinsics=merged_intrinsics,
        )

    @classmethod
    def merge(cls, decoders: list["ModelInputDecoder"]) -> "ModelInputDecoder":
        """
        Merge multiple ModelInputDecoder instances by concatenating along the view dimension.

        Args:
            decoders: List of ModelInputDecoder instances to merge

        Returns:
            A new ModelInputDecoder with concatenated views
        """
        if len(decoders) == 0:
            raise ValueError("Cannot merge empty list of decoders")
        if len(decoders) == 1:
            return decoders[0]

        from functools import reduce

        return reduce(lambda a, b: a + b, decoders)


@dataclass
class ModelInput:
    """Complete input data for the TokenGS model (combines encoder and decoder inputs)"""

    encoder: ModelInputEncoder
    decoder: ModelInputDecoder

    def __add__(self, other: "ModelInput") -> "ModelInput":
        """
        Merge two ModelInput instances by concatenating along the view dimension.

        Args:
            other: Another ModelInput instance to merge with

        Returns:
            A new ModelInput with concatenated views
        """
        return ModelInput(
            encoder=self.encoder + other.encoder,
            decoder=self.decoder + other.decoder,
        )

    @classmethod
    def merge(cls, inputs: list["ModelInput"]) -> "ModelInput":
        """
        Merge multiple ModelInput instances by concatenating along the view dimension.

        Args:
            inputs: List of ModelInput instances to merge

        Returns:
            A new ModelInput with concatenated views
        """
        if len(inputs) == 0:
            raise ValueError("Cannot merge empty list of inputs")
        if len(inputs) == 1:
            return inputs[0]

        from functools import reduce

        return reduce(lambda a, b: a + b, inputs)

    def to_ttt(
        self, masks_output: torch.Tensor | None = None, has_mask: bool = False
    ) -> tuple["ModelInput", "ModelSupervision"]:
        """
        Create ModelInput and ModelSupervision for test-time training.

        For TTT, we want to:
        1. Render back to the input views (not target views)
        2. Supervise using the input view RGB images as ground truth

        Args:
            masks_output: Optional masks [B, V_in, 1, H, W]. If None, creates all-ones masks.
            has_mask: Whether masks are present/should be used

        Returns:
            Tuple of (ModelInput for TTT, ModelSupervision for TTT)
        """
        # Invert cam_to_world to get cam_view (world_to_cam)
        cam_view_input = torch.inverse(self.encoder.cam_to_world_input).transpose(-2, -1)

        # Create decoder input that renders to input views
        decoder_input_ttt = ModelInputDecoder(
            time_embedding_target=None,  # Input views don't need target time embedding
            cam_view=cam_view_input,
            intrinsics=self.encoder.intrinsics_input,
        )

        # Create ModelInput for TTT (same encoder, modified decoder)
        model_input_ttt = ModelInput(
            encoder=self.encoder,
            decoder=decoder_input_ttt,
        )

        # Create supervision using unnormalized input view images
        # Note: We must use unnormalized images to match the supervision format used elsewhere
        assert self.encoder.images_rgb_unnormalized is not None, (
            "TTT requires unnormalized images for supervision. "
            "Ensure split_data() is called with a batch containing 'images_input'."
        )
        images_for_supervision = self.encoder.images_rgb_unnormalized

        if masks_output is None:
            # Default to all-ones masks (no masking)
            B, V_in, _, H, W = images_for_supervision.shape
            masks_output = torch.ones(
                B, V_in, 1, H, W, dtype=images_for_supervision.dtype, device=images_for_supervision.device
            )
        else:
            B = images_for_supervision.shape[0]

        # Convert has_mask bool to tensor [B]
        has_mask_tensor = torch.full((B,), has_mask, dtype=torch.bool, device=images_for_supervision.device)

        supervision_ttt = ModelSupervision(
            images_output=images_for_supervision,
            masks_output=masks_output,
            has_mask=has_mask_tensor,
            # Pass rays for 3D visibility loss if needed
            rays_os=self.encoder.rays_os,
            rays_ds=self.encoder.rays_ds,
        )

        return model_input_ttt, supervision_ttt

    @property
    def batch_size(self) -> int:
        return self.encoder.images_rgb.shape[0]


@dataclass
class ModelSupervision:
    """Supervision data for the TokenGS model"""

    images_output: torch.Tensor  # [B, V, 3, img_size, img_size] - ground truth images
    masks_output: torch.Tensor  # [B, V, 1, img_size, img_size] - ground truth masks
    has_mask: torch.Tensor  # [B] - boolean mask indicating if a given scene has masks

    # Ray information for 3D visibility loss
    rays_os: torch.Tensor | None = None  # [B, V, 3, H, W] - ray origins
    rays_ds: torch.Tensor | None = None  # [B, V, 3, H, W] - ray directions

    def __post_init__(self):
        assert (self.rays_os is not None) == (self.rays_ds is not None), (
            "rays_os and rays_ds must be either both None or both not None"
        )

    def __add__(self, other: "ModelSupervision") -> "ModelSupervision":
        """
        Merge two ModelSupervision instances by concatenating along the view dimension (dim=1).

        Args:
            other: Another ModelSupervision instance to merge with

        Returns:
            A new ModelSupervision with concatenated views
        """
        # Verify batch sizes match
        assert self.images_output.shape[0] == other.images_output.shape[0], (
            f"Batch sizes must match: {self.images_output.shape[0]} vs {other.images_output.shape[0]}"
        )

        # Concatenate along view dimension (dim=1)
        merged_images_output = torch.cat([self.images_output, other.images_output], dim=1)
        merged_masks_output = torch.cat([self.masks_output, other.masks_output], dim=1)
        # has_mask [B]: True if any of the concatenated views have mask data
        merged_has_mask = torch.logical_or(self.has_mask, other.has_mask)

        # Handle optional rays: concatenate when both have rays; otherwise set to None (no ray loss for merged)
        merged_rays_os = None
        merged_rays_ds = None
        if self.rays_os is not None and other.rays_os is not None:
            merged_rays_os = torch.cat([self.rays_os, other.rays_os], dim=1)
            merged_rays_ds = torch.cat([self.rays_ds, other.rays_ds], dim=1)
        # If only one has rays, drop rays for merged result so loss code uses image/mask only

        return ModelSupervision(
            images_output=merged_images_output,
            masks_output=merged_masks_output,
            has_mask=merged_has_mask,
            rays_os=merged_rays_os,
            rays_ds=merged_rays_ds,
        )

    @classmethod
    def merge(cls, supervisions: list["ModelSupervision"]) -> "ModelSupervision":
        """
        Merge multiple ModelSupervision instances by concatenating along the view dimension.

        Args:
            supervisions: List of ModelSupervision instances to merge

        Returns:
            A new ModelSupervision with concatenated views
        """
        if len(supervisions) == 0:
            raise ValueError("Cannot merge empty list of supervisions")
        if len(supervisions) == 1:
            return supervisions[0]

        # Use reduce to apply __add__ sequentially
        from functools import reduce

        return reduce(lambda a, b: a + b, supervisions)


def split_data(batch: dict, opt: Options) -> tuple[ModelInput, ModelSupervision]:
    """
    Split the batch dictionary into structured ModelInput and ModelSupervision objects.

    Args:
        batch: Dictionary from the dataloader containing mixed input and supervision data
        opt: Options object containing model configuration

    Returns:
        Tuple of (ModelInput, ModelSupervision)
    """
    # Extract the combined image tensor [B, V, C, H, W]
    images = batch["input"]  # C = 3 (RGB) + time_embedding_dim + 6 (plucker)

    # Split channels (all views initially)
    images_rgb_all = images[:, :, :3, :, :]
    plucker_all = images[:, :, -6:, :, :]

    # Extract time embeddings if enabled
    time_embedding_input = None
    time_embedding_target = None
    if opt.time_embedding:
        # Input views time embedding
        time_embedding_input = images[:, : opt.num_input_views, 3 : 3 + opt.time_embedding_dim, :, :]
        # Target view time embedding (first view after input views)
        time_embedding_target = images[
            :, opt.num_input_views : opt.num_input_views + 1, 3 : 3 + opt.time_embedding_dim, :, :
        ]

    # Handle optional input supervision mode
    if opt.use_input_supervision:
        cam_view = batch["cam_view_all"]
        intrinsics = batch["intrinsics_all"]
        images_output = batch["images_all"]
        masks_output = batch["masks_all"]
    else:
        cam_view = batch["cam_view"]
        intrinsics = batch["intrinsics"]
        images_output = batch["images_output"]
        masks_output = batch["masks_output"]

    # Create encoder input (only input views)
    encoder_input = ModelInputEncoder(
        images_rgb=images_rgb_all[:, : opt.num_input_views],
        plucker=plucker_all[:, : opt.num_input_views],
        rays_os=batch["rays_os"][:, : opt.num_input_views],
        rays_ds=batch["rays_ds"][:, : opt.num_input_views],
        intrinsics_input=batch["intrinsics_input"],
        cam_to_world_input=batch["cam_to_world_input"],
        time_embedding_input=time_embedding_input,
        time_embedding_target=time_embedding_target,
        images_rgb_unnormalized=batch["images_input"],
    )

    # Create decoder input (target time + rendering params)
    decoder_input = ModelInputDecoder(
        time_embedding_target=time_embedding_target,
        cam_view=cam_view,
        intrinsics=intrinsics,
    )

    # Combine into ModelInput
    model_input = ModelInput(
        encoder=encoder_input,
        decoder=decoder_input,
    )

    supervision = ModelSupervision(
        images_output=images_output,
        masks_output=masks_output,
        has_mask=batch["has_mask"],
        # for 3D visibility loss
        rays_os=batch["rays_os"],
        rays_ds=batch["rays_ds"],
    )

    return model_input, supervision


class TokenGS(nn.Module):
    """
    TokenGS model with encoder-decoder architecture.

    Uses separate encoder and decoder with cross-attention for Gaussian token processing.
    """

    def __init__(
        self,
        opt: Options,
    ):
        super().__init__()

        self.opt = opt

        self.img_size = (
            self.opt.img_size if not isinstance(self.opt.img_size, int) else [self.opt.img_size, self.opt.img_size]
        )

        # transformer for the encoder
        self.enc_depth = self.opt.enc_depth
        self.enc_embed_dim = self.opt.enc_embed_dim

        # LayerNorm factory with optional bias
        norm_layer_factory = partial(nn.LayerNorm, bias=not self.opt.layernorm_no_bias)

        # Setup tokenizer (concatenates RGB + Plucker, then linear projection)
        self.patch_embed = PatchEmbed(
            img_size=self.opt.img_size,
            patch_size=self.opt.patch_size,
            in_chans=3,
            embed_dim=self.opt.enc_embed_dim,
            norm_layer=norm_layer_factory if self.opt.use_emb_norm else None,
        )
        self.patch_plucker_embed = PatchEmbed(
            img_size=self.opt.img_size,
            patch_size=self.opt.patch_size,
            in_chans=6,
            embed_dim=self.opt.enc_embed_dim,
            norm_layer=norm_layer_factory if self.opt.use_emb_norm else None,
        )

        if self.opt.time_embedding:
            self.patch_time_embed = PatchEmbed(
                img_size=self.opt.img_size,
                patch_size=self.opt.patch_size,
                in_chans=self.opt.time_embedding_dim,
                embed_dim=self.opt.enc_embed_dim,
                # no normalization for time embedding
                # since it could be zero-initialized
                norm_layer=None,
            )
            self.patch_time_embed_tgt = PatchEmbed(
                img_size=self.opt.img_size,
                patch_size=self.opt.patch_size,
                in_chans=self.opt.time_embedding_dim,
                embed_dim=self.opt.enc_embed_dim,
                # no normalization for time embedding
                # since it could be zero-initialized
                norm_layer=None,
            )

            if self.opt.zero_init_time_embed:
                self.patch_time_embed.proj.weight.data.fill_(0.0)
                self.patch_time_embed.proj.bias.data.fill_(0.0)
                self.patch_time_embed_tgt.proj.weight.data.fill_(0.0)
                self.patch_time_embed_tgt.proj.bias.data.fill_(0.0)

        # Encoder-decoder architecture
        self.enc_dec_backbone = EncDecBackbone(opt)
        # Output normalization after decoder
        self.backbone_norm = nn.LayerNorm(self.opt.enc_embed_dim) if self.opt.use_enc_norm else nn.Identity()

        # Gaussian Renderer
        self.gs = GaussianRenderer(opt)

        # Activation head
        if self.opt.activation_head_type == "clip":
            self.activation_head = ClipActivationHead(opt)
        elif self.opt.activation_head_type == "objaverse":
            self.activation_head = ObjaverseActivationHead(opt)
        else:
            raise ValueError(f"Invalid activation head type: {self.opt.activation_head_type}")

        # LPIPS loss
        if self.opt.lambda_lpips > 0:
            self.lpips_loss = LPIPS(net="vgg")
            self.lpips_loss.requires_grad_(False)

        # Learnable GS tokens
        self.gs_tokens = nn.Parameter(torch.randn(self.opt.num_gs_tokens, self.opt.token_dim))
        nn.init.normal_(self.gs_tokens, std=self.opt.gs_token_std)

        if self.opt.num_dynamic_gs_tokens > 0:
            self.gs_tokens_dynamic = nn.Parameter(torch.randn(self.opt.num_dynamic_gs_tokens, self.opt.token_dim))
            nn.init.normal_(self.gs_tokens_dynamic, std=self.opt.gs_token_std)

    def state_dict(self, **kwargs):
        # remove lpips_loss
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if "lpips_loss" in k:
                del state_dict[k]
        return state_dict

    def forward_encoder(self, encoder_input: ModelInputEncoder) -> EncoderLatent:
        """
        Encode input views into latent representation (keys and values for cross-attention).

        Args:
            encoder_input: ModelInputEncoder containing input view data (only input views)

        Returns:
            EncoderLatent containing keys and values for cross-attention
        """
        B, V, _, H, W = encoder_input.images_rgb.shape
        height = int(H // self.opt.patch_size)
        width = int(W // self.opt.patch_size)

        assert height * self.opt.patch_size == H, f"H={H} must be divisible by patch_size={self.opt.patch_size}"
        assert width * self.opt.patch_size == W, f"W={W} must be divisible by patch_size={self.opt.patch_size}"

        # Reshape for tokenization
        images_rgb_reshaped = encoder_input.images_rgb.reshape(B * V, 3, H, W)
        plucker_reshaped = encoder_input.plucker.reshape(B * V, 6, H, W)

        # Embed RGB patches
        x = self.patch_embed(images_rgb_reshaped)
        # Add Plucker embeddings
        x_plucker_emb = self.patch_plucker_embed(plucker_reshaped)
        x = x + x_plucker_emb  # B*V, N, C

        # Add time embeddings if enabled (processed in 2D image space, then reshaped)
        if self.opt.time_embedding:
            time_emb_reshaped = encoder_input.time_embedding_input.reshape(B * V, self.opt.time_embedding_dim, H, W)
            x_time_emb = self.patch_time_embed(time_emb_reshaped)
            x = x + x_time_emb.reshape_as(x)

        x = x.reshape(B, -1, self.opt.enc_embed_dim)  # B*V, N, C ---> B, V*N, C

        # Run encoder on joint views and produce keys/values
        # x shape: (B, V*N, C) where all views are processed jointly
        image_feature_keys, image_feature_values = self.enc_dec_backbone._encode_to_kv(x)

        return EncoderLatent(
            keys=image_feature_keys,
            values=image_feature_values,
        )

    def get_gs_tokens(self, batch_size: int) -> torch.Tensor:
        """
        Get initial GS tokens (without time conditioning).
        Useful for test-time training where you want to optimize the GS tokens.
        Time conditioning is applied later in forward_decoder.

        Args:
            batch_size: Batch size

        Returns:
            GS tokens with shape [B, num_gs_tokens, C]
        """
        # Create initial GS tokens without time conditioning
        B = batch_size
        batch_gs_tokens = self.gs_tokens.unsqueeze(0).repeat(B, 1, 1)

        if self.opt.num_dynamic_gs_tokens > 0:
            batch_gs_tokens_dynamic = self.gs_tokens_dynamic.unsqueeze(0).repeat(B, 1, 1)
            batch_gs_tokens = torch.cat([batch_gs_tokens, batch_gs_tokens_dynamic], dim=1)

        return batch_gs_tokens

    def _apply_time_embedding_to_gs_tokens(
        self,
        gs_tokens: torch.Tensor,
        decoder_input: ModelInputDecoder,
    ) -> torch.Tensor:
        """
        Apply target time embedding to GS tokens.
        This is called by forward_decoder to condition GS tokens on target time.

        Args:
            gs_tokens: Base GS tokens [B, num_gs_tokens, C]
            decoder_input: ModelInputDecoder containing target time embedding

        Returns:
            Time-conditioned GS tokens [B, num_gs_tokens (+dynamic), C]
        """
        if not self.opt.time_embedding or decoder_input.time_embedding_target is None:
            # No time embedding or no target time provided
            return gs_tokens

        B = gs_tokens.shape[0]
        _, _, T_dim, H_tgt, W_tgt = decoder_input.time_embedding_target.shape

        # Process target time embedding through patch embedding
        time_emb_target = decoder_input.time_embedding_target.reshape(B, T_dim, H_tgt, W_tgt)
        x_time_emb_tgt = self.patch_time_embed_tgt(time_emb_target)

        if self.opt.num_dynamic_gs_tokens > 0:
            # For dynamic tokens: add time embedding only to dynamic tokens
            x_time_emb_tgt = x_time_emb_tgt.reshape(B, -1, self.opt.enc_embed_dim)[
                :, : self.opt.num_dynamic_gs_tokens, :
            ]
            gs_tokens[:, -self.opt.num_dynamic_gs_tokens :] = (
                gs_tokens[:, -self.opt.num_dynamic_gs_tokens :] + x_time_emb_tgt
            )
        else:
            # For standard learnable tokens: add time embedding to all GS tokens
            x_time_emb_tgt = x_time_emb_tgt.reshape(B, -1, self.opt.enc_embed_dim)[:, : self.opt.num_gs_tokens, :]
            gs_tokens = gs_tokens + x_time_emb_tgt

        return gs_tokens

    def forward_decoder(
        self,
        encoder_latent: EncoderLatent,
        decoder_input: ModelInputDecoder,
        gs_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Process latent representation (keys/values) to Gaussians using decoder with cross-attention.

        Args:
            encoder_latent: EncoderLatent containing keys and values from the encoder
            decoder_input: ModelInputDecoder containing rendering parameters and target time
            gs_tokens: Optional precomputed GS tokens [B, num_gs_tokens, C]. If None, creates new ones.

        Returns:
            Gaussians tensor [B, N, 14] where N is the number of Gaussians
        """
        B = encoder_latent.keys.shape[0]

        # Get or use provided GS tokens (without time conditioning)
        if gs_tokens is None:
            gs_tokens = self.get_gs_tokens(batch_size=B)

        # Apply time embedding to GS tokens
        gs_tokens = self._apply_time_embedding_to_gs_tokens(gs_tokens, decoder_input)

        # Run decoder with precomputed keys/values
        for layer in self.enc_dec_backbone.decoder_blocks:
            gs_tokens = layer(
                gs_tokens=gs_tokens,
                keys=encoder_latent.keys,
                values=encoder_latent.values,
            )

        gs_tokens = self.backbone_norm(gs_tokens)  # B, num_gs_tokens, C

        if self.opt.safe_mode:
            # Clamp GS tokens for numerical stability with bf16
            gs_tokens = torch.clamp(gs_tokens, min=-50, max=50)

        # Convert to Gaussians
        gaussians = self.activation_head(gs_tokens)

        # give gaussians an offset to the z axis so it is visible when initialized
        gaussians[..., 2] = gaussians[..., 2] + self.opt.gaussian_z_offset

        return gaussians

    def forward_gaussians(self, model_input: ModelInput) -> torch.Tensor:
        """
        Generate Gaussians from model input.
        This is a convenience method that combines encoding and decoding.
        For test-time training, use forward_encoder(), get_gs_tokens(), and forward_decoder() separately.

        Args:
            model_input: ModelInput containing all necessary input data

        Returns:
            Gaussians tensor [B, N, 14] where N is the number of Gaussians
        """
        # Encode input views
        encoder_latent = self.forward_encoder(model_input.encoder)

        # Decode to Gaussians (time conditioning is applied inside forward_decoder)
        gaussians = self.forward_decoder(encoder_latent, model_input.decoder)

        return gaussians

    def render_gaussians(self, gaussians: torch.Tensor, decoder_input: ModelInputDecoder, output_size=None) -> dict:
        """
        Render Gaussians to images.

        Args:
            gaussians: Gaussian parameters [B, N, 14]
            decoder_input: ModelInputDecoder containing camera parameters
            output_size: Optional (H, W) to override render resolution (e.g. (512, 512)).

        Returns:
            Dictionary with 'images_pred', 'alphas_pred', etc.
        """
        # background color
        if self.opt.bg_color == "white":
            bg_color = torch.ones(3, dtype=gaussians.dtype, device=gaussians.device)
        elif self.opt.bg_color == "black":
            bg_color = torch.zeros(3, dtype=gaussians.dtype, device=gaussians.device)
        elif self.opt.bg_color == "grey":
            bg_color = torch.ones(3, dtype=gaussians.dtype, device=gaussians.device) * 0.5
        else:
            raise ValueError(f"Invalid background color: {self.opt.bg_color}")

        results = self.gs.render(
            gaussians,
            decoder_input.cam_view,
            bg_color=bg_color,
            intrinsics=decoder_input.intrinsics,
            output_size=output_size,
        )
        return results

    def compute_loss(
        self, gaussians: torch.Tensor, decoder_input: ModelInputDecoder, supervision: ModelSupervision
    ) -> dict:
        """
        Compute loss given gaussians and supervision.

        Args:
            gaussians: Gaussian parameters [B, N, 14]
            decoder_input: ModelInputDecoder containing camera parameters
            supervision: ModelSupervision containing ground truth data

        Returns:
            Dictionary containing 'loss', 'psnr', and other loss/metric components
        """
        # Render gaussians
        render_results = self.render_gaussians(gaussians, decoder_input)

        pred_images = render_results["images_pred"]
        pred_masks = render_results["alphas_pred"]
        means2d_pred = render_results["means2d_pred"]
        depths_pred = render_results["depths_pred"]
        gt_images = supervision.images_output

        # add a fake batch dimension on 1 (to minimize code changes in _compute_loss_from_renders) and vmap to compute loss per scene
        if supervision.rays_os is None:
            in_dims = (0,) * 7 + ((None,) * 2)
            rays_os = supervision.rays_os
            rays_ds = supervision.rays_ds
        else:
            in_dims = (0,) * 9
            rays_os = supervision.rays_os.unsqueeze(1)
            rays_ds = supervision.rays_ds.unsqueeze(1)

        results_per_scene = torch.vmap(self._compute_loss_from_renders, in_dims=in_dims)(
            pred_images.unsqueeze(1),
            pred_masks.unsqueeze(1),
            means2d_pred.unsqueeze(1),
            supervision.images_output.unsqueeze(1),
            supervision.masks_output.unsqueeze(1),
            supervision.has_mask.unsqueeze(1),
            gaussians[..., :3].unsqueeze(1),
            rays_os,
            rays_ds,
        )

        if "loss_mask" in results_per_scene:
            # normalize mask loss by the number of scenes with masks
            results_per_scene["loss_mask"] = results_per_scene["loss_mask"] / (
                supervision.has_mask.float().sum() + 1e-6
            )

        result_means = {k: v.mean() for k, v in results_per_scene.items()}
        results_per_scene = {f"{k}_per_scene": v for k, v in results_per_scene.items()}

        # LPIPS loss (computed outside vmap since it's a neural network module)
        # Use normalize=True so [0, 1] images are converted internally (matches metrics.py)
        if self.opt.lambda_lpips > 0:
            gt_flat = gt_images.view(-1, 3, self.img_size[0], self.img_size[1])
            pred_flat = pred_images.view(-1, 3, self.img_size[0], self.img_size[1])
            gt_flat = F.interpolate(gt_flat, size=(256, 256), mode="bilinear", align_corners=False)
            pred_flat = F.interpolate(pred_flat, size=(256, 256), mode="bilinear", align_corners=False)
            loss_lpips = self.lpips_loss(gt_flat, pred_flat, normalize=True).mean()
            result_means["loss_lpips"] = loss_lpips
            # Add LPIPS to the total loss
            result_means["loss"] = result_means["loss"] + self.opt.lambda_lpips * loss_lpips

        results = {
            **result_means,
            **results_per_scene,
            # add rendered outputs for visualization
            "images_pred": pred_images,
            "alphas_pred": pred_masks,
            "images_output": gt_images,
            "depths_pred": depths_pred,
        }

        return results

    def _compute_loss_from_renders(
        self,
        pred_images: torch.Tensor,
        pred_masks: torch.Tensor,
        means2d_pred: torch.Tensor,
        gt_images: torch.Tensor,
        gt_masks: torch.Tensor,
        has_mask: torch.Tensor,
        means3d_pred: torch.Tensor,
        rays_os: torch.Tensor,
        rays_ds: torch.Tensor,
    ) -> dict:
        """Computes loss from already rendered images and masks.

        Args:
            pred_images: Predicted images [B, V, C, img_size, img_size]
            pred_masks: Predicted masks [B, V, 1, img_size, img_size]
            means2d_pred: Predicted means2d [B, V, N, 2]
            gt_images: Ground truth images [B, V, 3, img_size, img_size]
            gt_masks: Ground truth masks [B, V, 1, img_size, img_size]
            has_mask: Boolean mask indicating if a given scene has masks [B]
            rays_os: Ray origins [B, V, 3, H, W]
            rays_ds: Ray directions [B, V, 3, H, W]
        """
        results = {}
        assert has_mask.shape == (pred_images.shape[0],), "has_mask must have the same batch size as pred_images"

        # Prepare ground truth
        bg_color = torch.ones(3, dtype=pred_images.dtype, device=pred_images.device)
        gt_images = gt_images * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)

        # MSE loss
        loss_mse = F.mse_loss(pred_images, gt_images)
        loss = loss_mse
        results["loss_mse"] = loss_mse

        # Mask loss
        if self.opt.lambda_mask > 0:
            loss_mse_mask = F.mse_loss(pred_masks, gt_masks, reduction="none").mean(dim=(1, 2, 3, 4))
            loss_mse_mask = torch.where(has_mask, loss_mse_mask, torch.zeros_like(loss_mse_mask)).sum()
            results["loss_mask"] = loss_mse_mask
            loss = loss + self.opt.lambda_mask * loss_mse_mask

        # Note: LPIPS loss is computed outside vmap in compute_loss()
        # since it's a neural network module that cannot be vmapped

        # SSIM loss
        if self.opt.lambda_ssim > 0:
            from ..fused_ssim import fused_ssim

            ssim = fused_ssim(
                pred_images.view(-1, 3, self.img_size[0], self.img_size[1]),
                gt_images.view(-1, 3, self.img_size[0], self.img_size[1]),
            )
            loss_ssim = (1 - ssim) / 2
            loss = loss + self.opt.lambda_ssim * loss_ssim
            results["loss_ssim"] = loss_ssim

        # Visibility loss
        if self.opt.lambda_visibility > 0:
            uv = torch.nan_to_num(means2d_pred, nan=0.0, posinf=1e6, neginf=-1e6)  # [B, V, N, 2]

            uv_norm = (uv / torch.tensor([self.img_size[1], self.img_size[0]], device=uv.device)) * 2 - 1
            out_of_bounds = F.relu(torch.abs(uv_norm) - self.opt.vis_bound)  # [B, V, N, 2]
            out_of_bounds = out_of_bounds.sum(-1)  # [B, V, N]

            vis_loss = out_of_bounds.min(dim=1).values
            if self.opt.visibility_distance_threshold > 0:
                vis_loss = vis_loss.clamp(max=self.opt.visibility_distance_threshold)
            vis_loss = vis_loss.mean()
            loss = loss + vis_loss * self.opt.lambda_visibility
            results["loss_visibility"] = vis_loss

        results["loss"] = loss

        # PSNR metric
        with torch.no_grad():
            psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2, dim=(-1, -2, -3)))
            results["psnr"] = psnr.mean()

        return results

    def forward(self, data, skip_loss=False):
        """
        Forward pass of the TokenGS model.

        Args:
            data: Dictionary from dataloader
            skip_loss: If True, skip loss computation

        Returns:
            Dictionary containing results including 'loss', 'gaussians', 'images_pred', etc.
        """
        # Split data into structured input and supervision
        model_input, supervision = split_data(data, self.opt)

        # Generate Gaussians from input
        if self.opt.use_ttt_for_eval:
            gaussians = self.forward_ttt(
                model_input, supervision, n_steps=self.opt.ttt_n_steps, lr=self.opt.ttt_lr, method=self.opt.ttt_method
            )  # [B, N, 14]
        else:
            gaussians = self.forward_gaussians(model_input)  # [B, N, 14]

        # Compute loss or just render
        if skip_loss:
            results = self.render_gaussians(gaussians, model_input.decoder)
            results["gaussians"] = gaussians
            return results

        # Compute loss
        results = self.compute_loss(gaussians, model_input.decoder, supervision)
        results["gaussians"] = gaussians
        results["images_output"] = supervision.images_output
        return results

    def forward_ttt(
        self,
        model_input: ModelInput,
        supervision: ModelSupervision,
        n_steps: int = 10,
        lr: float = 1e-3,
        method: Literal["tokens", "gaussians"] = "tokens",
    ) -> torch.Tensor:
        """
        Forward pass of the TokenGS model using test-time training for `n_steps`.

        Args:
            model_input: ModelInput containing encoder and decoder inputs (same signature as forward_gaussians)
            n_steps: Number of test-time training steps
            lr: Learning rate for the optimizer
            method: Method to use for test-time training ("tokens" or "gaussians")
        """
        if method == "tokens":
            return self.forward_ttt_tokens(model_input, supervision, n_steps, lr)
        elif method == "gaussians":
            return self.forward_ttt_gaussians(model_input, supervision, n_steps, lr)
        else:
            raise ValueError(f"Invalid method: {method}")

    def forward_ttt_tokens(
        self, model_input: ModelInput, supervision: ModelSupervision, n_steps: int = 10, lr: float = 1e-3
    ) -> torch.Tensor:
        """
        Forward pass of the TokenGS model using test-time training of the Gaussian tokens for `n_steps`.

        This method optimizes the Gaussian tokens by rendering to input views and
        comparing against the input view images as supervision.

        Args:
            model_input: ModelInput containing encoder and decoder inputs (same signature as forward_gaussians)
            n_steps: Number of test-time training steps
            lr: Learning rate for the optimizer

        Returns:
            Gaussians tensor [B, N, 14] where N is the number of Gaussians
        """
        # Convert to TTT inputs (render to input views, supervise with input images)
        if self.opt.ttt_supervision_mode == "input_only":
            model_input_ttt, supervision_ttt = model_input.to_ttt(masks_output=None, has_mask=False)
        elif self.opt.ttt_supervision_mode == "cond_only":
            model_input_ttt = model_input
            supervision_ttt = supervision
        elif self.opt.ttt_supervision_mode == "both":
            model_input_ttt, supervision_ttt = model_input.to_ttt(masks_output=None, has_mask=True)
            n_input = model_input_ttt.decoder.cam_view.shape[1]
            n_cond = model_input.decoder.cam_view.shape[1]
            model_input_ttt.decoder = ModelInputDecoder.merge(
                [model_input_ttt.decoder] + [model_input.decoder] * int(n_input / n_cond * 2)
            )
            supervision_ttt = ModelSupervision.merge([supervision_ttt] + [supervision] * int(n_input / n_cond * 2))
        else:
            raise ValueError(f"Invalid TTT supervision mode: {self.opt.ttt_supervision_mode}")

        with freeze_model_parameters(self):
            return self._forward_ttt_tokens(model_input_ttt, supervision_ttt, n_steps, lr)

    @torch.no_grad()
    def _forward_ttt_tokens(
        self,
        model_input: ModelInput,
        supervision: ModelSupervision,
        n_steps: int = 10,
        lr: float = 1e-3,
    ) -> torch.Tensor:
        """Inner implementation of forward_ttt_tokens, assumes the model is frozen."""
        encoder_latent = self.forward_encoder(model_input.encoder)

        with torch.set_grad_enabled(True):
            gs_tokens = self.get_gs_tokens(batch_size=model_input.batch_size).detach().requires_grad_(True)
            optim = torch.optim.Adam([gs_tokens], lr=lr)
            for i in range(n_steps):
                optim.zero_grad()
                gaussians = self.forward_decoder(encoder_latent, model_input.decoder, gs_tokens=gs_tokens)
                loss = self.compute_loss(gaussians, model_input.decoder, supervision)["loss"]
                loss.backward()
                optim.step()

        # compute the final result
        return self.forward_decoder(encoder_latent, model_input.decoder, gs_tokens=gs_tokens.detach())

    @torch.no_grad()
    def forward_ttt_gaussians(self, model_input: ModelInput, n_steps: int = 10, lr: float = 1e-3) -> torch.Tensor:
        """
        Forward pass of the TokenGS model using test-time training of the Gaussian parameters for `n_steps`.

        Compared to forward_ttt_tokens, this method only optimizes the Gaussian parameters, not the Gaussian tokens.
        Uses DirectClipHead to optimize in unconstrained space and project back to valid Gaussian manifold.

        Args:
            model_input: ModelInput containing encoder and decoder inputs (same signature as forward_gaussians)
            n_steps: Number of test-time training steps
            lr: Learning rate for the optimizer

        Returns:
            Gaussians tensor [B, N, 14] where N is the number of Gaussians
        """
        # Assert we're using ClipActivationHead
        assert isinstance(self.activation_head, ClipActivationHead), (
            f"forward_ttt_gaussians requires ClipActivationHead, got {type(self.activation_head)}"
        )

        # Convert to TTT inputs (render to input views, supervise with input images)
        model_input_ttt, supervision_ttt = model_input.to_ttt()

        # Forward to get gaussians
        gaussians = self.forward_gaussians(model_input_ttt)

        # Create a DirectClipHead instance
        direct_head = DirectClipHead(self.opt).to(gaussians.device)

        # Pseudo-invert the gaussians to obtain unconstrained parameter tensor
        params = direct_head.forward_inverse(gaussians)

        with torch.set_grad_enabled(True):
            params.requires_grad_(True)

            # Optimize the param tensor, projecting back to valid manifold at each step
            optim = torch.optim.Adam([params], lr=lr)
            for _ in range(n_steps):
                optim.zero_grad()
                # Project params back to valid Gaussian manifold
                gaussians_constrained = direct_head(params)
                loss = self.compute_loss(gaussians_constrained, model_input_ttt.decoder, supervision_ttt)["loss"]
                loss.backward()
                optim.step()

            # Final projection to get optimized gaussians
            gaussians_optimized = direct_head(params)

        return gaussians_optimized.detach()
