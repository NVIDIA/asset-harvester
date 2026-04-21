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

import copy
from dataclasses import dataclass
from typing import Any, Literal

import tyro


@dataclass
class Options:
    ### wandb
    use_wandb: bool = False
    experiment_name: str = "tokengs"
    out_dir = "outputs"
    project_name = "4D Scene Reconstruction"
    debug: bool = False
    evaluating: bool = False
    deferred_bp: bool = False
    safe_mode: bool = False
    use_layerscale: bool = True
    # Safe mode options
    safe_mode_loss_threshold: float = 1.0  # Skip batch if loss exceeds this value
    safe_mode_param_threshold: float = 1e4  # Warn if parameter exceeds this value
    grad_norm_cap: float = 5000
    # number of test samples
    num_test_samples: int = 1000000

    ### model
    # image size
    img_size: tuple[int, int] = (256, 256)
    # gaussian scale cap; maximum allowed value for the predicted gaussian scales
    gaussian_scale_cap: float = 0.3
    # background color
    bg_color: Literal["white", "black", "grey"] = "grey"

    # encoder
    enc_depth: int = 24
    enc_embed_dim: int = 1024
    enc_num_heads: int = 16
    mlp_ratio: int = 4
    patch_size: int = 8

    ### dataset
    data_mode: tuple[tuple[str, int], ...] = (("re10k", 1),)
    # camera near plane
    znear: float = 0.1
    # camera far plane
    zfar: float = 500
    # number of all views (input + output)
    num_views: int = 8
    # number of views
    num_input_views: int = 4
    camera_normalization_method: Literal["objaverse", "mean_cam", "first_cam"] = "first_cam"
    camera_scale_method: Literal["constant", "distance", "bound"] = "constant"
    dnear: float = 0.1
    dfar: float = 500
    # num workers
    num_workers: int = 16
    seed: int = 42
    use_objaverse_sampling: bool = False

    ### training
    # workspace
    workspace: str = "./workspace"
    # resume
    resume: str | None = None
    # max training time in seconds
    max_training_time_seconds: int = (4 * 60 - 10) * 60  # 4 hours (for ORD) minus 10 minutes for safety
    # batch size (per-GPU)
    batch_size: int = 8
    # gradient accumulation
    gradient_accumulation_steps: int = 1
    # training epochs
    num_epochs: int = 30
    # lpips loss weight
    lambda_lpips: float = 0.5
    # mask loss weight
    lambda_mask: float = 0.0
    # ssim loss weight
    lambda_ssim: float = 0.0
    # visibility loss weight
    lambda_visibility: float = 0.0
    # visibility distance threshold
    visibility_distance_threshold: float = 1
    # visibility bound
    vis_bound: float = 1.0
    # gradient clip
    gradient_clip: float = 1.0
    # mixed precision
    mixed_precision: str = "bf16"
    # learning rate (it is later multiplied by the batch_size)
    lr: float = 4e-4
    # gs token learning rate multiplier (gs_token_lr = lr * gs_token_lr_multiplier)
    gs_token_lr_multiplier: float = 1.0
    # learning rate scheduler type
    lr_scheduler: Literal["onecycle", "constant"] = "onecycle"
    # final div factor
    final_div_factor: float = 1000
    # pct start steps
    pct_start_steps: int = 1000
    # exclude weight decay for 1D parameters (biases, layernorm)
    no_weight_decay_for_1d_params: bool = True

    ### testing
    # test image path
    test_path: str | None = None
    # number of batches to dump media files (videos, PLY) for during evaluation
    eval_n_media_dumps: int = 0

    ### model
    model_type: str = "tokengs"

    max_iters_per_epoch: int = 1000000

    # disable bias term in LayerNorm
    layernorm_no_bias: bool = False

    ### dynamic
    use_interp_target: bool = False
    # use time embedding
    # Time embedding is used for dynamic reconstruction; True by default.
    time_embedding: bool = True
    time_embedding_dim: int = 2

    skip_eval: bool = False

    ### logging
    # save videos instead of horizontally stacked images
    log_videos: bool = False
    # fps for video logging
    log_video_fps: int = 4

    ### TokenGS
    # deconv patch size d will make the actual unpatchify output dxd
    deconv_patch_size: int = 8
    # init_deconv to small values
    init_deconv: bool = True
    # initialize time embedding to zero
    zero_init_time_embed: bool = True
    # use layernorm in the embedding projection
    use_emb_norm: bool = True
    # use layernorm in the encoder output
    use_enc_norm: bool = False
    # activation head type
    activation_head_type: Literal["clip", "objaverse"] = "clip"
    # scale shift
    scale_shift: Literal["default"] | float = "default"
    # use input supervision
    use_input_supervision: bool = True
    # gaussian z offset
    # this is used for the xyz model to make the gaussians visible when initialized
    gaussian_z_offset: float = 1.0
    # gs token args (always learnable)
    num_gs_tokens: int = 1024
    token_dim: int = 1024
    num_dynamic_gs_tokens: int = 0
    init_dynamic_tokens_from_static: bool = False
    init_tokens_from_existing: bool = False
    gs_token_std: float = 1e-2
    use_causal_mask: bool = False
    # other args
    opacity_min: float = 0.0
    scale_min: float = 0.0
    # decoder depth, if none, use the same as encoder depth
    dec_depth: int | None = None
    decoder_attn_order: Literal["self_cross", "cross_self"] = "cross_self"
    use_ttt_for_eval: bool = False
    ttt_n_steps: int = 10
    ttt_lr: float = 1e-5
    ttt_method: Literal["tokens", "gaussians"] = "tokens"
    ttt_supervision_mode: Literal["input_only", "cond_only", "both"] = "input_only"

    ### augmentation

    # augmentation prob for grid distortion
    prob_grid_distortion: float = 0.0
    # augmentation prob for camera jitter
    prob_cam_jitter: float = 0.0
    # grid distortion strength
    grid_distortion_strength: float = 0.0
    # random reflect
    random_reflect: bool = False
    # random scale
    random_scale: bool = False
    # scale range
    scale_range: tuple[float, float] = (0.5, 1.5)
    # camera augmentation
    camera_augmentation: bool = False
    # test with camera augmentation
    test_with_camera_augmentation: bool = False
    # camera augmentation params
    rot_deg_range: tuple[float, float] = (0, 3)
    trans_range: tuple[float, float] = (0, 0.03)
    intrin_range: tuple[float, float] = (0, 0.08)
    camera_augmentation_probability: float = 0.5

    ### dataset registry kwargs override
    dataset_kwargs: dict[str, str] | None = None

    def __post_init__(self):
        pass

    def evolve(self, **changes: Any) -> "Options":
        """Create a deep copy with specified fields updated."""
        new_instance = copy.deepcopy(self)
        for key, value in changes.items():
            if not hasattr(new_instance, key):
                raise AttributeError(f"Options has no attribute '{key}'")
            setattr(new_instance, key, value)
        return new_instance


# all the default settings
config_defaults: dict[str, Options] = {}
config_doc: dict[str, str] = {}

config_doc["default"] = "Default options (base Options with no preset overrides)."
config_defaults["default"] = Options()

config_doc["tokengs"] = "TokenGS base model with encoder-decoder architecture"
config_defaults["tokengs"] = Options(
    time_embedding=False,
    data_mode=(("assetharvest", 1),),
    znear=0.025,
    zfar=125,
    gaussian_scale_cap=0.075,
    img_size=(512, 512),
    num_gs_tokens=4096,
    activation_head_type="objaverse",
    camera_normalization_method="objaverse",
    gaussian_z_offset=0.0,
    bg_color="white",
    lambda_lpips=5.0,
    lambda_ssim=0.0,
    lambda_visibility=0.0,
    lambda_mask=0.0,
    num_input_views=16,
    num_views=20,
    use_input_supervision=False,
    enc_depth=24,
    dec_depth=1,
)


AllConfigs = tyro.extras.subcommand_type_from_defaults(config_defaults, config_doc)
