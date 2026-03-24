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

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class BaseConfig:
    def get(self, attribute_name, default=None):
        return getattr(self, attribute_name, default)

    def pop(self, attribute_name, default=None):
        if hasattr(self, attribute_name):
            value = getattr(self, attribute_name)
            delattr(self, attribute_name)
            return value
        else:
            return default

    def __str__(self):
        return json.dumps(asdict(self), indent=4)


@dataclass
class NREDataConfig(BaseConfig):
    # Data source configs
    use_relative_coords: bool = True  # make cameras relative to the first camera
    plucker_scene_scale: float = 30.0  # a normalization factor to make plucker embeddings close to -1,1 range
    mask_out_background_target: str | None = "white"
    mask_out_background_cond: str | None = False
    get_cam_from_skeleton: bool = False
    load_3d_skeleton: bool = False


@dataclass
class DataConfig(BaseConfig):
    nre_data_config: NREDataConfig | None = None


@dataclass
class ConditionerConfig(BaseConfig):
    augment_cond: bool = True
    augment_sigma_sample_p_mean: float = -3.0  # Mean of the augment sigma
    augment_sigma_sample_p_std: float = 1.0  # Std of the augment sigma
    augment_sigma_sample_multiplier: float = 4.0  # Multipler of augment sigma
    ray_embedding_func: str = "downsample"  # either "vae" or "downsample"
    concat_ray_embedding: bool = False
    concat_cond_mask: bool = False


@dataclass
class ModelConfig(BaseConfig):
    model: str = "SanaMS_600M_P1_D28"
    image_size: int = 512
    in_channels: int = 32
    out_channels: int = 32
    mixed_precision: str = "fp16"  # ['fp16', 'fp32', 'bf16']
    fp32_attention: bool = True
    load_from: str | None = None
    resume_from: dict[str, Any] | None = field(
        default_factory=lambda: {
            "checkpoint": None,
            "load_ema": False,
            "resume_lr_scheduler": True,
            "resume_optimizer": True,
        }
    )
    aspect_ratio_type: str = "ASPECT_RATIO_1024"
    multi_scale: bool = True
    pe_interpolation: float = 1.0
    micro_condition: bool = False
    attn_type: str = "linear"
    autocast_linear_attn: bool = False
    ffn_type: str = "glumbconv"
    mlp_acts: list[str | None] = field(default_factory=lambda: ["silu", "silu", None])
    mlp_ratio: float = 2.5
    use_pe: bool = False
    pos_embed_type: str = "sincos"
    pe_list: list[int] | None = None
    qk_norm: bool = False
    class_dropout_prob: float = 0.0
    linear_head_dim: int = 32
    cross_norm: bool = False
    cfg_scale: int = 4
    cond_on_rays: bool = False
    cond_on_mask: bool = False
    camera_emb: bool = False
    camera_emb_dim: int = 17
    aug_emb: bool = False
    aug_emb_dim: int = 1
    brightness_emb: bool = False
    skeletal_emb: bool = False
    skeletal_emb_dim: int = 63
    guidance_type: str = "classifier-free"
    pag_applied_layers: list[int] = field(default_factory=lambda: [14])
    _base_: list[str] = field(default_factory=lambda: [])
    extra: Any = None


@dataclass
class AEConfig(BaseConfig):
    vae_type: str = "dc-ae"
    vae_pretrained: str = "mit-han-lab/dc-ae-f32c32-sana-1.0"
    scale_factor: float = 0.41407
    vae_latent_dim: int = 32
    vae_downsample_rate: int = 32
    sample_posterior: bool = True
    weight_dtype: str = "float32"
    extra: Any = None


@dataclass
class TextEncoderConfig(BaseConfig):
    text_encoder_name: str = "gemma-2-2b-it"
    augment_cradio_brightness: bool = False
    augment_cradio_brightness_scale: float = 0.5
    caption_channels: int = 2304
    y_norm: bool = True
    y_norm_scale_factor: float = 1.0
    model_max_length: int = 300
    chi_prompt: list[str | None] = field(default_factory=lambda: [])
    extra: Any = None


@dataclass
class SchedulerConfig(BaseConfig):
    train_sampling_steps: int = 1000
    predict_v: bool = True
    noise_schedule: str = "linear_flow"
    pred_sigma: bool = False
    vis_sampler: str = "flow_dpm-solver"
    flow_shift: float = 1.0
    # logit-normal timestep
    weighting_scheme: str | None = "logit_normal"
    logit_mean: float = 0.0
    logit_std: float = 1.0
    extra: Any = None


@dataclass
class SanaConfig(BaseConfig):
    data: DataConfig
    model: ModelConfig
    cond: ConditionerConfig
    vae: AEConfig
    text_encoder: TextEncoderConfig
    scheduler: SchedulerConfig
    work_dir: str = "output/"
    resume_from: str | None = None
    load_from: str | None = None
    debug: bool = False
    caching: bool = False
    report_to: str = "wandb"
    tracker_project_name: str = "sana-baseline"
    name: str = "baseline"
    loss_report_name: str = "loss"
    slurm_name: str = "nvr_elm_llm"
    _base_: list[str | None] = field(default_factory=list)
    re_init_cam_embed: bool = False


def model_init_config(config: SanaConfig, latent_size: int = 32):

    pred_sigma = getattr(config.scheduler, "pred_sigma", True)
    learn_sigma = getattr(config.scheduler, "learn_sigma", True) and pred_sigma
    return {
        "input_size": latent_size,
        "pe_interpolation": config.model.pe_interpolation,
        "config": config,
        "model_max_length": config.text_encoder.model_max_length,
        "qk_norm": config.model.qk_norm,
        "micro_condition": config.model.micro_condition,
        "caption_channels": config.text_encoder.caption_channels,
        "y_norm": config.text_encoder.y_norm,
        "attn_type": config.model.attn_type,
        "ffn_type": config.model.ffn_type,
        "mlp_ratio": config.model.mlp_ratio,
        "mlp_acts": list(config.model.mlp_acts),
        "in_channels": config.vae.vae_latent_dim,
        "y_norm_scale_factor": config.text_encoder.y_norm_scale_factor,
        "use_pe": config.model.use_pe,
        "pos_embed_type": config.model.pos_embed_type,
        "linear_head_dim": config.model.linear_head_dim,
        "pred_sigma": pred_sigma,
        "learn_sigma": learn_sigma,
        "cross_norm": config.model.cross_norm,
        "pe_list": config.model.pe_list,
    }


@dataclass
class SanaInferenceConfig(SanaConfig):
    n_worker: int = 16
    worker_rank: int = 1
    ckpt_path: str = "/mnt/scratch/sana/epoch_1_step_12984.pth"
    load_ema: bool = False
    work_dir: str = "output/inference"
    cfg_scale: float = 2.0
    sampler: str = "flow_dpm-solver"
    seed: int = 0
    step: int = 30
    prompt: str = "A vehicle in natural condition, white background"
    data_path: str = "/mnt/scratch/nre_single_object_data/out_spec4-merged-0000-6000_cogvlm_segs_clip_filtered_good/"
    use_native_cross_attn: bool = True
    max_input_views: int = 4
    output_views: int = 16
    synthetic_data_brightness_aug: bool = False
    fov_aug_enabled: bool = False
    fov_aug_on_target: bool = False  # also apply fov augmentation to target views
    fov_aug_min: float = 0.7
    fov_aug_max: float = 1.5

    # Skeleton controlnet options
    use_skeleton_controlnet: bool = False
    skeleton_vis: bool = True  # Whether to save skeleton visualizations
    control_weight: float = 1.0
    run_lgm: bool = False
    lgm_ckpt_path: str | None = (
        "/lustre/fs12/portfolios/nvr/projects/nvr_torontoai_3dscenerecon/users/kangxuey/AH_v2_checkpoints/tokengs_jit.pt"
    )
