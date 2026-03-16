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
Sprint (distilled) pipeline for multiview image generation.

This pipeline performs fast 2-step inference using a distilled multiview Sana model.
Key differences from the standard SparseViewDiTPipeline:
- No classifier-free guidance (CFG is distilled into the model)
- Only 2 denoising steps (vs 20-30)
- TrigFlow parameterization with SCM (Shortcut Consistency Model)
- guidance_scale is passed as a conditioning signal to the model, not used for CFG
- Uses DPMSolverMultistepScheduler with sigma_data scaling
"""

from typing import Any

import torch
import tqdm.auto as tqdm
from diffusers.utils import logging
from PIL import Image

from multiview_diffusion.pipelines.sparseviewdit_pipeline import SparseViewDiTPipeline, retrieve_timesteps
from multiview_diffusion.schedulers import TrigFlowScheduler

logger = logging.get_logger(__name__)


class SparseViewDiTSprintPipeline(SparseViewDiTPipeline):
    """Fast 2-step inference pipeline for distilled multiview Sana models.

    Inherits all data processing, c-radio encoding, and VAE decode from
    :class:`SanaMultiViewPipeline`.  Only the denoising loop differs:

    * No CFG — ``guidance_scale`` is passed as a conditioning embedding.
    * 2-step SCM denoising with TrigFlow parameterization.
    """

    def __init__(
        self,
        vae,
        transformer,
        scheduler: TrigFlowScheduler,
        image_encoder=None,
        image_processor=None,
    ):
        super().__init__(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            image_encoder=image_encoder,
            image_processor=image_processor,
        )

    @torch.no_grad()
    def __call__(
        self,
        data_dict,
        *,
        conditioning_images: list[Image.Image] | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        num_inference_steps: int = 2,
        max_timesteps: float = 1.57080,  # pi/2
        intermediate_timesteps: float = 1.3,
        guidance_scale: float = 4.5,
        generator: torch.Generator | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
        flow_shift: float = 1.0,  # Not used in Sprint pipeline
    ) -> dict[str, Any] | tuple:
        """Fast 2-step multiview generation (no CFG).

        Args:
            data_dict: **Required.** Preprocessed multiview AttrDict.
            guidance_scale: Embedded into the model as a conditioning signal
                (not used for classifier-free guidance).
            num_inference_steps: Number of SCM steps (default 2).
            max_timesteps / intermediate_timesteps: SCM scheduler params.
        """
        device = next(self.transformer.parameters()).device
        model_dtype = next(self.transformer.parameters()).dtype

        # --- 1. Prepare multiview inputs (no CFG) ---
        t_prepare_start = torch.cuda.Event(enable_timing=True)
        t_prepare_end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t_prepare_start.record()
        mv_inputs = self.prepare_multiview_inputs(
            data_dict=data_dict,
            device=device,
            dtype=model_dtype,
            do_classifier_free_guidance=False,
        )
        t_prepare_end.record()
        torch.cuda.synchronize()
        prepare_ms = t_prepare_start.elapsed_time(t_prepare_end)
        print(f"   Prepare multiview inputs: {prepare_ms / 1000:.2f}s")

        num_views = len(data_dict.x)
        num_target_views = data_dict.n_target

        # --- 2. Encode prompt (c-radio, single pass) ---
        t_extract_start = torch.cuda.Event(enable_timing=True)
        t_extract_end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t_extract_start.record()
        if prompt_embeds is None:
            if conditioning_images is None:
                conditioning_images = self._extract_conditioning_images(
                    data_dict,
                    num_target_views,
                    num_views,
                )
            prompt_embeds, prompt_attention_mask, _, _ = self.encode_image_prompt(
                conditioning_images,
                device,
                do_classifier_free_guidance=False,
            )
        t_extract_end.record()
        torch.cuda.synchronize()
        extract_ms = t_extract_start.elapsed_time(t_extract_end)
        print(f"   Extract conditioning images: {extract_ms / 1000:.2f}s")

        # --- 3. Build model kwargs (no CFG doubling) ---
        mv_kwargs = self._build_multiview_kwargs(
            mv_inputs,
            do_classifier_free_guidance=False,
        )

        # --- 4. Initial latents ---
        try:
            sigma_data = self.scheduler.config.sigma_data
        except:
            sigma_data = 0.5
            logger.warning("sigma_data not set in scheduler config, using default value of 0.5")
        latents = torch.randn_like(mv_inputs["clean_images"]) * sigma_data

        # --- 5. Timesteps ---
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            max_timesteps=max_timesteps,
            intermediate_timesteps=intermediate_timesteps,
        )
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(0)

        # --- 6. Guidance embedding ---
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0]).to(model_dtype)
        if hasattr(self.transformer, "config") and hasattr(self.transformer.config, "guidance_embeds_scale"):
            guidance = guidance * self.transformer.config.guidance_embeds_scale
        
        # --- 7. Denoising loop (TrigFlow + SCM) ---
        denoised = None
        t_loop_start = torch.cuda.Event(enable_timing=True)
        t_loop_end = torch.cuda.Event(enable_timing=True)
        with torch.autocast("cuda", dtype=model_dtype):
            t_loop_start.record()
            for i, t in tqdm.tqdm(enumerate(timesteps[:-1]), total=len(timesteps[:-1]), desc="Denoising"):
                timestep = t.expand(latents.shape[0])
                latents_model_input = latents / sigma_data

                scm_t = torch.sin(timestep) / (torch.cos(timestep) + torch.sin(timestep))
                scm_t_exp = scm_t.view(-1, 1, 1, 1)

                latent_model_input = latents_model_input * torch.sqrt(scm_t_exp**2 + (1 - scm_t_exp) ** 2)
                scm_t = scm_t * 1000.0

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    timestep=scm_t,
                    guidance=guidance,
                    return_dict=False,
                    **mv_kwargs,
                )[0]

                # Inverse TrigFlow transformation in float32 for numerical stability
                # with torch.autocast("cuda", enabled=False):
                noise_pred = (
                    (1 - 2 * scm_t_exp) * latent_model_input.float()
                    + (1 - 2 * scm_t_exp + 2 * scm_t_exp**2) * noise_pred.float()
                ) / torch.sqrt(scm_t_exp**2 + (1 - scm_t_exp) ** 2)
                noise_pred = noise_pred * sigma_data

                if isinstance(self.scheduler, TrigFlowScheduler):
                    latents = self.scheduler.step(
                        model_output=noise_pred, timeindex=i, timestep=t, return_dict=False, sample=latents
                    )[0]
                else:
                    latents, denoised = self.scheduler.step(
                        noise_pred,
                        timestep,
                        latents,
                        return_dict=False,
                    )

            t_loop_end.record()
            torch.cuda.synchronize()
            loop_ms = t_loop_start.elapsed_time(t_loop_end)
            n_steps = len(timesteps) - 1
            print(
                f"   Denoising loop: {loop_ms / 1000:.2f}s "
                f"({loop_ms / n_steps:.1f}ms/step, {n_steps} steps)"
            )

            # --- 8. Decode ---
            if denoised is None:
                denoised = latents
            target_latents = (denoised / sigma_data)[:num_target_views]

            if output_type == "latent":
                if return_dict:
                    return {"images": target_latents}
                return (target_latents,)

            t_decode_start = torch.cuda.Event(enable_timing=True)
            t_decode_end = torch.cuda.Event(enable_timing=True)
            t_decode_start.record()
            images = self.decode_latents(target_latents, output_type=output_type)
            t_decode_end.record()
            torch.cuda.synchronize()
            decode_ms = t_decode_start.elapsed_time(t_decode_end)
            print(f"   VAE decode: {decode_ms / 1000:.2f}s")

        if return_dict:
            return {"images": images}
        return (images,)
