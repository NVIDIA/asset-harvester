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
Diffusers pipeline for multiview image generation using SparseViewDiTTransformer2DModelNative.
"""

import inspect
from typing import Any

import numpy as np
import torch
import tqdm.auto as tqdm
from diffusers import AutoencoderDC, DiffusionPipeline
from diffusers.image_processor import PixArtImageProcessor
from diffusers.models.autoencoders.autoencoder_dc import EncoderOutput
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.utils import logging
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor, Gemma2PreTrainedModel, GemmaTokenizer, GemmaTokenizerFast

# Import DPMS sampler and VAE utilities
# from ..schedulers.dpm_solver import DPMS
from ..models.sparseviewdit import SparseViewDiTTransformer2DModelNative

logger = logging.get_logger(__name__)


def retrieve_timesteps(
    scheduler,
    num_inference_steps=None,
    device=None,
    timesteps=None,
    sigmas=None,
    **kwargs,
):
    if timesteps is not None:
        accepts = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts:
            raise ValueError(f"{scheduler.__class__} does not support custom timesteps.")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accepts = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts:
            raise ValueError(f"{scheduler.__class__} does not support custom sigmas.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        print(f"Setting timesteps to {timesteps.tolist()}")
    return timesteps, num_inference_steps


class SparseViewDiTPipeline(DiffusionPipeline):
    """
    Pipeline for multiview image generation using SparseViewDiT models.

    This pipeline supports:
    - Generating multiple views of a scene/object
    - Camera-aware generation with pose and FOV conditioning
    - Plucker ray embeddings for 3D consistency
    - Partial conditioning (inpainting some views while generating others)
    - Classifier-free guidance

    Args:
        vae: VAE model for encoding/decoding images
        text_encoder: Text encoder (e.g., Gemma-2, c-radio)
        tokenizer: Tokenizer for text encoder
        transformer: SparseViewDiTTransformer2DModelNative
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderDC,
        transformer: SparseViewDiTTransformer2DModelNative,
        scheduler: DPMSolverMultistepScheduler | None = None,
        text_encoder: Gemma2PreTrainedModel | None = None,
        tokenizer: GemmaTokenizer | GemmaTokenizerFast | None = None,
        image_encoder: AutoModel | None = None,
        image_processor: CLIPImageProcessor | None = None,
    ):
        super().__init__()

        modules = dict(vae=vae, transformer=transformer)
        if text_encoder is not None:
            modules["text_encoder"] = text_encoder
        if tokenizer is not None:
            modules["tokenizer"] = tokenizer
        if scheduler is not None:
            modules["scheduler"] = scheduler
        else:
            modules["scheduler"] = scheduler
        self.register_modules(**modules)

        self.image_encoder = image_encoder
        self.cradio_image_processor = image_processor

        if hasattr(self, "vae") and self.vae is not None:
            if hasattr(self.vae, "config") and hasattr(self.vae.config, "encoder_block_out_channels"):
                self.vae_scale_factor = 2 ** (len(self.vae.config.encoder_block_out_channels) - 1)
            elif hasattr(self.vae, "cfg"):
                self.vae_scale_factor = 32
            else:
                self.vae_scale_factor = 32
        else:
            self.vae_scale_factor = 32

        self.image_processor = PixArtImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def encode_image_prompt(
        self,
        conditioning_images: list[Image.Image | torch.Tensor],
        device: torch.device,
        do_classifier_free_guidance: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode conditioning images using c-radio to create prompt embeddings.

        This is the default conditioning mode for SparseViewDiT.

        Args:
            conditioning_images: List of PIL Images or tensors in [-1, 1] range
            device: Device to place embeddings on
            do_classifier_free_guidance: Whether to create null embeddings for CFG

        Returns:
            Tuple of (prompt_embeds, prompt_attention_mask)
        """
        if self.image_encoder is None or self.cradio_image_processor is None:
            raise ValueError(
                "c-radio image encoder and processor must be provided for image conditioning. "
                "Pass image_encoder and image_processor when creating the pipeline."
            )

        # Convert tensors to PIL images if needed
        pil_images = []
        for img in conditioning_images:
            if isinstance(img, torch.Tensor):
                # Convert from [-1, 1] to [0, 255]
                if img.dim() == 4:  # [B, C, H, W]
                    img = img[0]  # Take first in batch
                img_np = ((img + 1.0) / 2.0 * 255).clamp(0, 255)
                img_np = img_np.permute(1, 2, 0).to(torch.uint8).cpu().numpy()
                pil_img = Image.fromarray(img_np)
                pil_images.append(pil_img)
            else:
                pil_images.append(img)

        # If no conditioning images, create a grey placeholder
        if len(pil_images) == 0:
            grey_img = Image.fromarray(np.ones((512, 512, 3), dtype=np.uint8) * 128)
            pil_images = [grey_img]

        # Process with c-radio
        with torch.no_grad():
            pixel_values = self.cradio_image_processor(
                images=pil_images, return_tensors="pt", do_resize=True
            ).pixel_values
            pixel_values = pixel_values.to(device)

            summary, features = self.image_encoder(pixel_values)

            # Max pool along first dimension (aggregate multiple conditioning views)
            features = torch.amax(features, dim=0, keepdim=True)

        # Format for model
        prompt_embeds = features  # [1, num_tokens, dim]
        prompt_attention_mask = torch.ones(
            (prompt_embeds.shape[0], prompt_embeds.shape[1]), dtype=torch.int64, device=device
        )
        negative_prompt_embeds = None
        negative_prompt_attention_mask = None

        # Create null embeddings for CFG if needed
        if do_classifier_free_guidance:
            grey_img = Image.fromarray(np.ones((512, 512, 3), dtype=np.uint8) * 128)
            with torch.no_grad():
                pv = self.cradio_image_processor(
                    images=[grey_img],
                    return_tensors="pt",
                    do_resize=True,
                ).pixel_values.to(device)
                _null_summary, null_features = self.image_encoder(pv)
                null_features = torch.amax(null_features, dim=0, keepdim=True)

            negative_prompt_embeds = null_features
            negative_prompt_attention_mask = torch.ones(
                negative_prompt_embeds.shape[0],
                negative_prompt_embeds.shape[1],
                dtype=torch.int64,
                device=device,
            )

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

    def prepare_multiview_inputs(
        self,
        data_dict,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        do_classifier_free_guidance: bool = True,
        augment_cond: bool = False,
        augment_sigma_sample_p_mean: float = -3.0,
        augment_sigma_sample_p_std: float = 1.0,
        ray_embedding_func: str = "downsample",
    ) -> dict[str, torch.Tensor]:
        """
        Process multiview data (similar to CondProcessor) and prepare all inputs for the model.

        This method:
        1. Encodes images with VAE to latent space
        2. Encodes/downsamples Plucker rays
        3. Creates conditioning mask
        4. Creates camera embeddings (c2w_relatives + fovs)
        5. Handles CFG by creating conditional/unconditional pairs

        Args:
            data_dict: AttrDict with keys:
                - x: images tensor [N, 3, H, W] in [-1, 1]
                - x_white_background: images with white background [N, 3, H, W]
                - n_target: number of target views to generate
                - plucker_image: Plucker ray embeddings [N, 6, H, W]
                - c2w_relatives: camera matrices [N, 4, 4]
                - fovs: field of view values [N]
                - relative_brightness: per-view brightness [N]
            device: Device for tensors
            dtype: Data type for model (bf16/fp16)
            do_classifier_free_guidance: Whether to prepare CFG pairs
            augment_cond: Whether to add noise to conditioning latents
            augment_sigma_sample_p_mean: Mean for augmentation noise
            augment_sigma_sample_p_std: Std for augmentation noise
            ray_embedding_func: How to encode rays ("downsample" or "vae")

        Returns:
            Dict with all tensors prepared for model.forward_with_dpmsolver:
                - clean_images: VAE-encoded images for conditioning
                - rays: Encoded/downsampled Plucker rays
                - cond_mask: Conditioning mask (0 for target, 1 for cond)
                - camera_emb: Camera embeddings [N, 17]
                - relative_brightness: Per-view brightness [N, 1]
                - data_info: img_hw and aspect_ratio info
                - y_mask: Attention mask for prompt
                - x_seq_len: Sequence lengths for batching
        """
        import math
        from statistics import NormalDist

        from torchvision.transforms.functional import resize as torch_resize

        VAE_MINIBATCH_SIZE = 16

        # Get scaling factor
        if hasattr(self.vae, "cfg"):
            scaling_factor = self.vae.cfg.scaling_factor if self.vae.cfg.scaling_factor else 0.41407
        else:
            scaling_factor = 0.41407

        # Get all images and rays
        n_target = data_dict.n_target
        images = data_dict.x.to(device, dtype=self.vae.dtype)
        plucker_rays = data_dict.plucker_image.to(device, dtype=self.vae.dtype)

        num_views = images.shape[0]

        # Encode images with VAE in batches
        n_batch = math.ceil(num_views / VAE_MINIBATCH_SIZE)
        x_encoded_list = []
        ray_encoded_list = []

        for i in range(n_batch):
            start = i * VAE_MINIBATCH_SIZE
            end = min((i + 1) * VAE_MINIBATCH_SIZE, num_views)

            with torch.no_grad():
                z = self.vae.encode(images[start:end])
                if isinstance(z, EncoderOutput):
                    z = z.latent
                z = z * scaling_factor

                # Encode or downsample rays
                if ray_embedding_func == "vae":
                    z_ray_0 = self.vae.encode(plucker_rays[start:end, :3])
                    if isinstance(z_ray_0, EncoderOutput):
                        z_ray_0 = z_ray_0.latent
                    z_ray_0 = z_ray_0 * scaling_factor
                    z_ray_1 = self.vae.encode(plucker_rays[start:end, 3:])
                    if isinstance(z_ray_1, EncoderOutput):
                        z_ray_1 = z_ray_1.latent
                    z_ray_1 = z_ray_1 * scaling_factor
                    z_ray = torch.cat([z_ray_0, z_ray_1], dim=1)
                else:  # "downsample"
                    z_ray = torch_resize(plucker_rays[start:end], z.shape[2])

            x_encoded_list.append(z)
            ray_encoded_list.append(z_ray)

        x_encoded = torch.cat(x_encoded_list, dim=0)  # [N, C, H, W]
        ray_encoded = torch.cat(ray_encoded_list, dim=0)  # [N, 6, H, W]

        latent_size = x_encoded.shape[-1]

        # Separate target and conditioning views
        x_enc_target = x_encoded[:n_target]
        x_enc_cond = x_encoded[n_target:]

        # Apply optional noise augmentation to conditioning latents
        if augment_cond and x_enc_cond.shape[0] > 0:
            from statistics import NormalDist

            gaussian_dist = NormalDist(mu=augment_sigma_sample_p_mean, sigma=augment_sigma_sample_p_std)
            cdf_val = np.random.uniform()
            log_sigma = torch.tensor(gaussian_dist.inv_cdf(cdf_val), device=device)
            sigma = torch.exp(log_sigma).to(device=device, dtype=x_encoded.dtype)
            epsilon = torch.randn_like(x_enc_cond)
            x_enc_cond = x_enc_cond + epsilon * sigma.view(-1, 1, 1, 1)

        # Reconstruct encoded images and create cond_mask
        if x_enc_cond.shape[0] == 0:
            x_nvCHW = x_enc_target
            cond_mask = torch.zeros(n_target, x_encoded.shape[1], latent_size, latent_size, device=device)
        else:
            x_nvCHW = torch.cat([x_enc_target, x_enc_cond], dim=0)
            cond_mask_target = torch.zeros(n_target, x_encoded.shape[1], latent_size, latent_size, device=device)
            cond_mask_cond = torch.ones(
                num_views - n_target, x_encoded.shape[1], latent_size, latent_size, device=device
            )
            cond_mask = torch.cat([cond_mask_target, cond_mask_cond], dim=0)

        # Create camera embeddings: [c2w_relatives (16) + fov (1)] = 17 dims
        c2w_relatives = data_dict.c2w_relatives.to(device)
        fovs = data_dict.fovs.to(device)
        camera_emb = torch.cat([c2w_relatives.reshape(-1, 16), fovs.unsqueeze(-1)], dim=-1)  # [N, 17]

        # Get relative brightness
        relative_brightness = data_dict.relative_brightness.to(device)
        if relative_brightness.dim() == 1:
            relative_brightness = relative_brightness.unsqueeze(-1)  # [N, 1]

        # Cast to model dtype
        x_nvCHW = x_nvCHW.to(dtype)
        ray_encoded = ray_encoded.to(dtype)
        cond_mask = cond_mask.to(dtype)
        camera_emb = camera_emb.to(dtype)
        relative_brightness = relative_brightness.to(dtype)

        # Prepare outputs
        result = {
            "clean_images": x_nvCHW,
            "rays": ray_encoded,
            "cond_mask": cond_mask,
            "camera_emb": camera_emb,
            "relative_brightness": relative_brightness,
            "x_seq_len": [num_views],
            "n_target": n_target,
            "latent_size": latent_size,
        }

        # If CFG, double the tensors (uncond first, then cond)
        if do_classifier_free_guidance:
            uncond_mask = torch.zeros_like(cond_mask)
            result["clean_images_uncond"] = x_nvCHW.clone()
            result["cond_mask_uncond"] = uncond_mask
            result["rays_uncond"] = ray_encoded.clone()
            result["camera_emb_uncond"] = camera_emb.clone()
            result["relative_brightness_uncond"] = relative_brightness.clone()

        return result

    def _extract_conditioning_images(self, data_dict, num_target_views, num_views):
        """Extract conditioning images from data_dict for c-radio encoding."""
        cond_images = []
        for idx in range(num_target_views, num_views):
            img_tensor = data_dict.x_white_background[idx]
            pil_img = Image.fromarray(
                ((img_tensor + 1.0) / 2.0 * 255).clamp(0, 255).permute(1, 2, 0).to(torch.uint8).cpu().numpy()
            )
            cond_images.append(pil_img)
        return cond_images

    def _build_multiview_kwargs(
        self,
        mv_inputs: dict[str, torch.Tensor],
        do_classifier_free_guidance: bool,
        aug_sigmas=None,
    ) -> dict[str, Any]:
        """Build the multiview keyword arguments dict.

        For CFG the conditional and unconditional tensors are concatenated
        (uncond first) so they can be passed through the model in one shot
        alongside the doubled ``latent_model_input``.
        """
        if do_classifier_free_guidance:
            joined_images = torch.cat(
                [
                    mv_inputs["clean_images_uncond"] * mv_inputs["cond_mask_uncond"],
                    mv_inputs["clean_images"] * mv_inputs["cond_mask"],
                ],
                dim=0,
            )
            joined_cond_mask = torch.cat([mv_inputs["cond_mask_uncond"], mv_inputs["cond_mask"]])
            joined_rays = torch.cat([mv_inputs["rays_uncond"], mv_inputs["rays"]])
            joined_camera_emb = torch.cat([mv_inputs["camera_emb_uncond"], mv_inputs["camera_emb"]])
            joined_brightness = torch.cat([mv_inputs["relative_brightness_uncond"], mv_inputs["relative_brightness"]])
            x_seq_len = mv_inputs["x_seq_len"] + mv_inputs["x_seq_len"]
        else:
            joined_images = mv_inputs["clean_images"] * mv_inputs["cond_mask"]
            joined_cond_mask = mv_inputs["cond_mask"]
            joined_rays = mv_inputs["rays"]
            joined_camera_emb = mv_inputs["camera_emb"]
            joined_brightness = mv_inputs["relative_brightness"]
            x_seq_len = mv_inputs["x_seq_len"]

        return dict(
            clean_images=joined_images,
            x_seq_len=x_seq_len,
            cond_mask=joined_cond_mask,
            rays=joined_rays,
            camera_emb=joined_camera_emb,
            aug_sigmas=aug_sigmas,
            relative_brightness=joined_brightness,
        )

    def decode_latents(
        self,
        latents: torch.Tensor,
        output_type: str = "pil",
    ) -> list[Image.Image] | np.ndarray | torch.Tensor:
        """Decode VAE latents to images."""
        scaling_factor = self.vae.config.scaling_factor if self.vae.config.scaling_factor else 0.41407
        images = self.vae.decode(latents / scaling_factor).sample

        if output_type == "pil":
            images = (images + 1.0) / 2.0
            images = images.clamp(0, 1)
            images = (images * 255).permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
            return [Image.fromarray(img) for img in images]
        elif output_type == "np":
            images = (images + 1.0) / 2.0
            images = images.clamp(0, 1)
            return images.permute(0, 2, 3, 1).cpu().numpy()
        else:
            return images

    @torch.no_grad()
    def __call__(
        self,
        # Pre-processed data dict (from preproc pipeline)
        data_dict,
        *,
        # Sampling parameters
        num_inference_steps: int = 20,
        timesteps: list[int] | None = None,
        sigmas: list[float] | None = None,
        guidance_scale: float = 4.5,
        flow_shift: float = 1.0,
        # Prompt parameters
        prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        negative_prompt: str | list[str] | None = None,
        null_prompt_embeds: torch.Tensor | None = None,  # Pre-computed null embeddings for CFG
        # Output parameters
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> dict[str, Any] | tuple:
        """
        Generate multiview images.

        Args:
            prompt: Text prompt for generation
            camera_matrices: Camera pose matrices for each view [num_views, 4, 4]
            fovs: Field of view for each camera [num_views]
            rays: Plucker ray embeddings [num_views, 6, H, W] (optional)
            cond_mask: Which views are conditioning vs target [num_views, 1, H, W]
            conditioning_images: Known conditioning images [num_views, 3, H, W] (optional)
            num_target_views: Number of views to generate (first N views)
            aug_sigmas: Camera augmentation sigmas (optional)
            relative_brightness: Relative brightness per view (optional)
            skeletal_poses: 3D skeleton poses (optional)
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale
            flow_shift: Flow matching shift parameter
            generator: Random generator for reproducibility
            latents: Pre-generated latents (optional)
            prompt_embeds: Pre-computed prompt embeddings (optional)
            output_type: "pil", "np", or "latent"
            return_dict: Whether to return dict or tuple

        Returns:
            Dictionary or tuple with generated images
        """
        try:
            device = self._execution_device
        except AttributeError:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")

        do_classifier_free_guidance = guidance_scale > 1.0

        # Get model dtype
        model_dtype = next(self.transformer.parameters()).dtype

        # Prepare all multiview inputs using data_dict
        t_prepare_start = torch.cuda.Event(enable_timing=True)
        t_prepare_end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t_prepare_start.record()
        mv_inputs = self.prepare_multiview_inputs(
            data_dict=data_dict,
            device=device,
            dtype=model_dtype,
            do_classifier_free_guidance=do_classifier_free_guidance,
        )
        t_prepare_end.record()
        torch.cuda.synchronize()
        prepare_ms = t_prepare_start.elapsed_time(t_prepare_end)
        print(f"   Prepare multiview inputs: {prepare_ms / 1000:.2f}s")

        num_views = len(data_dict.x)
        num_target_views = data_dict.n_target

        # Extract conditioning images for c-radio from data_dict
        t_extract_start = torch.cuda.Event(enable_timing=True)
        t_extract_end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t_extract_start.record()
        if prompt_embeds is None:
            conditioning_images = self._extract_conditioning_images(
                data_dict,
                num_target_views,
                num_views,
            )
            (
                prompt_embeds,
                prompt_attention_mask,
                negative_prompt_embeds,
                negative_prompt_attention_mask,
            ) = self.encode_image_prompt(
                conditioning_images=conditioning_images,
                device=device,
                do_classifier_free_guidance=do_classifier_free_guidance,
            )
        t_extract_end.record()
        torch.cuda.synchronize()
        extract_ms = t_extract_start.elapsed_time(t_extract_end)
        print(f"   Extract conditioning images: {extract_ms / 1000:.2f}s")
        # Prepare initial latents from clean_images shape
        initial_latents = torch.randn_like(mv_inputs["clean_images"])

        # Concatenate for CFG (negative first, then positive — same as SanaPipeline)
        if do_classifier_free_guidance:
            if negative_prompt_embeds is None:
                raise ValueError("negative_prompt_embeds required for CFG")
            prompt_embeds_combined = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_mask_combined = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
        else:
            prompt_embeds_combined = prompt_embeds
            prompt_mask_combined = prompt_attention_mask

        mv_kwargs = self._build_multiview_kwargs(
            mv_inputs,
            do_classifier_free_guidance,
        )

        # --- 4. Prepare timesteps ---
        self.scheduler.flow_shift = flow_shift
        sched_timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
        )
        initial_latents = torch.randn_like(mv_inputs["clean_images"])

        latent_channels = initial_latents.shape[1]
        latents = initial_latents.clone()  # [num_views, C, H, W]

        # --- 6. Denoising loop ---
        timestep_scale = (
            getattr(self.transformer.config, "timestep_scale", 1.0) if hasattr(self.transformer, "config") else 1.0
        )

        t_loop_start = torch.cuda.Event(enable_timing=True)
        t_loop_end = torch.cuda.Event(enable_timing=True)
        with torch.autocast("cuda", dtype=model_dtype):
            t_loop_start.record()
            for i, t in tqdm.tqdm(enumerate(sched_timesteps), total=len(sched_timesteps), desc="Denoising"):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                timestep = t.expand(latent_model_input.shape[0]) * timestep_scale

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds_combined,
                    encoder_attention_mask=prompt_mask_combined,
                    timestep=timestep,
                    return_dict=False,
                    **mv_kwargs,
                )[0]

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                if self.transformer.config.out_channels // 2 == latent_channels:
                    noise_pred = noise_pred.chunk(2, dim=1)[0]

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            t_loop_end.record()
            torch.cuda.synchronize()
            loop_ms = t_loop_start.elapsed_time(t_loop_end)
            print(
                f"   Denoising loop: {loop_ms / 1000:.2f}s "
                f"({loop_ms / num_inference_steps:.1f}ms/step, {num_inference_steps} steps)"
            )

            # --- 7. Extract target views and decode ---
            denoised_latents = latents[:num_target_views]
            torch.cuda.empty_cache()

            if output_type == "latent":
                if return_dict:
                    return {"images": denoised_latents, "n_target": num_target_views}
                return (denoised_latents,)

            t_decode_start = torch.cuda.Event(enable_timing=True)
            t_decode_end = torch.cuda.Event(enable_timing=True)
            t_decode_start.record()
            images = self.decode_latents(denoised_latents, output_type=output_type)
            t_decode_end.record()
            torch.cuda.synchronize()
            decode_ms = t_decode_start.elapsed_time(t_decode_end)
            print(f"   VAE decode: {decode_ms / 1000:.2f}s")

        if return_dict:
            return {"images": images, "n_target": num_target_views}
        return (images,)
