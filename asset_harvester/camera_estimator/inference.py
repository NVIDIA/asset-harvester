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
Inference utilities and AHCEstimator for camera/object attribute prediction.

Ported from ah-camera-estimator/inference.py. The AHCEstimator class accepts
an optional already-loaded C-RADIO (model, processor) pair so that the backbone
can be shared with the diffusion pipeline (avoiding a second copy in VRAM).

Standalone usage:
    estimator = AHCEstimator(checkpoint_path, device="cuda")
    result = estimator.run("/path/to/frame_mask_folder")

Shared-backbone usage (inside run_inference.py):
    cradio_model, cradio_processor = get_c_radio(device=device)
    estimator = AHCEstimator(
        checkpoint_path,
        device=device,
        cradio_model=cradio_model,
        cradio_image_processor=cradio_processor,
    )
    result = estimator.run("/path/to/frame_mask_folder")
    # result keys: lwh, frame_filenames, mask_filenames, cam_poses, dists, fov
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T

from .models import AttributeModel, CRadioBackboneAdapter

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
GRAY_VALUE = 128  # RGB gray for masked background [0, 255]
_AHC_IMAGE_SIZE = 224  # input resolution expected by the AHC MLP head


def apply_mask(image: np.ndarray, mask: np.ndarray, gray: int = GRAY_VALUE) -> np.ndarray:
    """Set background (mask == 0) to gray; keep foreground unchanged."""
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    fg = (mask > 0).astype(np.uint8)
    fg_3d = np.stack([fg, fg, fg], axis=-1)
    gray_img = np.full_like(image, gray)
    return np.where(fg_3d, image, gray_img).astype(np.uint8)


def load_model_from_pruned_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    cradio_model: nn.Module | None = None,
    cradio_image_processor=None,
    model_cfg: dict | None = None,
) -> AttributeModel:
    """
    Build an AttributeModel from a pruned MLP checkpoint.

    The checkpoint only needs to contain model.head.* (or head.*) keys.
    The backbone (C-RADIO) is not stored in the pruned checkpoint; it is
    loaded via get_c_radio() when cradio_model is None, or re-used from
    the provided cradio_model / cradio_image_processor pair.

    Args:
        checkpoint_path: Path to a pruned .safetensors file.
        device: Target device.
        cradio_model: Pre-loaded C-RADIO model. If None, loads from HuggingFace.
        cradio_image_processor: Pre-loaded CLIPImageProcessor. If None, loads from HuggingFace.
        model_cfg: Fallback model config dict (backbone_name, head_hidden_dim,
                   head_num_layers). Overridden by hyper_parameters.model_cfg
                   embedded in the checkpoint when present.
    """
    try:
        from omegaconf import OmegaConf

        _have_omegaconf = True
    except ImportError:
        _have_omegaconf = False

    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        ckpt = {"state_dict": load_file(checkpoint_path, device="cpu")}
    else:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    hp = ckpt.get("hyper_parameters", {})
    if hp and "model_cfg" in hp and _have_omegaconf:
        model_cfg = OmegaConf.to_container(hp["model_cfg"], resolve=True)
    elif hp and "model_cfg" in hp:
        model_cfg = dict(hp["model_cfg"])

    mc = model_cfg or {}
    if _have_omegaconf:
        from omegaconf import DictConfig

        if isinstance(mc, DictConfig):
            mc = OmegaConf.to_container(mc, resolve=True)

    head_hidden_dim = mc.get("head_hidden_dim", 1024)
    head_num_layers = mc.get("head_num_layers", 4)

    # Build or reuse backbone
    if cradio_model is None or cradio_image_processor is None:
        from asset_harvester.multiview_diffusion.utils.model_builder import get_c_radio

        cradio_model, cradio_image_processor = get_c_radio(device=device)

    backbone = CRadioBackboneAdapter(cradio_model, cradio_image_processor)
    feat_dim = backbone.feat_dim

    model = AttributeModel(
        backbone=backbone,
        feat_dim=feat_dim,
        head_hidden_dim=head_hidden_dim,
        head_num_layers=head_num_layers,
    )

    state = ckpt.get("state_dict", ckpt)
    head_state: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if "model.head" in k:
            head_state[k.replace("model.head", "head")] = v
        elif k.startswith("head."):
            head_state[k] = v

    if not head_state:
        raise ValueError(f"No model.head.* keys found in {checkpoint_path}. ")

    head_state_stripped = {k.replace("head.", "", 1): v for k, v in head_state.items()}
    model.head.load_state_dict(head_state_stripped, strict=True)
    return model.to(device).eval()


class AHCEstimator:
    """
    Stateful wrapper for camera/object attribute estimation from masked images.

    Loads the pruned MLP head once on construction. The C-RADIO backbone can be
    shared with the diffusion pipeline by passing the already-loaded model and
    processor; if omitted they are loaded from HuggingFace automatically.

    Args:
        checkpoint_path: Path to AHC .safetensors checkpoint.
        device: Target device string or torch.device (default: "cuda" if available).
        cradio_model: Optional pre-loaded C-RADIO AutoModel.
        cradio_image_processor: Optional pre-loaded CLIPImageProcessor.
        batch_size: Number of images to process per forward pass.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str | torch.device | None = None,
        cradio_model: nn.Module | None = None,
        cradio_image_processor=None,
        batch_size: int = 8,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.batch_size = batch_size

        self.model = load_model_from_pruned_checkpoint(
            checkpoint_path=checkpoint_path,
            device=self.device,
            cradio_model=cradio_model,
            cradio_image_processor=cradio_image_processor,
        )

        self._transform = T.Compose(
            [
                T.Resize((_AHC_IMAGE_SIZE, _AHC_IMAGE_SIZE)),
                T.ToTensor(),
            ]
        )

    @torch.no_grad()
    def run(
        self,
        pairs: list[tuple[str, str]],
        debug: bool = False,
    ) -> dict:
        """
        Run attribute estimation on all frame/mask pairs in a folder.

        Args:
            pairs: List of (frame_path, mask_path) pairs.
            debug: If True, save masked images to <input_folder>/debug_masked/.

        Returns:
            Dict with keys:
                lwh (list[float]): Averaged object dimensions [L, W, H].
                frame_filenames (list[str])
                mask_filenames (list[str])
                cam_poses (list[list[float]]): Per-frame unit direction vectors (3,).
                dists (list[float]): Per-frame camera distances.
                fov (list[float]): Per-frame field-of-view values in degrees.
        """
        frame_filenames: list[str] = []
        mask_filenames: list[str] = []
        dists: list[float] = []
        lwh_list: list[list[float]] = []
        fov_list: list[float] = []
        cam_poses: list[list[float]] = []

        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            imgs = []
            for frame_path, mask_path in batch:
                img = np.array(Image.open(frame_path).convert("RGB").resize((512, 512)))
                mask = np.array(Image.open(mask_path).resize((512, 512), Image.NEAREST))
                masked = apply_mask(img, mask)
                imgs.append(self._transform(Image.fromarray(masked)))

            x = torch.stack(imgs).to(self.device)
            out = self.model(x)

            for j, (frame_path, mask_path) in enumerate(batch):
                frame_filenames.append(os.path.basename(frame_path))
                mask_filenames.append(os.path.basename(mask_path))
                dists.append(float(out["distance"][j].item()))
                lwh_list.append(out["lwh"][j].cpu().tolist())
                fov_list.append(float(out["fov"][j].item()))
                cam_poses.append(out["cam_pose"][j].cpu().tolist())

        lwh = np.mean(lwh_list, axis=0).tolist()
        return {
            "lwh": lwh,
            "frame_filenames": frame_filenames,
            "mask_filenames": mask_filenames,
            "cam_poses": cam_poses,
            "dists": dists,
            "fov": fov_list,
        }

    def run_and_save(
        self,
        pairs: list[tuple[Path, Path]],
        output_path: str | Path | None = None,
        debug: bool = False,
    ) -> dict:
        """Run estimation and write results to a camera_data.json file."""
        result = self.run(pairs, debug=debug)
        if output_path is None:
            output_path = Path(pairs[0][0].parent) / "camera_data.json"
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        return result
