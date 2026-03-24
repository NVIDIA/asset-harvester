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

import os

import numpy as np
import torch
from PIL import Image

# prevent NVFuser from reusing incompatible compilations between subsequent calls
torch._C._jit_set_bailout_depth(0)  # type: ignore[attr-defined]
os.environ["PYTORCH_NVFUSER_DISABLE"] = "1"


def mask2bbox(masks: torch.Tensor) -> torch.Tensor:
    """Compute tight bounding boxes from binary masks."""
    N = masks.shape[0]
    bboxes = masks.new_zeros((N, 4), dtype=torch.float32)
    x_any = torch.any(masks, dim=1)
    y_any = torch.any(masks, dim=2)
    for i in range(N):
        x = torch.where(x_any[i, :])[0]
        y = torch.where(y_any[i, :])[0]
        if len(x) > 0 and len(y) > 0:
            bboxes[i, :] = bboxes.new_tensor([x[0], y[0], x[-1] + 1, y[-1] + 1])

    return bboxes


class Mask2FormerSegmentationEstimator:
    """Inferencer for JIT saved models."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        input_size: tuple[int, int] | None = None,
        mean: tuple[float, float, float] | None = None,
        std: tuple[float, float, float] | None = None,
        pred_score_thr: float = 0.3,
    ) -> None:
        self.device = device
        self.input_size = input_size
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1) if mean is not None else None
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1) if std is not None else None
        self.pred_score_thr = pred_score_thr

        try:
            self.model = torch.jit.load(model_path, map_location=device)
        except Exception as e:
            raise RuntimeError(f"Failed to load JIT model from {model_path}: {e}")

    def cleanup(self) -> None:
        """Clean up model resources and free GPU memory."""
        if hasattr(self, "model"):
            self.model.cpu()
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def preprocess(self, img: np.ndarray) -> torch.Tensor:
        """Preprocess a batch of images."""
        if not isinstance(img, torch.Tensor):
            img_tensor = torch.from_numpy(img).to(torch.float32)
        else:
            img_tensor = img.to(torch.float32)
        img_tensor = img_tensor.to(self.device)

        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)

        img_tensor = img_tensor.permute(0, 3, 1, 2).float() / 255.0

        if self.input_size is not None and self.input_size != img_tensor.shape[2:4]:
            img_tensor = torch.nn.functional.interpolate(img_tensor, size=self.input_size, mode="bicubic")

        if self.mean is not None and self.std is not None:
            img_tensor = (img_tensor - self.mean) / self.std

        return img_tensor

    def postprocess(self, results: tuple[torch.Tensor, ...]):
        """Postprocess the results. Filter out instances with score below pred_score_thr."""
        pan_seg, ins_labels, ins_masks, ins_scores, sem_seg = results
        pan_seg = pan_seg.cpu()
        ins_labels = ins_labels.cpu()
        ins_masks = ins_masks.cpu()
        ins_scores = ins_scores.cpu()
        sem_seg = sem_seg.cpu()

        valid_mask = ins_scores[0] >= self.pred_score_thr

        semantic_seg = sem_seg
        instance_seg = {
            "classes": ins_labels[0][valid_mask].numpy(),
            "instance_masks": np.packbits(ins_masks[0][valid_mask].numpy()),
            "scores": ins_scores[0][valid_mask].numpy(),
            "bboxes": mask2bbox(ins_masks[0])[valid_mask].numpy(),
        }
        return semantic_seg, instance_seg

    def postprocess_batch(self, results: tuple[torch.Tensor, ...]) -> list[tuple[torch.Tensor, dict[str, np.ndarray]]]:
        """Postprocess the results. Filter out instances with score below pred_score_thr."""
        pan_seg, ins_labels, ins_masks, ins_scores, sem_seg = results
        pan_seg = pan_seg.cpu()
        ins_labels = ins_labels.cpu()
        ins_masks = ins_masks.cpu()
        ins_scores = ins_scores.cpu()
        sem_seg = sem_seg.cpu()

        batch_size = ins_scores.shape[0]
        processed_results = []
        for i in range(batch_size):
            valid_mask = ins_scores[i] >= self.pred_score_thr

            semantic_seg = sem_seg[i] if sem_seg.numel() > 0 else sem_seg
            instance_seg = {
                "classes": ins_labels[0][valid_mask].numpy(),
                "instance_masks": np.packbits(ins_masks[0][valid_mask].numpy()),
                "scores": ins_scores[0][valid_mask].numpy(),
                "bboxes": mask2bbox(ins_masks[0])[valid_mask].numpy(),
            }

            processed_results.append((semantic_seg, instance_seg))

        return processed_results

    def predict(self, img: np.ndarray | Image.Image) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
        """Predict semantic and instance segmentation for a single image."""
        if isinstance(img, Image.Image):
            img = np.array(img)

        tensor = self.preprocess(img)

        with torch.no_grad():
            results = self.model(tensor)

        results = self.postprocess(results)
        return results

    def predict_batch(self, img: np.ndarray | list[np.ndarray]) -> list[tuple[torch.Tensor, dict[str, np.ndarray]]]:
        """Predict semantic and instance segmentation for batch images."""
        if isinstance(img, list):
            img_array = np.stack([np.array(x) for x in img], axis=0)
        else:
            img_array = img

        tensor = self.preprocess(img_array)

        with torch.no_grad():
            results = self.model(tensor)

        results = self.postprocess_batch(results)
        return results
