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

"""Image moderation utility backed by Llama Guard 3 Vision."""

from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

DEFAULT_IMAGE_GUARD_MODEL_ID = "meta-llama/Llama-Guard-3-11B-Vision"
# Llama Guard family examples use a 0.5 unsafe-probability threshold for binary gating.
DEFAULT_IMAGE_GUARD_THRESHOLD = 0.5
_MAX_NEW_TOKENS = 16
_PROMPT = (
    "Classify whether this user-supplied image is safe to process in an automated asset-generation pipeline. "
    "Reply with exactly one word: safe or unsafe."
)
_REPORT_FILENAME = "image_guard_report.json"


class ImageGuardError(RuntimeError):
    """Raised when Llama Guard returns an unexpected moderation result."""


@dataclass
class ImageGuardResult:
    """Concise moderation result for one image."""

    passed: bool
    score: float
    label: str
    raw_response: str
    model_id: str
    inference_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ImageGuardSampleResult:
    """Moderation result for one sample containing one or more images."""

    kept_pairs: list[tuple[str, str]]
    kept_indices: list[int]
    report: dict


class ImageGuard:
    """Lazy-loading wrapper around Llama Guard 3 Vision."""

    def __init__(
        self,
        model_id: str = DEFAULT_IMAGE_GUARD_MODEL_ID,
        threshold: float = DEFAULT_IMAGE_GUARD_THRESHOLD,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if dtype is None:
            if self.device.type == "cuda" and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            elif self.device.type == "cuda":
                dtype = torch.float16
            else:
                dtype = torch.float32
        self.dtype = dtype
        self.model_id = model_id
        self.threshold = threshold
        self._processor = None
        self._model = None
        self._safe_token_ids: list[int] = []
        self._unsafe_token_ids: list[int] = []

    def _load(self) -> None:
        if self._processor is not None and self._model is not None:
            return

        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
        ).to(self.device)
        self._model.eval()
        self._safe_token_ids = self._resolve_label_token_ids("safe")
        self._unsafe_token_ids = self._resolve_label_token_ids("unsafe")

    def load(self) -> None:
        """Eagerly load the model so configuration/auth errors surface immediately."""
        self._load()

    def unload(self) -> None:
        if self._model is not None:
            self._model.to("cpu")
        self._processor = None
        self._model = None
        self._safe_token_ids = []
        self._unsafe_token_ids = []
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def check_image(self, image: str | Path | Image.Image | np.ndarray) -> ImageGuardResult:
        self._load()
        image_pil = self._coerce_image(image)
        start_time = time.perf_counter()
        text, score = self._generate_response_and_score(image_pil)
        inference_seconds = time.perf_counter() - start_time
        label = self._parse_label(text)
        return ImageGuardResult(
            passed=label == "safe" and score < self.threshold,
            score=score,
            label=label,
            raw_response=text,
            model_id=self.model_id,
            inference_seconds=inference_seconds,
        )

    def moderate_sample(
        self,
        image_pairs: list[tuple[str, str]],
        track_id: str,
        output_dir: str,
        allowed_indices: list[int] | None = None,
    ) -> ImageGuardSampleResult:
        """Moderate one sample, write a concise report, and keep only passing images."""
        kept_pairs = []
        kept_indices = []
        image_reports = []
        total_inference_seconds = 0.0

        for pair_idx, (frame_path, mask_path) in enumerate(image_pairs):
            image_name = os.path.basename(frame_path)
            try:
                result = self.check_image(frame_path)
                total_inference_seconds += result.inference_seconds
                image_reports.append(
                    {
                        "image": image_name,
                        "passed": result.passed,
                        "score": result.score,
                        "label": result.label,
                        "raw_response": result.raw_response,
                        "inference_seconds": result.inference_seconds,
                    }
                )
                if result.passed:
                    kept_pairs.append((frame_path, mask_path))
                    kept_indices.append(allowed_indices[pair_idx] if allowed_indices is not None else pair_idx)
            except Exception as e:
                image_reports.append(
                    {
                        "image": image_name,
                        "passed": False,
                        "error": str(e),
                    }
                )

        total_images = len(image_pairs)
        report = {
            "enabled": True,
            "model_id": self.model_id,
            "threshold": self.threshold,
            "sample_passed": bool(kept_pairs),
            "total_images": total_images,
            "used_images": len(kept_pairs),
            "skipped_images": total_images - len(kept_pairs),
            "elapsed_seconds": total_inference_seconds,
            "avg_seconds_per_image": total_inference_seconds / total_images if total_images else 0.0,
            "images": image_reports,
        }
        self.write_report(output_dir, track_id, report)
        return ImageGuardSampleResult(
            kept_pairs=kept_pairs,
            kept_indices=kept_indices,
            report=report,
        )

    @staticmethod
    def write_report(output_dir: str, track_id: str, report: dict) -> None:
        """Write a concise moderation report alongside sample outputs."""
        report_dir = os.path.join(output_dir, track_id)
        os.makedirs(report_dir, exist_ok=True)
        with open(os.path.join(report_dir, _REPORT_FILENAME), "w") as f:
            json.dump(report, f, indent=2)

    def _generate_response_and_score(self, image: Image.Image) -> tuple[str, float]:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": _PROMPT},
                ],
            }
        ]
        prompt = self._processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._processor(images=[image], text=prompt, return_tensors="pt")
        inputs = inputs.to(self.device)

        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=_MAX_NEW_TOKENS,
                output_scores=True,
                return_dict_in_generate=True,
            )

        prompt_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
        output_tokens = generated.sequences[0][prompt_len:]
        if hasattr(self._processor, "decode"):
            text = self._processor.decode(output_tokens, skip_special_tokens=True)
        else:
            text = self._processor.tokenizer.decode(output_tokens, skip_special_tokens=True)
        score = self._compute_unsafe_score(generated.scores)
        return text.strip(), score

    def _compute_unsafe_score(self, scores: tuple[torch.Tensor, ...]) -> float:
        if not scores:
            raise ImageGuardError("Image guard returned no generation scores.")

        first_token_scores = torch.softmax(scores[0][0].float(), dim=-1)
        unsafe_prob = first_token_scores[self._unsafe_token_ids].max().item()
        safe_prob = first_token_scores[self._safe_token_ids].max().item()
        normalizer = safe_prob + unsafe_prob
        if normalizer <= 0.0:
            raise ImageGuardError("Image guard produced a degenerate safe/unsafe score.")
        return unsafe_prob / normalizer

    def _resolve_label_token_ids(self, label: str) -> list[int]:
        variants = [
            label,
            f" {label}",
            label.capitalize(),
            f" {label.capitalize()}",
        ]
        token_ids = []
        for variant in variants:
            ids = self._processor.tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                token_ids.append(ids[0])
        unique_ids = sorted(set(token_ids))
        if not unique_ids:
            raise ImageGuardError(f"Could not resolve a single-token variant for label {label!r}.")
        return unique_ids

    @staticmethod
    def _coerce_image(image: str | Path | Image.Image | np.ndarray) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = np.stack([image, image, image], axis=-1)
            return Image.fromarray(image.astype(np.uint8)).convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image)}")

    @staticmethod
    def _parse_label(text: str) -> str:
        normalized = text.strip().lower()
        if normalized.startswith("unsafe"):
            return "unsafe"
        if normalized.startswith("safe"):
            return "safe"
        if "unsafe" in normalized:
            return "unsafe"
        if "safe" in normalized:
            return "safe"
        raise ImageGuardError(f"Unexpected image guard response: {text!r}")
