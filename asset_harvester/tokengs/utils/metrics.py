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
Image quality metrics for evaluation.
Provides PSNR, LPIPS, and SSIM metrics for image comparison.
Aligned with standard implementations for reproducibility.
"""

import numpy as np
import torch
from einops import reduce
from lpips import LPIPS
from skimage.metrics import structural_similarity


def masked_mean(
    x: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute mean of masked values by soft blending.

    Args:
        x: Input tensor of shape (...,).
        mask: Optional mask tensor in [0, 1]. Shape will be broadcasted to match x.

    Returns:
        Masked mean of x as a scalar tensor.
    """
    eps = 1e-6

    if mask is None:
        return x.mean()

    mask = torch.broadcast_to(mask, x.shape)
    return (x * mask).sum() / mask.sum().clip(min=eps)


class MetricsCalculator:
    """
    A modular calculator for image quality metrics including PSNR, LPIPS, and SSIM.

    Supports optional masks to compute metrics only within masked regions.

    Usage:
        metrics_calc = MetricsCalculator(device='cuda')
        psnr = metrics_calc.calculate_psnr(pred_images, gt_images)
        lpips = metrics_calc.calculate_lpips(pred_images, gt_images)
        ssim = metrics_calc.calculate_ssim(pred_images, gt_images)

        # With mask
        mask = torch.ones_like(pred_images[:, :1])  # [B, 1, H, W]
        psnr = metrics_calc.calculate_psnr(pred_images, gt_images, mask=mask)
    """

    def __init__(self, device="cuda", lpips_net="vgg"):
        """
        Initialize the metrics calculator.

        Args:
            device: Device to run computations on ('cuda' or 'cpu')
            lpips_net: Network to use for LPIPS ('vgg' or 'alex')
        """
        self.device = device
        self.lpips_model = None
        self.lpips_net = lpips_net

    def _get_lpips(self):
        """Get or create LPIPS model (cached per device)."""
        if self.lpips_model is None:
            self.lpips_model = LPIPS(net=self.lpips_net).to(self.device)
            self.lpips_model.requires_grad_(False)
            self.lpips_model.eval()
        return self.lpips_model

    @torch.no_grad()
    def calculate_psnr(self, pred_images, gt_images, mask=None, reduction="mean"):
        """
        Calculate Peak Signal-to-Noise Ratio (PSNR).

        Follows the standard implementation: clips values to [0, 1] and computes
        MSE over spatial and channel dimensions.

        Args:
            pred_images: Predicted images, tensor of shape [B, C, H, W] or [B, V, C, H, W]
            gt_images: Ground truth images, same shape as pred_images
            mask: Optional mask tensor of shape [B, 1, H, W] or [B, V, 1, H, W] in [0, 1].
                  The metric is computed only on pixels with mask > 0.
            reduction: How to reduce the result ('mean', 'none', or 'sum')

        Returns:
            PSNR value(s) in dB
        """
        # Clip values to valid range [0, 1]
        pred_images = pred_images.clip(min=0, max=1)
        gt_images = gt_images.clip(min=0, max=1)

        # Handle 5D tensors [B, V, C, H, W]
        if len(pred_images.shape) == 5:
            B, V, C, H, W = pred_images.shape
            pred_images = pred_images.reshape(B * V, C, H, W)
            gt_images = gt_images.reshape(B * V, C, H, W)
            if mask is not None:
                mask = mask.reshape(B * V, 1, H, W)
            was_5d = True
        else:
            was_5d = False

        # Compute MSE per image
        mse_per_pixel = (pred_images - gt_images) ** 2

        if mask is None:
            # No mask: compute mean over C, H, W dimensions
            mse = reduce(mse_per_pixel, "b c h w -> b", "mean")
        else:
            # With mask: compute masked mean per image
            psnr_list = []
            for i in range(mse_per_pixel.shape[0]):
                mse_val = masked_mean(mse_per_pixel[i], mask[i])
                psnr_list.append(mse_val)
            mse = torch.stack(psnr_list)

        psnr = -10 * mse.log10()

        # Reshape back if input was 5D
        if was_5d:
            psnr = psnr.reshape(B, V)

        if reduction == "mean":
            return psnr.mean()
        elif reduction == "sum":
            return psnr.sum()
        else:  # 'none'
            return psnr

    @torch.no_grad()
    def calculate_lpips(self, pred_images, gt_images, mask=None, reduction="mean"):
        """
        Calculate Learned Perceptual Image Patch Similarity (LPIPS).

        Follows the standard implementation: uses normalize=True which handles
        the conversion from [0, 1] to [-1, 1] internally.
        When mask is provided, images are masked before computing LPIPS, and the
        LPIPS output is averaged over masked regions.

        Args:
            pred_images: Predicted images, tensor of shape [B, C, H, W] or [B, V, C, H, W]
                        Values should be in range [0, 1]
            gt_images: Ground truth images, same shape as pred_images
            mask: Optional mask tensor of shape [B, 1, H, W] or [B, V, 1, H, W] in [0, 1].
                  The metric is computed only on pixels with mask > 0.
            reduction: How to reduce the result ('mean', 'none', or 'sum')

        Returns:
            LPIPS value(s) (lower is better)
        """
        lpips_model = self._get_lpips()

        # Handle 5D tensors [B, V, C, H, W]
        if len(pred_images.shape) == 5:
            B, V, C, H, W = pred_images.shape
            pred_images = pred_images.reshape(B * V, C, H, W)
            gt_images = gt_images.reshape(B * V, C, H, W)
            if mask is not None:
                mask = mask.reshape(B * V, 1, H, W)
            was_5d = True
        else:
            was_5d = False

        # Apply mask to images if provided
        if mask is not None:
            pred_images_masked = pred_images * mask
            gt_images_masked = gt_images * mask
        else:
            pred_images_masked = pred_images
            gt_images_masked = gt_images

        # Use normalize=True to handle [0, 1] -> [-1, 1] conversion automatically
        lpips_output = lpips_model.forward(gt_images_masked, pred_images_masked, normalize=True)

        if mask is None:
            # Extract the scalar value (LPIPS returns [B, 1, 1, 1])
            lpips_value = lpips_output[:, 0, 0, 0]
        else:
            # Compute masked mean over spatial dimensions
            lpips_value_list = []
            for i in range(lpips_output.shape[0]):
                # lpips_output is [B, 1, H', W'] where H', W' might be smaller than H, W
                lpips_map = lpips_output[i, 0]  # [H', W']

                # Resize mask to match LPIPS output spatial dimensions if needed
                if lpips_map.shape != mask[i, 0].shape:
                    mask_resized = torch.nn.functional.interpolate(
                        mask[i : i + 1], size=lpips_map.shape, mode="bilinear", align_corners=False
                    )[0, 0]
                else:
                    mask_resized = mask[i, 0]

                lpips_val = masked_mean(lpips_map, mask_resized)
                lpips_value_list.append(lpips_val)
            lpips_value = torch.stack(lpips_value_list)

        # Reshape back if input was 5D
        if was_5d:
            lpips_value = lpips_value.reshape(B, V)

        if reduction == "mean":
            return lpips_value.mean()
        elif reduction == "sum":
            return lpips_value.sum()
        else:  # 'none'
            return lpips_value

    @torch.no_grad()
    def calculate_ssim(self, pred_images, gt_images, mask=None, reduction="mean"):
        """
        Calculate Structural Similarity Index (SSIM).

        Follows the standard implementation: uses skimage's structural_similarity
        with win_size=11, gaussian_weights=True, channel_axis=0, data_range=1.0.
        Computes SSIM per image.

        Note: When mask is provided, images are masked before computing SSIM. For a more
        sophisticated partial convolution approach (as in DyCheck), a more complex
        implementation would be needed.

        Args:
            pred_images: Predicted images, tensor of shape [B, C, H, W] or [B, V, C, H, W]
            gt_images: Ground truth images, same shape as pred_images
            mask: Optional mask tensor of shape [B, 1, H, W] or [B, V, 1, H, W] in [0, 1].
                  Images are masked before computing SSIM.
            reduction: How to reduce the result ('mean', 'none', or 'sum')

        Returns:
            SSIM value(s) (higher is better, range [0, 1])
        """
        # Handle 5D tensors [B, V, C, H, W]
        if len(pred_images.shape) == 5:
            B, V, C, H, W = pred_images.shape
            pred_images = pred_images.reshape(B * V, C, H, W)
            gt_images = gt_images.reshape(B * V, C, H, W)
            if mask is not None:
                mask = mask.reshape(B * V, 1, H, W)
            was_5d = True
        else:
            was_5d = False

        # Apply mask if provided
        if mask is not None:
            pred_images = pred_images * mask
            gt_images = gt_images * mask

        # Compute SSIM per image using skimage
        ssim_values = []
        for gt, pred in zip(gt_images, pred_images):
            ssim_val = structural_similarity(
                gt.detach().cpu().numpy(),
                pred.detach().cpu().numpy(),
                win_size=11,
                gaussian_weights=True,
                channel_axis=0,
                data_range=1.0,
            )
            ssim_values.append(ssim_val)

        ssim_tensor = torch.tensor(ssim_values, dtype=pred_images.dtype, device=pred_images.device)

        # Reshape back if input was 5D
        if was_5d:
            ssim_tensor = ssim_tensor.reshape(B, V)

        if reduction == "mean":
            return ssim_tensor.mean()
        elif reduction == "sum":
            return ssim_tensor.sum()
        else:  # 'none'
            return ssim_tensor

    @torch.no_grad()
    def calculate_all_metrics(self, pred_images, gt_images, mask=None, reduction="mean"):
        """
        Calculate all metrics (PSNR, LPIPS, SSIM) at once.

        Args:
            pred_images: Predicted images, tensor of shape [B, C, H, W] or [B, V, C, H, W]
            gt_images: Ground truth images, same shape as pred_images
            mask: Optional mask tensor of shape [B, 1, H, W] or [B, V, 1, H, W] in [0, 1].
                  The metrics are computed only on pixels with mask > 0.
            reduction: How to reduce the result ('mean', 'none', or 'sum')

        Returns:
            Dictionary with keys 'psnr', 'lpips', 'ssim'
        """
        return {
            "psnr": self.calculate_psnr(pred_images, gt_images, mask=mask, reduction=reduction),
            "lpips": self.calculate_lpips(pred_images, gt_images, mask=mask, reduction=reduction),
            "ssim": self.calculate_ssim(pred_images, gt_images, mask=mask, reduction=reduction),
        }


# Convenience functions for quick usage without creating a MetricsCalculator instance
_default_calculator = None


def get_default_calculator(device="cuda"):
    """Get or create the default metrics calculator."""
    global _default_calculator
    if _default_calculator is None:
        _default_calculator = MetricsCalculator(device=device)
    return _default_calculator


@torch.no_grad()
def calculate_psnr(pred_images, gt_images, mask=None, reduction="mean"):
    """Calculate PSNR. See MetricsCalculator.calculate_psnr for details."""
    calculator = get_default_calculator(device=pred_images.device)
    return calculator.calculate_psnr(pred_images, gt_images, mask=mask, reduction=reduction)


@torch.no_grad()
def calculate_lpips(pred_images, gt_images, mask=None, reduction="mean"):
    """Calculate LPIPS. See MetricsCalculator.calculate_lpips for details."""
    calculator = get_default_calculator(device=pred_images.device)
    return calculator.calculate_lpips(pred_images, gt_images, mask=mask, reduction=reduction)


@torch.no_grad()
def calculate_ssim(pred_images, gt_images, mask=None, reduction="mean"):
    """Calculate SSIM. See MetricsCalculator.calculate_ssim for details."""
    calculator = get_default_calculator(device=pred_images.device)
    return calculator.calculate_ssim(pred_images, gt_images, mask=mask, reduction=reduction)


@torch.no_grad()
def calculate_all_metrics(pred_images, gt_images, mask=None, reduction="mean"):
    """Calculate all metrics. See MetricsCalculator.calculate_all_metrics for details."""
    calculator = get_default_calculator(device=pred_images.device)
    return calculator.calculate_all_metrics(pred_images, gt_images, mask=mask, reduction=reduction)


class MetricsTracker:
    """
    A tracker for accumulating metrics across multiple batches/samples.

    Usage:
        tracker = MetricsTracker()

        # During evaluation loop
        for data in dataloader:
            results = model(data)
            # Pass metrics dictionary directly
            tracker.update(results)
            # Or pass individual metrics
            # tracker.update(psnr=25.3, lpips=0.12, ssim=0.95)

            # Track timing
            import time
            start = time.time()
            output = model.forward_gaussian(x)
            tracker.update(forward_gaussian_time=time.time() - start)

        # Get statistics
        stats = tracker.get_stats()
        print(f"PSNR: {stats['psnr']['mean']:.4f} ± {stats['psnr']['std']:.4f}")

        # Save to file
        tracker.save_to_file('metrics.txt')
    """

    def __init__(self):
        """Initialize the metrics tracker."""
        self.metrics = {}
        self._timing_units = {}  # Store units for timing metrics

    def update(self, metrics_dict=None, **kwargs):
        """
        Update metrics with new values.

        Args:
            metrics_dict: Optional dictionary of metrics (e.g., {'psnr': 25.3, 'lpips': 0.12})
            **kwargs: Alternative way to pass metrics as keyword arguments

        Examples:
            tracker.update({'psnr': 25.3, 'lpips': 0.12})
            tracker.update(psnr=25.3, lpips=0.12)
        """
        # Support both dictionary and keyword arguments
        if metrics_dict is not None:
            if not isinstance(metrics_dict, dict):
                raise TypeError("metrics_dict must be a dictionary")
            kwargs.update(metrics_dict)

        for metric_name, value in kwargs.items():
            if metric_name not in self.metrics:
                self.metrics[metric_name] = []

            # Handle both scalar values and tensors
            if isinstance(value, torch.Tensor):
                value = value.item()

            self.metrics[metric_name].append(value)

    def get_values(self, metric_name):
        """
        Get all values for a specific metric.

        Args:
            metric_name: Name of the metric

        Returns:
            numpy array of values
        """
        if metric_name not in self.metrics:
            return np.array([])
        return np.array(self.metrics[metric_name])

    def get_stats(self, metric_name=None):
        """
        Get statistics for metrics.

        Args:
            metric_name: If provided, return stats for specific metric.
                        If None, return stats for all metrics.

        Returns:
            Dictionary with 'mean', 'std', 'min', 'max', 'count' for each metric
        """
        if metric_name is not None:
            values = self.get_values(metric_name)
            if len(values) == 0:
                return None
            return {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "count": len(values),
            }

        # Return stats for all metrics
        stats = {}
        for name in self.metrics.keys():
            stats[name] = self.get_stats(name)
        return stats

    def get_mean(self, metric_name):
        """Get mean value for a specific metric."""
        stats = self.get_stats(metric_name)
        return stats["mean"] if stats else None

    def get_std(self, metric_name):
        """Get standard deviation for a specific metric."""
        stats = self.get_stats(metric_name)
        return stats["std"] if stats else None

    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
        self._timing_units = {}

    def _is_timing_metric(self, metric_name):
        """Check if a metric is a timing metric based on name."""
        timing_keywords = ["time", "duration", "latency", "elapsed"]
        return any(keyword in metric_name.lower() for keyword in timing_keywords)

    def _format_metric_value(self, metric_name, value):
        """Format metric value based on its type."""
        if self._is_timing_metric(metric_name):
            # Format timing in milliseconds or seconds
            if value < 1.0:
                return f"{value * 1000:.2f}ms"
            else:
                return f"{value:.3f}s"
        else:
            return f"{value:.4f}"

    def __len__(self):
        """Return the number of samples tracked."""
        if not self.metrics:
            return 0
        # All metrics should have the same length
        return len(next(iter(self.metrics.values())))

    def __str__(self):
        """String representation of current metrics."""
        stats = self.get_stats()
        lines = ["Metrics Statistics:"]
        for name, stat in stats.items():
            if stat:
                lines.append(
                    f"  {name.upper()}: {stat['mean']:.4f} ± {stat['std']:.4f} "
                    f"[{stat['min']:.4f}, {stat['max']:.4f}] (n={stat['count']})"
                )
        return "\n".join(lines)

    def print_summary(self, title="Evaluation Metrics", short=False):
        """
        Print a formatted summary of metrics.

        Args:
            title: Title to display above the metrics
            short: If True, print compact single-line format. If False, print full formatted table.
        """
        stats = self.get_stats()

        if short:
            # Compact single-line format for inline progress updates
            metric_strs = []
            for name, stat in stats.items():
                if stat:
                    mean_str = self._format_metric_value(name, stat["mean"])
                    metric_strs.append(f"{name.upper()}: {mean_str}")
            print(f"[{title}] " + ", ".join(metric_strs))
        else:
            # Full formatted table for final results
            print("=" * 50)
            print(title + ":")
            for name, stat in stats.items():
                if stat:
                    mean_str = self._format_metric_value(name, stat["mean"])
                    std_str = self._format_metric_value(name, stat["std"])
                    print(f"  {name.upper():20s}: {mean_str:>10s} ± {std_str:<10s}")
            print("=" * 50)

    def save_to_file(self, filepath):
        """
        Save metrics to a text file.

        Args:
            filepath: Path to save the metrics file
        """
        stats = self.get_stats()

        with open(filepath, "w") as f:
            f.write("Evaluation Metrics:\n")
            f.write("=" * 70 + "\n")

            # Separate image quality and timing metrics
            quality_metrics = {}
            timing_metrics = {}
            for name, stat in stats.items():
                if stat:
                    if self._is_timing_metric(name):
                        timing_metrics[name] = stat
                    else:
                        quality_metrics[name] = stat

            # Write image quality metrics
            if quality_metrics:
                f.write("\nImage Quality Metrics:\n")
                f.write("-" * 70 + "\n")
                for name, stat in quality_metrics.items():
                    mean_str = self._format_metric_value(name, stat["mean"])
                    std_str = self._format_metric_value(name, stat["std"])
                    min_str = self._format_metric_value(name, stat["min"])
                    max_str = self._format_metric_value(name, stat["max"])
                    f.write(f"{name.upper():20s}: {mean_str:>10s} ± {std_str:<10s} [{min_str:>10s}, {max_str:<10s}]\n")

            # Write timing metrics
            if timing_metrics:
                f.write("\nTiming Metrics:\n")
                f.write("-" * 70 + "\n")
                for name, stat in timing_metrics.items():
                    mean_str = self._format_metric_value(name, stat["mean"])
                    std_str = self._format_metric_value(name, stat["std"])
                    min_str = self._format_metric_value(name, stat["min"])
                    max_str = self._format_metric_value(name, stat["max"])
                    f.write(f"{name.upper():20s}: {mean_str:>10s} ± {std_str:<10s} [{min_str:>10s}, {max_str:<10s}]\n")

            f.write("\n" + "=" * 70 + "\n")
            f.write("Per-sample metrics:\n")
            f.write("-" * 70 + "\n")

            # Write per-sample metrics
            n_samples = len(self)
            if n_samples > 0:
                # Get all metric names
                metric_names = list(self.metrics.keys())

                # Write header
                f.write(f"{'Sample':<10}")
                for name in metric_names:
                    f.write(f"{name.upper():<15}")
                f.write("\n")
                f.write("-" * 70 + "\n")

                # Write values
                for i in range(n_samples):
                    f.write(f"{i:<10}")
                    for name in metric_names:
                        value = self.metrics[name][i]
                        value_str = self._format_metric_value(name, value)
                        f.write(f"{value_str:<15}")
                    f.write("\n")

        print(f"Metrics saved to {filepath}")
