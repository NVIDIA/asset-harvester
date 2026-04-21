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

from __future__ import annotations

import datetime
import json
import os
import time
import warnings
from dataclasses import asdict
from typing import TYPE_CHECKING

import imageio
import numpy as np
import torch
import tyro
from accelerate import Accelerator
from safetensors.torch import load_file

from .data import get_multi_dataloader
from .models import model_registry
from .options import AllConfigs
from .utils.gaussians import Gaussians

warnings.filterwarnings("ignore")

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


def _get_summary_writer_cls():
    # Delay TensorBoard import so CLI help works even if a user-site tensorboard
    # install is incompatible with the active environment's NumPy version.
    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter


def check_gradients_and_clip(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    accelerator: Accelerator,
    opt,
    out: dict,
    data: dict,
    epoch: int,
    iteration: int,
    writer: SummaryWriter,
    log_images: bool = True,
) -> tuple[torch.Tensor | None, bool]:
    """
    Perform safety checks on loss and gradients, then clip gradients.

    This function:
    1. Checks for NaN/Inf in per-scene losses (if safe_mode enabled)
    2. Checks for loss spikes above threshold (if safe_mode enabled)
    3. Checks for NaN/Inf in gradients (if safe_mode enabled)
    4. Clips gradients using the specified gradient_clip value
    5. Checks if gradient norm exceeds grad_norm_cap

    Args:
        model: The model being trained
        optimizer: The optimizer (used to zero gradients if needed)
        scheduler: The learning rate scheduler (stepped if batch is skipped)
        accelerator: Accelerator instance for distributed training
        opt: Options/config object containing training hyperparameters
        out: Output dictionary from model.forward() containing loss and metrics
        data: Input data batch
        epoch: Current epoch number
        iteration: Current iteration number
        writer: TensorBoard writer for logging
        log_images: Whether to log debug images on failure (default: True)

    Returns:
        tuple: (grad_norm, should_skip)
            - grad_norm: The gradient norm after clipping (or None if skipped)
            - should_skip: True if the batch should be skipped, False otherwise
    """
    should_skip = False

    # Only perform checks and clipping when gradients are synchronized
    if not accelerator.sync_gradients:
        return None, False

    # Safe mode: Check for NaN/Inf in loss
    if opt.safe_mode:
        loss_components = [
            ("loss_per_scene", "Loss"),
            ("loss_mse_per_scene", "MSE"),
            ("loss_ssim_per_scene", "SSIM"),
            ("loss_visibility_per_scene", "Visibility"),
        ]

        # Check each loss component for NaN/Inf
        for key, label in loss_components:
            if key not in out:
                continue

            loss_component = out[key]
            if not torch.isfinite(loss_component).all():
                bad_indices = torch.where(~torch.isfinite(loss_component))[0].tolist()
                print(f"[SAFETY] {label} loss contains NaN/Inf at iteration {iteration}! Skipping batch.")
                print(f"  Affected scenes: {bad_indices}")
                for k, lbl in loss_components:
                    if k in out:
                        print(f"  {lbl}: {out[k]}")

                if log_images:
                    log_debug_images(opt, accelerator, data, out, epoch, iteration, writer)
                scheduler.step()
                return None, True

        # Check for loss spikes
        loss_per_scene = out["loss_per_scene"]
        spike_mask = loss_per_scene > opt.safe_mode_loss_threshold
        if spike_mask.any():
            spike_indices = torch.where(spike_mask)[0].tolist()
            print(f"[SAFETY] Loss spike detected at iteration {iteration}! Skipping batch.")
            print(f"  Threshold: {opt.safe_mode_loss_threshold}, Affected scenes: {spike_indices}")
            for k, lbl in loss_components:
                if k in out:
                    print(f"  {lbl}: {out[k]}")

            if log_images:
                log_debug_images(opt, accelerator, data, out, epoch, iteration, writer)
            scheduler.step()
            return None, True

        # Safe mode: Check for NaN/Inf in gradients BEFORE clipping
        has_bad_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"[SAFETY] NaN gradient detected in {name} at iteration {iteration}! Skipping batch.")
                    has_bad_grad = True
                    break
                if torch.isinf(param.grad).any():
                    print(f"[SAFETY] Inf gradient detected in {name} at iteration {iteration}! Skipping batch.")
                    has_bad_grad = True
                    break

        if has_bad_grad:
            optimizer.zero_grad()  # Clear bad gradients
            scheduler.step()
            return None, True

    # Clip gradients
    grad_norm = accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)

    # Check if gradient norm is too large
    if grad_norm > opt.grad_norm_cap:
        print(f"Grad norm {grad_norm.item()} is too large! Skipping")
        scheduler.step()
        should_skip = True
        return None, should_skip

    return grad_norm, should_skip


def setup_workspace_and_status(opt, accelerator):
    """Setup workspace directory and check for existing completion."""
    status_dir = os.path.join(opt.workspace, "status")
    complete_file = os.path.join(status_dir, "COMPLETE")

    if accelerator.is_main_process:
        os.makedirs(status_dir, exist_ok=True)
        if os.path.exists(complete_file):
            raise RuntimeError(
                f"Found existing COMPLETE file at {complete_file}, remove it if you want to run the job again"
            )


def load_checkpoint_and_resume(opt, accelerator):
    """Load checkpoint and resume training state."""
    epoch_start = 0
    wandb_run_id = None

    if os.path.exists(f"{opt.workspace}/model.safetensors") and os.path.exists(f"{opt.workspace}/metadata.json"):
        if accelerator.is_main_process:
            print(f"Resuming from {opt.workspace}/model.safetensors")
        opt.resume = f"{opt.workspace}/model.safetensors"

        with open(f"{opt.workspace}/metadata.json") as f:
            dc = json.load(f)
            epoch_start = dc["epoch"] + 1
            if "wandb_run_id" in dc:
                wandb_run_id = dc["wandb_run_id"]

    return epoch_start, wandb_run_id


def setup_wandb(opt, accelerator, epoch_start, wandb_run_id):
    """Setup wandb logging."""
    if not accelerator.is_main_process or not opt.use_wandb:
        return None, None

    import wandb

    summary_writer_cls = _get_summary_writer_cls()

    run_name = datetime.datetime.now().strftime("%b %d, %I:%M%p")

    # Initialize wandb - resume if we have a run_id, otherwise create new run
    if wandb_run_id and epoch_start > 0:
        wandb.init(
            config=asdict(opt), project=opt.project_name, group=opt.experiment_name, id=wandb_run_id, resume="must"
        )
        print(f"Resuming wandb run {wandb_run_id}")
    else:
        wandb.init(
            config=asdict(opt),
            project=opt.project_name,
            group=opt.experiment_name,
            name=f"{opt.experiment_name} {run_name}",
        )

    # Store wandb run id for potential future resuming
    wandb_run_id = wandb.run.id
    print(f"wandb run id: {wandb.run.id}")

    tensorboard_root_dir = f"{opt.out_dir}/{opt.experiment_name}" if opt.experiment_name else None
    wandb.tensorboard.patch(root_logdir=tensorboard_root_dir, save=False)
    writer = summary_writer_cls(log_dir=tensorboard_root_dir)
    print(f"tensorboard root dir: {tensorboard_root_dir}")

    return wandb_run_id, writer


def load_model_checkpoint(opt, model, accelerator, epoch_start):
    """Load model checkpoint with tolerance for shape mismatches."""
    if opt.resume is None or opt.resume == "None":
        return

    if opt.resume.endswith("safetensors"):
        ckpt = load_file(opt.resume, device="cpu")
    else:
        ckpt = torch.load(opt.resume, map_location="cpu")

    # tolerant load (only load matching shapes)
    state_dict = model.state_dict()
    for k, v in ckpt.items():
        if k in state_dict:
            if state_dict[k].shape == v.shape:
                state_dict[k].copy_(v)
            else:
                accelerator.print(
                    f"[WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored."
                )
        else:
            accelerator.print(f"[WARN] unexpected param {k}: {v.shape}")

    if opt.init_tokens_from_existing and epoch_start == 0:
        _initialize_tokens_from_existing(ckpt, state_dict, accelerator)

    if opt.init_dynamic_tokens_from_static and epoch_start == 0 and "gs_tokens" in ckpt:
        _initialize_dynamic_tokens_from_static(ckpt, state_dict, accelerator)


def _initialize_tokens_from_existing(ckpt, state_dict, accelerator):
    """Initialize tokens from existing checkpoint."""
    with torch.no_grad():
        for token_type in ["gs_tokens", "gs_tokens_dynamic"]:
            if token_type not in ckpt:
                continue
            pretrained_tokens = ckpt[token_type]
            current_tokens = state_dict[token_type]
            N_old = pretrained_tokens.shape[0]
            N_new = current_tokens.shape[0]

            if N_new != N_old:
                accelerator.print(
                    f"[INFO] Initializing {token_type} from pretrained tokens, N_old: {N_old}, N_new: {N_new}"
                )

            if N_new <= N_old:
                # Downsample / take subset if fewer tokens
                idx = torch.linspace(0, N_old - 1, N_new).long()
                current_tokens.copy_(pretrained_tokens[idx])
            else:
                # Copy existing tokens first
                current_tokens[:N_old].copy_(pretrained_tokens)

                # Initialize additional tokens by sampling from pretrained tokens + noise
                extra_tokens = current_tokens[N_old:]
                repeat_factor = (extra_tokens.shape[0] + N_old - 1) // N_old

                expanded = pretrained_tokens.repeat((repeat_factor, 1))[: extra_tokens.shape[0]]
                noise = 0.01 * torch.randn_like(expanded)  # small perturbation
                extra_tokens.copy_(expanded + noise)


def _initialize_dynamic_tokens_from_static(ckpt, state_dict, accelerator):
    """Initialize dynamic tokens from static tokens."""
    accelerator.print("[INFO] Initializing dynamic tokens from static tokens")

    with torch.no_grad():
        static_tokens = ckpt["gs_tokens"]  # shape: [N_static, D]
        dynamic_tokens = state_dict["gs_tokens_dynamic"]  # shape: [N_dynamic, D]
        N_dynamic = dynamic_tokens.shape[0]
        N_static = static_tokens.shape[0]

        if N_dynamic == N_static:
            # direct copy
            dynamic_tokens.copy_(static_tokens)
        elif N_dynamic > N_static:
            # replicate or pad
            repeat_factor = (N_dynamic + N_static - 1) // N_static
            dynamic_tokens.copy_(static_tokens.repeat((repeat_factor, 1))[:N_dynamic])
        else:
            # random subset if fewer dynamic tokens
            idx = torch.randperm(N_static)[:N_dynamic]
            dynamic_tokens.copy_(static_tokens[idx])


def setup_optimizer_and_scheduler(opt, model, iters_per_epch, accelerator, epoch_start):
    """Setup optimizer and scheduler."""
    max_lr = opt.lr
    gs_token_lr = max_lr * opt.gs_token_lr_multiplier

    # Filter parameters based on gs_token_tuning mode
    if opt.test_time_gs_token_tuning:
        # Assert that model has gs_tokens
        has_gs_tokens = any("gs_token" in name for name, _ in model.named_parameters())
        assert has_gs_tokens, "Model must have gs_token parameters when test_time_gs_token_tuning is enabled"

        # Freeze all non-gs_token parameters to save memory
        params_to_optimize = []
        num_frozen = 0
        for name, p in model.named_parameters():
            if "gs_token" in name:
                p.requires_grad = True
                params_to_optimize.append(p)
            else:
                p.requires_grad = False
                num_frozen += 1

        if accelerator.is_main_process:
            num_total_params = sum(p.numel() for p in model.parameters())
            num_tuning_params = sum(p.numel() for p in params_to_optimize)
            print(
                f"[INFO] Test Time GS Token Tuning Mode: Optimizing {len(params_to_optimize)} parameters "
                f"({num_tuning_params:,} / {num_total_params:,} total parameters)"
            )
            print(f"[INFO] Frozen {num_frozen} parameter tensors (requires_grad=False)")
            for name, p in model.named_parameters():
                if "gs_token" in name:
                    print(f"  - {name}: {p.shape}")
    else:
        params_to_optimize = model.parameters()

    # Separate parameters into decay and no_decay groups (optional)
    if opt.no_weight_decay_for_1d_params:
        # Remove weight decay from 1D parameters (layernorm, biases)
        all_param_dict = {name: param for name, param in model.named_parameters()}
        # Filter out those that do not require grad
        optimized_param_dict = {name: param for name, param in all_param_dict.items() if param.requires_grad}

        # Separate parameters into: gs_tokens (with decay), decay_params, nodecay_params
        gs_token_decay_params, gs_token_nodecay_params = [], []
        decay_params, nodecay_params = [], []

        for name, param in optimized_param_dict.items():
            # Check if this is a gs_token parameter
            is_gs_token = "gs_token" in name

            # 1D parameters (biases, layernorm weights/biases) should not have weight decay
            if param.dim() == 1 or getattr(param, "_no_weight_decay", False):
                if is_gs_token:
                    gs_token_nodecay_params.append(param)
                else:
                    nodecay_params.append(param)
            else:
                if is_gs_token:
                    gs_token_decay_params.append(param)
                else:
                    decay_params.append(param)

        # Create optimizer groups with different learning rates
        optim_groups = []
        if len(gs_token_decay_params) > 0:
            optim_groups.append({"params": gs_token_decay_params, "weight_decay": 0.05, "lr": gs_token_lr})
        if len(gs_token_nodecay_params) > 0:
            optim_groups.append({"params": gs_token_nodecay_params, "weight_decay": 0.0, "lr": gs_token_lr})
        if len(decay_params) > 0:
            optim_groups.append({"params": decay_params, "weight_decay": 0.05, "lr": max_lr})
        if len(nodecay_params) > 0:
            optim_groups.append({"params": nodecay_params, "weight_decay": 0.0, "lr": max_lr})

        if accelerator.is_main_process:
            num_gs_token_decay = sum(p.numel() for p in gs_token_decay_params)
            num_gs_token_nodecay = sum(p.numel() for p in gs_token_nodecay_params)
            num_decay = sum(p.numel() for p in decay_params)
            num_nodecay = sum(p.numel() for p in nodecay_params)

            if num_gs_token_decay + num_gs_token_nodecay > 0:
                print(
                    f"[INFO] GS Token params (lr={gs_token_lr:.2e}): {len(gs_token_decay_params)} tensors with decay ({num_gs_token_decay:,} params), "
                    f"{len(gs_token_nodecay_params)} tensors without decay ({num_gs_token_nodecay:,} params)"
                )
            if num_decay + num_nodecay > 0:
                print(
                    f"[INFO] Other params (lr={max_lr:.2e}): {len(decay_params)} tensors with decay ({num_decay:,} params), "
                    f"{len(nodecay_params)} tensors without decay ({num_nodecay:,} params)"
                )
    else:
        # Apply weight decay to all parameters, but separate gs_tokens by learning rate
        all_param_dict = {name: param for name, param in model.named_parameters()}
        optimized_param_dict = {name: param for name, param in all_param_dict.items() if param.requires_grad}

        gs_token_params = []
        other_params = []
        for name, param in optimized_param_dict.items():
            if "gs_token" in name:
                gs_token_params.append(param)
            else:
                other_params.append(param)

        optim_groups = []
        if len(gs_token_params) > 0:
            optim_groups.append({"params": gs_token_params, "weight_decay": 0.05, "lr": gs_token_lr})
        if len(other_params) > 0:
            optim_groups.append({"params": other_params, "weight_decay": 0.05, "lr": max_lr})

        if accelerator.is_main_process:
            num_gs_token = sum(p.numel() for p in gs_token_params)
            num_other = sum(p.numel() for p in other_params)
            if num_gs_token > 0:
                print(f"[INFO] GS Token params (lr={gs_token_lr:.2e}): {num_gs_token:,} parameters")
            if num_other > 0:
                print(f"[INFO] Other params (lr={max_lr:.2e}): {num_other:,} parameters")

    optimizer = torch.optim.AdamW(optim_groups, lr=max_lr, betas=(0.9, 0.95), fused=True)

    # scheduler (per-iteration)
    if opt.lr_scheduler == "constant":
        # Constant learning rate scheduler
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
        if accelerator.is_main_process:
            print(f"[INFO] Using constant learning rate scheduler with lr={max_lr}")
    else:
        # OneCycleLR scheduler (default)
        total_steps = opt.num_epochs * iters_per_epch * accelerator.state.num_processes
        pct_start = opt.pct_start_steps * accelerator.state.num_processes / total_steps
        pct_start = min(
            pct_start, 0.99
        )  # clip when running with small datasets (local debugging) as scheduler raises with pct_start > 1.0
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            final_div_factor=opt.final_div_factor,
        )
        if accelerator.is_main_process:
            print(f"[INFO] Using OneCycleLR scheduler with max_lr={max_lr}")

    if epoch_start > 0:
        optimizer.load_state_dict(torch.load(os.path.join(opt.workspace, "optimizer.pth"), map_location="cpu"))
        scheduler.load_state_dict(torch.load(os.path.join(opt.workspace, "scheduler.pth")))

    return optimizer, scheduler


def save_checkpoint(opt, accelerator, model, optimizer, scheduler, epoch, wandb_run_id):
    """Save model checkpoint and metadata."""
    accelerator.wait_for_everyone()
    accelerator.save_model(model, opt.workspace)
    accelerator.save_model(model, os.path.join(opt.workspace, "backup"))

    if accelerator.is_main_process:
        torch.save(optimizer.state_dict(), os.path.join(opt.workspace, "optimizer.pth"))
        torch.save(scheduler.state_dict(), os.path.join(opt.workspace, "scheduler.pth"))

        metadata = {"epoch": epoch}
        if wandb_run_id:
            metadata["wandb_run_id"] = wandb_run_id

        with open(f"{opt.workspace}/metadata.json", "w") as f:
            json.dump(metadata, f)

        torch.save(optimizer.state_dict(), os.path.join(opt.workspace, "backup", "optimizer.pth"))
        torch.save(scheduler.state_dict(), os.path.join(opt.workspace, "backup", "scheduler.pth"))
        with open(f"{opt.workspace}/backup/metadata.json", "w") as f:
            json.dump(metadata, f)


def log_debug_images(opt, accelerator, data, out, epoch, i, writer):
    # Ensure images directory exists
    if not accelerator.is_main_process:
        print("Skipping debug image logging on non-main process")
        return

    os.makedirs(f"{opt.workspace}/images", exist_ok=True)

    gt_images = data["images_output"].detach().cpu().numpy()  # [B, V, 3, output_size, output_size]
    pred_images = np.clip(out["images_pred"].detach().cpu().numpy(), 0, 1)  # [B, V, 3, output_size, output_size]
    gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(
        -1, gt_images.shape[1] * gt_images.shape[4], 3
    )  # [B*output_size, V*output_size, 3]
    imageio.imwrite(
        f"{opt.workspace}/images/debug_gt_images_{epoch}_{i}.jpg", (np.clip(gt_images, 0, 1) * 255).astype(np.uint8)
    )

    pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[4], 3)
    imageio.imwrite(
        f"{opt.workspace}/images/debug_pred_images_{epoch}_{i}.jpg", (np.clip(pred_images, 0, 1) * 255).astype(np.uint8)
    )

    if opt.use_wandb:
        writer.add_image("image/debug_gt", gt_images.clip(0, 1.0), epoch, dataformats="HWC")
        writer.add_image("image/debug_pred", pred_images.clip(0, 1.0), epoch, dataformats="HWC")
    print("Logged debug images on main process")


def log_training_images(opt, accelerator, data, out, epoch, i, writer, is_train=True):
    """Log training/evaluation images or videos."""
    if not accelerator.is_main_process:
        return

    prefix = "train" if is_train else "eval"

    # Ensure images directory exists
    os.makedirs(f"{opt.workspace}/images", exist_ok=True)

    gt_images = data["images_output"].detach().cpu().numpy()  # [B, V, 3, output_size, output_size]
    pred_images = np.clip(out["images_pred"].detach().cpu().numpy(), 0, 1)  # [B, V, 3, output_size, output_size]

    if opt.log_videos:
        # Create videos: each view becomes a frame
        # Process each batch item separately
        B, V, C, H, W = gt_images.shape

        for b in range(B):
            # GT video: [V, H, W, C]
            gt_frames = gt_images[b].transpose(0, 2, 3, 1)  # [V, C, H, W] -> [V, H, W, C]
            gt_frames = (np.clip(gt_frames, 0, 1) * 255).astype(np.uint8)

            # Pred video: [V, H, W, C]
            pred_frames = pred_images[b].transpose(0, 2, 3, 1)  # [V, C, H, W] -> [V, H, W, C]
            pred_frames = (np.clip(pred_frames, 0, 1) * 255).astype(np.uint8)

            # Save videos
            if B == 1:
                gt_path = f"{opt.workspace}/images/{prefix}_gt_video_{epoch}_{i}.mp4"
                pred_path = f"{opt.workspace}/images/{prefix}_pred_video_{epoch}_{i}.mp4"
            else:
                gt_path = f"{opt.workspace}/images/{prefix}_gt_video_{epoch}_{i}_b{b}.mp4"
                pred_path = f"{opt.workspace}/images/{prefix}_pred_video_{epoch}_{i}_b{b}.mp4"

            imageio.mimwrite(gt_path, gt_frames, fps=opt.log_video_fps, quality=8)
            imageio.mimwrite(pred_path, pred_frames, fps=opt.log_video_fps, quality=8)

        # For tensorboard/wandb, still log the first frame as an image
        if opt.use_wandb:
            gt_first_frame = gt_images[0, 0].transpose(1, 2, 0)  # [H, W, C]
            pred_first_frame = pred_images[0, 0].transpose(1, 2, 0)  # [H, W, C]
            writer.add_image(f"image/{prefix}_gt", np.clip(gt_first_frame, 0, 1.0), epoch, dataformats="HWC")
            writer.add_image(f"image/{prefix}_pred", np.clip(pred_first_frame, 0, 1.0), epoch, dataformats="HWC")
    else:
        # Original behavior: horizontally stack views
        gt_images = gt_images.transpose(0, 3, 1, 4, 2).reshape(
            -1, gt_images.shape[1] * gt_images.shape[4], 3
        )  # [B*output_size, V*output_size, 3]
        imageio.imwrite(
            f"{opt.workspace}/images/{prefix}_gt_images_{epoch}_{i}.jpg",
            (np.clip(gt_images, 0, 1) * 255).astype(np.uint8),
        )

        pred_images = pred_images.transpose(0, 3, 1, 4, 2).reshape(-1, pred_images.shape[1] * pred_images.shape[4], 3)
        imageio.imwrite(
            f"{opt.workspace}/images/{prefix}_pred_images_{epoch}_{i}.jpg",
            (np.clip(pred_images, 0, 1) * 255).astype(np.uint8),
        )

        if opt.use_wandb:
            writer.add_image(f"image/{prefix}_gt", gt_images.clip(0, 1.0), epoch, dataformats="HWC")
            writer.add_image(f"image/{prefix}_pred", pred_images.clip(0, 1.0), epoch, dataformats="HWC")


def should_stop_training(opt, accelerator, start_time, epoch, epoch_start):
    """Check if training should stop due to time constraints."""
    should_exit = False
    if accelerator.is_main_process:
        time_elapsed_seconds = time.time() - start_time
        time_per_epoch = time_elapsed_seconds / (epoch - epoch_start + 1)
        time_left = opt.max_training_time_seconds - time_elapsed_seconds

        print(f"[INFO] {time_per_epoch:.2f} seconds per epoch, {time_left:.2f} seconds left.")
        if time_left <= 2 * time_per_epoch:  # Not enough time left
            should_exit = True
            print(f"[INFO] Stopping training at epoch {epoch} because we don't have enough time left.")
        else:
            print("[INFO] Continuing to next epoch.")

    # Broadcast decision to all processes using accelerate's built-in utilities
    should_exit = bool(accelerator.gather(torch.tensor([should_exit], device=accelerator.device)).sum().item() > 0)
    return should_exit


def train_epoch(
    opt,
    accelerator,
    model,
    optimizer,
    scheduler,
    train_dataloader,
    iters_per_epch,
    epoch,
    writer,
    start_time,
    train_dataset,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_psnr = 0
    log_time = time.time()
    total_skipped_batches = 0

    def train_step(data, should_log_images, iteration):
        """Execute a single training step and return metrics.

        Returns:
            tuple: (loss_value, psnr_value, should_skip)
            - loss_value: detached loss tensor
            - loss_value_detailed: dictionary containing detailed loss values
            - psnr_value: detached psnr tensor
            - should_skip: boolean indicating if step was skipped due to large grad norm
        """
        optimizer.zero_grad()

        out = model(data)
        loss = out["loss"]
        psnr = out["psnr"]
        accelerator.backward(loss)

        # Perform safety checks and gradient clipping
        grad_norm, should_skip = check_gradients_and_clip(
            model, optimizer, scheduler, accelerator, opt, out, data, epoch, iteration, writer
        )

        # If batch should be skipped, return early
        if should_skip:
            return None, None, None, True

        optimizer.step()

        # Safe mode: Check parameters after update for extreme values
        if opt.safe_mode:
            has_bad_params = False
            for name, param in model.named_parameters():
                if param is not None:
                    if torch.isnan(param).any():
                        print(
                            f"[SAFETY] NaN in parameter {name} after update at iteration {iteration}! This should not happen."
                        )
                        has_bad_params = True
                        break
                    if torch.isinf(param).any():
                        print(
                            f"[SAFETY] Inf in parameter {name} after update at iteration {iteration}! This should not happen."
                        )
                        has_bad_params = True
                        break
                    # Check for extreme parameter values
                    param_max = param.abs().max()
                    if param_max > opt.safe_mode_param_threshold:
                        print(
                            f"[SAFETY] Extreme parameter value in {name}: {param_max.item():.2e} (threshold: {opt.safe_mode_param_threshold:.2e}) at iteration {iteration}!"
                        )
                        has_bad_params = True
                        break

            if has_bad_params:
                print("[SAFETY] Bad parameters detected! Training may be unstable. Consider loading from checkpoint.")

        scheduler.step()

        loss_value = loss.detach()
        psnr_value = psnr.detach()
        # detailed loss values (only include losses that are present)
        loss_value_detailed = {
            "loss_mse": out["loss_mse"].detach(),
        }
        if "loss_ssim" in out:
            loss_value_detailed["loss_ssim"] = out["loss_ssim"].detach()
        if "loss_visibility" in out:
            loss_value_detailed["loss_visibility"] = out["loss_visibility"].detach()

        # Log images if needed, then let out and data be dereferenced
        if should_log_images:
            log_training_images(opt, accelerator, data, out, epoch, iteration, writer, is_train=True)

        return loss_value, loss_value_detailed, psnr_value, should_skip

    train_dataset.set_rng_epoch(epoch)
    if accelerator.is_main_process:
        print(f"[INFO] Setting RNG epoch to {epoch}")

    for i, data in enumerate(iter(train_dataloader)):
        if i >= opt.max_iters_per_epoch:
            break

        # Determine if we need to log images this iteration
        should_log_images = accelerator.is_main_process and (i % 100 == 0)

        with accelerator.accumulate(model):
            loss_value, loss_value_detailed, psnr_value, should_skip = train_step(data, should_log_images, i)

            if should_skip:
                total_skipped_batches += 1
                continue

            total_loss += loss_value
            total_psnr += psnr_value

            if opt.use_wandb and accelerator.is_main_process:
                writer.add_scalar("psnr/train_iteration", psnr_value.item(), epoch * iters_per_epch + i)
                writer.add_scalar("loss/train_iteration", loss_value.item(), epoch * iters_per_epch + i)
                try:
                    writer.add_scalar("time/train_iteration", time.time() - start_time, epoch * iters_per_epch + i)
                    for key, value in loss_value_detailed.items():
                        writer.add_scalar(
                            f"loss_detailed/{key}/train_iteration", value.item(), epoch * iters_per_epch + i
                        )
                except Exception:
                    pass

        if accelerator.is_main_process:
            # logging
            if i % 10 == 0:
                mem_free, mem_total = torch.cuda.mem_get_info()
                print(
                    f"[INFO] {i}/{iters_per_epch} mem: {(mem_total - mem_free) / 1024**3:.2f}/{mem_total / 1024**3:.2f}G lr: {scheduler.get_last_lr()[0]:.10f} loss: {loss_value.item():.6f} time: {time.time() - log_time:.6f}"
                )
                log_time = time.time()

    total_loss = accelerator.gather_for_metrics(total_loss).mean()
    total_psnr = accelerator.gather_for_metrics(total_psnr).mean()

    if accelerator.is_main_process:
        total_loss /= iters_per_epch
        total_psnr /= iters_per_epch

        # Log training stats including skipped batches
        skip_rate = (total_skipped_batches / iters_per_epch) * 100 if iters_per_epch > 0 else 0
        log_msg = f"[train] epoch: {epoch} loss: {total_loss.item():.6f} psnr: {total_psnr.item():.4f}"
        if total_skipped_batches > 0:
            log_msg += f" | skipped: {total_skipped_batches}/{iters_per_epch} ({skip_rate:.1f}%)"
        accelerator.print(log_msg)

        if opt.use_wandb:
            writer.add_scalar("psnr/train", total_psnr.item(), epoch)
            writer.add_scalar("loss/train", total_loss.item(), epoch)
            if total_skipped_batches > 0:
                writer.add_scalar("safety/skipped_batches", total_skipped_batches, epoch)
                writer.add_scalar("safety/skip_rate", skip_rate, epoch)


def log_gaussian_histograms(opt, all_gaussians, epoch, writer):
    """Log histograms of Gaussian properties to wandb."""
    # Concatenate all gaussians: [B, N, 14] -> [B*N, 14]
    gaussians = Gaussians.from_raw(torch.cat(all_gaussians, dim=0).reshape(-1, 14))

    # Log position histograms (x, y, z separately)
    # Clamp to bin range so out-of-bounds values appear in extreme bins
    pos_bins = torch.linspace(-25, 25, 100)
    for i, label in enumerate("xyz"):
        writer.add_histogram(
            f"gaussian/pos_{label}", gaussians.xyz[:, i].clamp(-25, 25), global_step=epoch, bins=pos_bins
        )

    # Log opacity histogram [0, 1]
    opacity_bins = torch.linspace(0, 1, 100)
    writer.add_histogram(
        "gaussian/opacity", gaussians.opacity.flatten().clamp(0, 1), global_step=epoch, bins=opacity_bins
    )

    # Log scale histograms [opt.scale_min, opt.scale_cap]
    scale_bins = torch.linspace(opt.scale_min, opt.gaussian_scale_cap, 100)
    for i, label in enumerate("xyz"):
        writer.add_histogram(
            f"gaussian/scale_{label}",
            gaussians.scaling[:, i].clamp(opt.scale_min, opt.gaussian_scale_cap),
            global_step=epoch,
            bins=scale_bins,
        )

    # Log rotation histograms [-1, 1] (quaternion components)
    rotation_bins = torch.linspace(-1, 1, 100)
    for i, label in enumerate("wxyz"):
        writer.add_histogram(
            f"gaussian/rotation_{label}", gaussians.rotation[:, i].clamp(-1, 1), global_step=epoch, bins=rotation_bins
        )

    # Log RGB histograms [0, 1]
    rgb_bins = torch.linspace(0, 1, 100)
    for i, label in enumerate("rgb"):
        writer.add_histogram(f"gaussian/rgb_{label}", gaussians.rgb[:, i].clamp(0, 1), global_step=epoch, bins=rgb_bins)


def evaluate_epoch(opt, accelerator, model, test_dataloader, epoch, writer):
    """Evaluate for one epoch."""
    with torch.inference_mode():
        model.eval()

        total_psnr = 0
        all_gaussians = []
        for i, data in enumerate(iter(test_dataloader)):
            out = model(data)
            psnr = out["psnr"]
            total_psnr += psnr.detach()

            # Collect gaussians for histogram logging
            if accelerator.is_main_process:
                all_gaussians.append(out["gaussians"].detach().cpu())

            # save some images
            log_training_images(opt, accelerator, data, out, epoch, i, writer, is_train=False)

        torch.cuda.empty_cache()

        total_psnr = accelerator.gather_for_metrics(total_psnr).mean()
        if accelerator.is_main_process:
            total_psnr /= len(test_dataloader)
            accelerator.print(f"[eval] epoch: {epoch} psnr: {total_psnr:.4f}")

            if opt.use_wandb:
                writer.add_scalar("psnr/eval", total_psnr.item(), epoch)

                # Log Gaussian property histograms
                if len(all_gaussians) > 0:
                    log_gaussian_histograms(opt, all_gaussians, epoch, writer)


def main():
    start_time = time.time()
    opt = tyro.cli(AllConfigs)

    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision, gradient_accumulation_steps=opt.gradient_accumulation_steps
    )

    # Setup workspace and status
    setup_workspace_and_status(opt, accelerator)

    # Load checkpoint and resume
    epoch_start, wandb_run_id = load_checkpoint_and_resume(opt, accelerator)

    # Setup wandb
    wandb_run_id, writer = setup_wandb(opt, accelerator, epoch_start, wandb_run_id)

    if accelerator.is_main_process:
        print(opt)

        config_save_path = os.path.join(opt.workspace, "config.yaml")
        if not os.path.exists(config_save_path):
            with open(config_save_path, "w") as f:
                f.write(tyro.extras.to_yaml(opt))
            print(f"[INFO] Config saved to {config_save_path=}")

    # model
    model = model_registry[opt.model_type](opt)

    # Load model checkpoint
    load_model_checkpoint(opt, model, accelerator, epoch_start)

    # Data
    train_dataloader, test_dataloader, train_dataset, test_dataset = get_multi_dataloader(opt, accelerator)

    iters_per_epch = min(len(train_dataloader), opt.max_iters_per_epoch)

    # Optimizer and scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(opt, model, iters_per_epch, accelerator, epoch_start)

    # accelerate
    model, optimizer, scheduler, train_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader, test_dataloader
    )

    # loop
    os.makedirs(opt.workspace, exist_ok=True)

    evaluate_epoch(opt, accelerator, model, test_dataloader, epoch_start - 1, writer)

    epoch = epoch_start
    while epoch < opt.num_epochs:
        # train
        train_epoch(
            opt,
            accelerator,
            model,
            optimizer,
            scheduler,
            train_dataloader,
            iters_per_epch,
            epoch,
            writer,
            start_time,
            train_dataset,
        )

        # checkpoint
        save_checkpoint(opt, accelerator, model, optimizer, scheduler, epoch, wandb_run_id)

        if opt.skip_eval:
            epoch += 1
            continue

        # eval
        evaluate_epoch(opt, accelerator, model, test_dataloader, epoch, writer)

        # check if we want to continue with next epoch or stop and reschedule
        if should_stop_training(opt, accelerator, start_time, epoch, epoch_start):
            # Exit without writing COMPLETE file
            # The job will be requeued by the SLURM script
            accelerator.wait_for_everyone()
            return

        epoch += 1

    # If we get here, we've completed all epochs
    # Signal successful completion
    if accelerator.is_main_process:
        status_dir = os.path.join(opt.workspace, "status")
        with open(os.path.join(status_dir, "COMPLETE"), "w") as f:
            f.write(f"Training completed successfully after {opt.num_epochs} epochs")
        print(f"[INFO] Training completed successfully after {opt.num_epochs} epochs")

    accelerator.wait_for_everyone()
    return
