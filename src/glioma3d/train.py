"""
Training script for the multi-task 3D U-Net (segmentation + tumour localisation).

Reported model (runs/final_c24):

    python -m glioma3d.train --output-dir runs/final_c24

The defaults below (base_channels 24, localisation weight 0.3, local graph modules on, seed 42,
40 epochs, AdamW lr 3e-4 with cosine annealing, 96^3 patches, batch size 1 x 4 gradient
accumulation steps, AMP) are the settings of the reported model.

Data: patient-level train/val split from samples_clean.json (see glioma3d.prep.build_samples) and the
pre-processed cache (see glioma3d.prep.preprocess_cache). The test split is never loaded here.

Note: cudnn.benchmark is enabled for speed, so runs are not bit-wise reproducible.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from glioma3d.model import MultiTaskUNet3D


# ============================================================
# 1. Configuration
# ============================================================

SEED = 42

# Hardware: NVIDIA GTX 1650, 4 GB VRAM
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
NUM_WORKERS = 4         # Parallel data loading (use 2 if RAM is low, 0 if errors)
USE_AMP = True
MAX_GRAD_NORM = 1.0
MAX_SKIPPED_PER_EPOCH = 20

EPOCHS = 40
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-5

# Multi-task loss weights
SEGMENTATION_LOSS_WEIGHT = 1.0
LOCALIZATION_LOSS_WEIGHT = 0.3
USE_GRAPH = True  # False -> ablation without the local graph message-passing modules

# Model configuration
IN_CHANNELS = 4
NUM_CLASSES = 4
BASE_CHANNELS = 24   # base width of the network (24 for the reported model; check the GPU peak in the log)
DROPOUT = 0.05

# Checkpoints and logs
OUTPUT_DIR = Path("runs/brain_tumor_experiment")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"

EARLY_STOPPING_PATIENCE = 15
MIN_DELTA = 1e-4

# Optional: resume from an existing checkpoint
RESUME_CHECKPOINT: Optional[str] = None

# Live console line is refreshed every N training steps
LOG_EVERY = 10

# Set True to enable TensorBoard when tensorboard is installed
ENABLE_TENSORBOARD = True

def format_time(seconds: float) -> str:
    seconds = int(max(0, seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{secs:02d}s"


def gpu_memory_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return 0.0


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable model parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# ============================================================
# 2. Reproducibility
# ============================================================

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # The input size is fixed (96^3 patches), so cudnn.benchmark is faster;
    # the price is that results are not bit-wise reproducible.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ============================================================
# 3. Device and mixed precision
# ============================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_grad_scaler(device: torch.device):
    """FP16 gradient scaling is used only on CUDA."""
    enabled = device.type == "cuda" and USE_AMP
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(device: torch.device):
    """Context manager compatible with common PyTorch versions."""
    if device.type == "cuda" and USE_AMP:
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
        )
    return torch.autocast(
        device_type=device.type,
        enabled=False,
    )


# ============================================================
# 4. Batch preparation & Safe Label Handling
# ============================================================

def get_batch_tensors(
    batch,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepares and validates image and mask tensors from batch."""
    if isinstance(batch, dict):
        image_keys = ("image", "mri", "volume", "input", "x")
        mask_keys = ("segmentation", "mask", "label", "target", "y")

        image = next((batch[k] for k in image_keys if k in batch), None)
        mask = next((batch[k] for k in mask_keys if k in batch), None)

        if image is None or mask is None:
            raise KeyError(
                "Dataset batches must contain an MRI tensor "
                "and a segmentation mask. Expected keys such as "
                "'image' and 'segmentation'."
            )
    elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
        image, mask = batch[0], batch[1]
    else:
        raise TypeError(
            "Unsupported batch format. Return a dictionary or "
            "(image, segmentation) tuple from the Dataset."
        )

    image = torch.as_tensor(image).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )

    mask = torch.as_tensor(mask).to(
        device=device,
        dtype=torch.long,
        non_blocking=True,
    )

    # Note: there is deliberately no clamp here. Label mapping/validation is done in data.py
    # so that a wrong label is never hidden.

    if image.ndim == 4:
        image = image.unsqueeze(0)

    if mask.ndim == 5 and mask.shape[1] == 1:
        mask = mask[:, 0]

    if mask.ndim == 3:
        mask = mask.unsqueeze(0)

    if image.ndim != 5:
        raise ValueError(f"Expected MRI shape [B,C,D,H,W], got {tuple(image.shape)}")

    if mask.ndim != 4:
        raise ValueError(f"Expected mask shape [B,D,H,W], got {tuple(mask.shape)}")

    if image.shape[0] != mask.shape[0]:
        raise ValueError("MRI and mask batch sizes do not match.")

    if image.shape[-3:] != mask.shape[-3:]:
        raise ValueError(
            "MRI and segmentation spatial dimensions must match. "
            f"Got {tuple(image.shape[-3:])} and {tuple(mask.shape[-3:])}."
        )

    # An isfinite check on the input would force a GPU sync; NaN/Inf is caught in the loss anyway.
    return image, mask


def validate_labels(
    mask: torch.Tensor,
    num_classes: int = NUM_CLASSES,
) -> None:
    """
    No clamp and no GPU sync: labels are validated in the Dataset (on the CPU).
    """
    return None


# ============================================================
# 5. Segmentation loss: Cross-Entropy + Soft Dice
# ============================================================

class DiceCrossEntropyLoss(nn.Module):
    """Multiclass segmentation loss."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        smooth: float = 1e-5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ce = F.cross_entropy(
            logits.float(),
            target.long(),
        )

        probabilities = torch.softmax(
            logits.float(),
            dim=1,
        )

        target_one_hot = F.one_hot(
            target.long(),
            num_classes=self.num_classes,
        ).permute(0, 4, 1, 2, 3).float()

        dimensions = (0, 2, 3, 4)

        intersection = (
            probabilities * target_one_hot
        ).sum(dim=dimensions)

        denominator = (
            probabilities.sum(dim=dimensions)
            + target_one_hot.sum(dim=dimensions)
        )

        dice_per_class = (
            2.0 * intersection + self.smooth
        ) / (
            denominator + self.smooth
        )

        foreground_dice = dice_per_class[1:]
        dice_loss = 1.0 - foreground_dice.mean()

        total = (
            self.ce_weight * ce
            + self.dice_weight * dice_loss
        )

        return total, ce.detach(), dice_loss.detach()


# ============================================================
# 6. Localization target and loss
# ============================================================

def make_localization_target(
    segmentation: torch.Tensor,
    output_shape: Tuple[int, int, int],
) -> torch.Tensor:
    """Create a coarse binary tumor target from the training mask."""
    tumor = (segmentation > 0).float().unsqueeze(1)
    return F.adaptive_max_pool3d(
        tumor,
        output_size=output_shape,
    )


class LocalizationBCELoss(nn.Module):
    """Binary localization objective."""

    def __init__(self, positive_weight: float = 5.0):
        super().__init__()
        self.positive_weight = positive_weight

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        positive_weight = torch.tensor(
            self.positive_weight,
            device=logits.device,
            dtype=torch.float32,
        )

        return F.binary_cross_entropy_with_logits(
            logits.float(),
            target.float(),
            pos_weight=positive_weight,
        )


# ============================================================
# 7. Evaluation metrics
# ============================================================

@torch.no_grad()
def calculate_dice_per_class(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = NUM_CLASSES,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """Return Dice for each foreground class."""
    prediction = logits.argmax(dim=1)
    scores = []

    for class_id in range(1, num_classes):
        pred_class = prediction == class_id
        true_class = target == class_id

        intersection = (
            pred_class & true_class
        ).sum().float()

        pred_sum = pred_class.sum().float()
        true_sum = true_class.sum().float()
        denominator = pred_sum + true_sum

        dice = torch.where(
            denominator > 0,
            (2.0 * intersection + smooth) / (denominator + smooth),
            torch.full_like(denominator, float("nan")),
        )
        scores.append(dice)

    return torch.stack(scores)


# ============================================================
# 8. Checkpoint saving and loading
# ============================================================

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_val_loss: float,
    best_val_dice: float,
    patience_counter: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                scheduler.state_dict() if scheduler is not None else None
            ),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_loss": best_val_loss,
            "best_val_dice": best_val_dice,
            "patience_counter": patience_counter,
            "config": {
                "in_channels": IN_CHANNELS,
                "num_classes": NUM_CLASSES,
                "base_channels": BASE_CHANNELS,
                "dropout": DROPOUT,
                "use_graph": USE_GRAPH,
                "loc_weight": LOCALIZATION_LOSS_WEIGHT,
                "seed": SEED,
            },
        },
        path,
    )


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device: torch.device,
):
    """Resume a training run."""
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        try:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except Exception:
            # The checkpoint was written with a different scheduler; fast-forward the cosine schedule to the current epoch
            for _ in range(int(checkpoint["epoch"])):
                scheduler.step()
            print("Scheduler state incompatible (old checkpoint); cosine fast-forwarded.")

    if checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_epoch = int(checkpoint["epoch"]) + 1
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    best_val_dice = float(checkpoint.get("best_val_dice", 0.0))
    patience_counter = int(checkpoint.get("patience_counter", 0))

    return start_epoch, best_val_loss, best_val_dice, patience_counter


# ============================================================
# 9. One training epoch
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    seg_loss_fn: DiceCrossEntropyLoss,
    loc_loss_fn: LocalizationBCELoss,
    device: torch.device,
    epoch: int,
    writer=None,
) -> Dict[str, float]:

    model.train()
    optimizer.zero_grad(set_to_none=True)

    if len(loader) == 0:
        raise RuntimeError("Training DataLoader is empty.")

    # All accumulators stay on the GPU so that no sync (.item()) is needed at every step
    acc = torch.zeros(5, device=device)  # total, seg, ce, dice, loc
    processed_batches = 0
    skipped_batches = 0
    optimizer_steps = 0
    dice_sum = torch.zeros(NUM_CLASSES - 1, device=device)
    dice_cnt = torch.zeros(NUM_CLASSES - 1, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start_time = time.time()
    last_log_time, last_log_step = start_time, 0

    for step, batch in enumerate(loader):
        image, mask = get_batch_tensors(batch, device)
        validate_labels(mask)

        group_start = (
            step // GRADIENT_ACCUMULATION_STEPS
        ) * GRADIENT_ACCUMULATION_STEPS

        group_size = min(
            GRADIENT_ACCUMULATION_STEPS,
            len(loader) - group_start,
        )

        with autocast_context(device):
            outputs = model(image)
            segmentation_logits = outputs["segmentation"]
            localization_logits = outputs["localization"]

            seg_loss, ce_loss, dice_loss = seg_loss_fn(
                segmentation_logits,
                mask,
            )

            localization_target = make_localization_target(
                mask,
                localization_logits.shape[-3:],
            )

            loc_loss = loc_loss_fn(
                localization_logits,
                localization_target,
            )

            total = (
                SEGMENTATION_LOSS_WEIGHT * seg_loss
                + LOCALIZATION_LOSS_WEIGHT * loc_loss
            )

        # If the loss of this batch is not finite (fp16 overflow or bad data) we skip the batch
        # instead of killing the whole run. One sync per step is negligible against ~3.5 s/it.
        if not bool(torch.isfinite(total.detach())):
            skipped_batches += 1
            in_ok = bool(torch.isfinite(image).all())
            print(
                f"\n[warning] non-finite loss at epoch {epoch} step {step}; batch skipped "
                f"(input finite: {in_ok}, max|x|={float(image.abs().nan_to_num(0, 0, 0).max()):.1f}) "
                f"| total skipped: {skipped_batches}",
                flush=True,
            )
            if skipped_batches > MAX_SKIPPED_PER_EPOCH:
                raise FloatingPointError(
                    f"More than {MAX_SKIPPED_PER_EPOCH} non-finite batches in epoch {epoch}; training is unstable."
                )
            del outputs, segmentation_logits, localization_logits, seg_loss, ce_loss, dice_loss, loc_loss, total
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (step + 1) == len(loader):
                # Continue the accumulation group if it holds valid gradients
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            continue

        scaler.scale(total / group_size).backward()

        processed_batches += 1
        acc += torch.stack([
            total.detach().float(), seg_loss.detach().float(), ce_loss.float(),
            dice_loss.float(), loc_loss.detach().float(),
        ])

        step_dice = calculate_dice_per_class(
            segmentation_logits.detach(), mask, NUM_CLASSES
        )
        ok = ~torch.isnan(step_dice)
        dice_sum[ok] += step_dice[ok]
        dice_cnt[ok] += 1

        should_step = (
            (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0
            or (step + 1) == len(loader)
        )

        if should_step:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=MAX_GRAD_NORM,
            )

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

        if (step + 1) % LOG_EVERY == 0 or (step + 1) == len(loader):
            # The only sync point: once every LOG_EVERY steps
            if not torch.isfinite(acc).all():
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch}, step {step}. "
                    "Inspect input intensities, labels and learning rate."
                )
            total_loss = acc[0].item()
            done = step + 1
            # Speed is measured over the recent window, not as an average since the start of the epoch,
            # so that a pause/sleep of the machine does not distort it
            now = time.time()
            sec_per_it = (now - last_log_time) / max(1, done - last_log_step)
            last_log_time, last_log_step = now, done
            eta = sec_per_it * (len(loader) - done)
            avg_dice = (dice_sum / dice_cnt.clamp(min=1)).tolist()
            current_lr = optimizer.param_groups[0]["lr"]
            dice_text = "/".join(f"{value:.2f}" for value in avg_dice)

            print(
                f"\rEp {epoch:03d}/{EPOCHS} | "
                f"{done:04d}/{len(loader):04d} ({100.0 * done / len(loader):5.1f}%) | "
                f"Loss avg {total_loss / processed_batches:.3f} | "
                f"Dice {dice_text} | "
                f"LR {current_lr:.1e} | "
                f"{sec_per_it:.2f}s/it | ETA {format_time(eta)} | "
                f"GPU {gpu_memory_gb():.1f}GB  ",
                end="",
                flush=True,
            )

            if writer is not None:
                global_step = (epoch - 1) * len(loader) + done
                writer.add_scalar("Step/loss_avg", total_loss / processed_batches, global_step)
                writer.add_scalar("Step/seg_loss_avg", acc[1].item() / processed_batches, global_step)
                writer.add_scalar("Step/loc_loss_avg", acc[4].item() / processed_batches, global_step)
                for class_id, value in enumerate(avg_dice, start=1):
                    writer.add_scalar(f"Step/train_dice_class_{class_id}", value, global_step)
                writer.add_scalar("Step/sec_per_it", sec_per_it, global_step)

    print()

    if processed_batches == 0:
        raise RuntimeError("No training batches were processed.")

    elapsed = time.time() - start_time

    acc_cpu = (acc / processed_batches).tolist()
    return {
        "loss": acc_cpu[0],
        "segmentation_loss": acc_cpu[1],
        "cross_entropy": acc_cpu[2],
        "dice_loss": acc_cpu[3],
        "localization_loss": acc_cpu[4],
        "optimizer_steps": float(optimizer_steps),
        "mean_foreground_dice": (dice_sum / dice_cnt.clamp(min=1)).mean().item(),
        "seconds": elapsed,
    }


# ============================================================
# 10. Validation epoch
# ============================================================

@torch.inference_mode()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    seg_loss_fn: DiceCrossEntropyLoss,
    loc_loss_fn: LocalizationBCELoss,
    device: torch.device,
) -> Dict[str, float]:

    model.eval()

    acc = torch.zeros(3, device=device)  # loss, seg, loc
    batch_count = 0

    dice_sum = torch.zeros(NUM_CLASSES - 1, device=device)
    dice_count = torch.zeros(NUM_CLASSES - 1, device=device)

    if len(loader) == 0:
        raise RuntimeError("Validation DataLoader is empty.")

    for batch in loader:
        image, mask = get_batch_tensors(batch, device)
        validate_labels(mask)

        with autocast_context(device):
            outputs = model(image)
            seg_logits = outputs["segmentation"]
            loc_logits = outputs["localization"]

            seg_loss, _, _ = seg_loss_fn(
                seg_logits,
                mask,
            )

            loc_target = make_localization_target(
                mask,
                loc_logits.shape[-3:],
            )

            loc_loss = loc_loss_fn(
                loc_logits,
                loc_target,
            )

            loss = (
                SEGMENTATION_LOSS_WEIGHT * seg_loss
                + LOCALIZATION_LOSS_WEIGHT * loc_loss
            )

        batch_dice = calculate_dice_per_class(
            seg_logits,
            mask,
            num_classes=NUM_CLASSES,
        )

        valid = ~torch.isnan(batch_dice)
        dice_sum[valid] += batch_dice[valid]
        dice_count[valid] += 1

        acc += torch.stack([loss.float(), seg_loss.float(), loc_loss.float()])
        batch_count += 1

        if batch_count % LOG_EVERY == 0 or batch_count == len(loader):
            if not torch.isfinite(acc).all():
                raise FloatingPointError("Non-finite validation loss detected.")
            running_dice = (dice_sum / dice_count.clamp(min=1)).mean().item()
            print(
                f"\r   Validating {batch_count:04d}/{len(loader):04d} | "
                f"Loss avg {acc[0].item() / batch_count:.3f} | "
                f"Dice avg {running_dice:.3f}  ",
                end="",
                flush=True,
            )

    print()
    dice_mean = dice_sum / dice_count.clamp(min=1)

    acc_cpu = (acc / batch_count).tolist()
    result = {
        "loss": acc_cpu[0],
        "segmentation_loss": acc_cpu[1],
        "localization_loss": acc_cpu[2],
        "mean_foreground_dice": dice_mean.mean().item(),
    }

    for class_id in range(1, NUM_CLASSES):
        result[f"dice_class_{class_id}"] = dice_mean[class_id - 1].item()

    return result


# ============================================================
# 11. Optional TensorBoard
# ============================================================

def create_tensorboard_writer():
    if not ENABLE_TENSORBOARD:
        return None

    try:
        from torch.utils.tensorboard import SummaryWriter
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        return SummaryWriter(log_dir=str(LOG_DIR))
    except ImportError:
        print(
            "TensorBoard is not installed; scalar logging "
            "will use the console and JSON history only."
        )
        return None


def log_metrics(
    writer,
    epoch: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    learning_rate: float,
) -> None:
    if writer is None:
        return

    for key, value in train_metrics.items():
        writer.add_scalar(f"Train/{key}", value, epoch)

    for key, value in val_metrics.items():
        writer.add_scalar(f"Validation/{key}", value, epoch)

    writer.add_scalar("Optimization/learning_rate", learning_rate, epoch)
    writer.flush()


# ============================================================
# 12. DataLoader integration
# ============================================================

def create_project_dataloaders(samples_json: str = "samples_clean.json", cache_dir: Optional[str] = None):
    """Build the train/val loaders with glioma3d.data.create_dataloaders."""
    from glioma3d.data import CACHE_DIR, create_dataloaders

    loaders = create_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        seed=SEED,
        samples_json=samples_json,
        cache_dir=Path(cache_dir) if cache_dir else CACHE_DIR,
    )

    if not isinstance(loaders, (tuple, list)) or len(loaders) != 2:
        raise TypeError("create_dataloaders must return (train_loader, val_loader).")

    train_loader, val_loader = loaders

    if len(train_loader) == 0:
        raise RuntimeError("Training DataLoader is empty.")

    if len(val_loader) == 0:
        raise RuntimeError("Validation DataLoader is empty.")

    return train_loader, val_loader


# ============================================================
# 13. Main training routine
# ============================================================

def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    resume_checkpoint: Optional[str] = None,
) -> None:

    set_seed(SEED)
    device = get_device()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("3D Brain Tumor Localization + Segmentation")
    print("=" * 65)
    print("Device:", device)
    print("Mixed precision:", USE_AMP and device.type == "cuda")
    print("Batch size:", BATCH_SIZE)
    print("Gradient accumulation:", GRADIENT_ACCUMULATION_STEPS)
    print("Effective batch size:", BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
    print("Epochs:", EPOCHS)
    print("Learning rate:", LEARNING_RATE)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "CUDA memory:",
            round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2),
            "GB",
        )

    model = MultiTaskUNet3D(
        in_channels=IN_CHANNELS,
        num_classes=NUM_CLASSES,
        base_channels=BASE_CHANNELS,
        dropout=DROPOUT,
        use_graph=USE_GRAPH,
    ).to(device)
    print(f"Ablation switches: use_graph={USE_GRAPH} | localization loss weight={LOCALIZATION_LOSS_WEIGHT} | seed={SEED}")

    print("Trainable parameters:", f"{count_parameters(model):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6,
    )

    scaler = create_grad_scaler(device)

    seg_loss_fn = DiceCrossEntropyLoss(
        num_classes=NUM_CLASSES,
        dice_weight=1.0,
        ce_weight=1.0,
    )

    loc_loss_fn = LocalizationBCELoss(
        positive_weight=5.0,
    )

    start_epoch = 1
    best_val_loss = float("inf")
    best_val_dice = 0.0
    patience_counter = 0

    if resume_checkpoint is not None:
        (
            start_epoch,
            best_val_loss,
            best_val_dice,
            patience_counter,
        ) = load_checkpoint(
            resume_checkpoint,
            model,
            optimizer,
            scheduler,
            scaler,
            device,
        )
        print(f"Resuming from epoch {start_epoch}; best validation loss = {best_val_loss:.5f}")

    writer = create_tensorboard_writer()
    history_path = OUTPUT_DIR / "history.json"

    if resume_checkpoint is not None and history_path.exists():
        with history_path.open("r", encoding="utf-8") as file:
            history = json.load(file)
    else:
        history = []

    run_start = time.time()

    try:
        for epoch in range(start_epoch, EPOCHS + 1):
            epoch_start = time.time()

            print(f"\nEpoch {epoch}/{EPOCHS}")
            print("-" * 65)

            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scaler=scaler,
                seg_loss_fn=seg_loss_fn,
                loc_loss_fn=loc_loss_fn,
                device=device,
                epoch=epoch,
                writer=writer,
            )

            val_metrics = validate_one_epoch(
                model=model,
                loader=val_loader,
                seg_loss_fn=seg_loss_fn,
                loc_loss_fn=loc_loss_fn,
                device=device,
            )

            current_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            epoch_seconds = time.time() - epoch_start

            log_metrics(writer, epoch, train_metrics, val_metrics, current_lr)

            record = {
                "epoch": epoch,
                "learning_rate": current_lr,
                "train": train_metrics,
                "validation": val_metrics,
                "epoch_seconds": epoch_seconds,
            }

            history.append(record)

            with history_path.open("w", encoding="utf-8") as file:
                json.dump(history, file, indent=2, ensure_ascii=False)

            save_checkpoint(
                CHECKPOINT_DIR / "last.pth",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_val_loss,
                best_val_dice,
                patience_counter,
            )

            # The main model-selection criterion is the validation Dice (best.pth); the loss is tracked separately.
            if val_metrics["loss"] < best_val_loss - MIN_DELTA:
                best_val_loss = val_metrics["loss"]
                save_checkpoint(
                    CHECKPOINT_DIR / "best_loss.pth",
                    model, optimizer, scheduler, scaler, epoch,
                    best_val_loss, best_val_dice, patience_counter,
                )

            improved = val_metrics["mean_foreground_dice"] > best_val_dice + MIN_DELTA

            if improved:
                best_val_dice = val_metrics["mean_foreground_dice"]
                patience_counter = 0

                save_checkpoint(
                    CHECKPOINT_DIR / "best.pth",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best_val_loss,
                    best_val_dice,
                    patience_counter,
                )
                print("Best checkpoint updated (by validation Dice).")
            else:
                patience_counter += 1

            epochs_done = epoch - start_epoch + 1
            remaining = (EPOCHS - epoch) * (time.time() - run_start) / epochs_done
            dice_text = " | ".join(
                f"c{class_id} {val_metrics[f'dice_class_{class_id}']:.4f}"
                for class_id in range(1, NUM_CLASSES)
            )
            print("=" * 65)
            print(f"EPOCH {epoch}/{EPOCHS} SUMMARY" + ("  ** new best **" if improved else ""))
            print(
                f"  Train | loss {train_metrics['loss']:.4f} | seg {train_metrics['segmentation_loss']:.4f} "
                f"| loc {train_metrics['localization_loss']:.4f} | Dice {train_metrics['mean_foreground_dice']:.4f}"
            )
            print(
                f"  Val   | loss {val_metrics['loss']:.4f} | seg {val_metrics['segmentation_loss']:.4f} "
                f"| loc {val_metrics['localization_loss']:.4f} | Dice {val_metrics['mean_foreground_dice']:.4f}"
            )
            print(f"  Val Dice per class | {dice_text}")
            print(
                f"  Best val loss {best_val_loss:.4f} | best Dice {best_val_dice:.4f} "
                f"| no-improve {patience_counter}/{EARLY_STOPPING_PATIENCE}"
            )
            print(
                f"  LR {current_lr:.2e} | epoch time {format_time(epoch_seconds)} "
                f"| est. remaining {format_time(remaining)}"
            )
            print("=" * 65)

            save_checkpoint(
                CHECKPOINT_DIR / "last.pth",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_val_loss,
                best_val_dice,
                patience_counter,
            )

            if device.type == "cuda":
                torch.cuda.empty_cache()

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered. Validation Dice did not improve.")
                break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        print("The latest completed epoch checkpoint is preserved.")

    except torch.cuda.OutOfMemoryError:
        print("\nCUDA out of memory. Training stopped safely.")
        print("Try smaller 3D patches or reduce BASE_CHANNELS.")
        raise

    finally:
        if writer is not None:
            writer.close()

    print("\nTraining routine finished.")
    print("Best validation loss:", best_val_loss)
    print("Best validation foreground Dice:", best_val_dice)
    print("Checkpoints:", CHECKPOINT_DIR.resolve())
    print("History:", history_path.resolve())


# ============================================================
# 14. Command-line entry point
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the joint 3D brain tumor localization and segmentation model."
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=RESUME_CHECKPOINT,
        help="Path to a training checkpoint to resume.",
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=None,
        help="Override BASE_CHANNELS (e.g. 16, 24, 32) for quick VRAM tests.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override OUTPUT_DIR so a new run does not overwrite old checkpoints.",
    )
    parser.add_argument(
        "--loc-weight",
        type=float,
        default=None,
        help="Weight of the localization loss (default 0.3). 0 = segmentation-only ablation.",
    )
    parser.add_argument(
        "--no-graph",
        action="store_true",
        help="Ablation: disable the local graph message-passing modules.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Training seed (initialisation, batch order, augmentation). The patient split is fixed in samples_clean.json.",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="samples_clean.json",
        help="samples_clean.json written by glioma3d.prep.build_samples.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Pre-processed cache directory (default: ./cache_v2 or $GLIOMA3D_CACHE_DIR).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.base_channels is not None:
        BASE_CHANNELS = args.base_channels
    if args.loc_weight is not None:
        LOCALIZATION_LOSS_WEIGHT = args.loc_weight
    if args.no_graph:
        USE_GRAPH = False
    if args.seed is not None:
        SEED = args.seed
    if args.output_dir is not None:
        OUTPUT_DIR = Path(args.output_dir)
        CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
        LOG_DIR = OUTPUT_DIR / "logs"
    train_loader, val_loader = create_project_dataloaders(args.samples, args.cache_dir)
    train(
        train_loader=train_loader,
        val_loader=val_loader,
        resume_checkpoint=args.resume,
    )