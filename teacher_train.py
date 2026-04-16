"""
teacher_train.py
----------------
Training loop for the Swin Transformer Tiny teacher model.

Pipeline
--------
1. Build model, optimiser, and LR schedule.
2. Train for ``cfg.teacher_epochs`` epochs.
   Each epoch:
     a. Forward pass → (embeddings, predictions, backbone_features)
     b. Compute TeacherTotalLoss (regression + ranking)
     c. Backward pass with optional AMP (torch.autocast)
     d. Gradient clipping
     e. Optimiser + scheduler step
3. Every ``cfg.eval_every_n_epochs`` epochs, evaluate on the val set
   and save a checkpoint if SRCC improved.
4. At the end, save the final checkpoint.

CLI
---
    python teacher_train.py            # uses default Config()
    python teacher_train.py --help     # show all options (via simple argparse)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from datasets import build_dataloaders
from evaluate import EvalResult, evaluate_model
from losses import TeacherTotalLoss
from teacher import TeacherModel, build_teacher
from utils import (
    AverageMeter,
    EarlyStopping,
    get_cosine_schedule_with_warmup,
    get_device,
    load_checkpoint,
    print_gpu_memory,
    print_model_info,
    save_checkpoint,
    set_seed,
)

# ------------------------------------------------------------------ #
# Wrapper to extract state_dict from a possibly DataParallel-wrapped model
# ------------------------------------------------------------------ #
def get_model_state_dict(model):
    return model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

# ------------------------------------------------------------------ #
# One training epoch
# ------------------------------------------------------------------ #

def train_one_epoch(
    model: TeacherModel,
    loader: DataLoader,
    criterion: TeacherTotalLoss,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    cfg: Config,
) -> Dict[str, float]:
    """
    Run a single training epoch.

    Parameters
    ----------
    model     : TeacherModel
    loader    : DataLoader  – yields dicts with 'image' (B,3,H,W) and 'mos' (B,)
    criterion : TeacherTotalLoss
    optimizer : Optimizer
    scaler    : GradScaler  – for AMP; used as identity if cfg.amp=False
    device    : torch.device
    epoch     : int  – 0-indexed
    cfg       : Config

    Returns
    -------
    dict with keys 'total', 'reg', 'rank', 'lr'  (all floats)
    """
    model.train()
    meters = {
        "total": AverageMeter("total"),
        "reg":   AverageMeter("reg"),
        "rank":  AverageMeter("rank"),
    }

    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch + 1:>3d} [train]",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    )

    for batch in pbar:
        images  = batch["image"].to(device, non_blocking=True)  # (B, 3, H, W)
        targets = batch["mos"].to(device, non_blocking=True)     # (B,)
        B = images.size(0)

        optimizer.zero_grad(set_to_none=True)

        # ---- Forward --------------------------------------------- #
        if cfg.amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _emb, predictions, _feat = model(images)         # (B,E), (B,1), (B,F)
                total, reg, rank = criterion(predictions, targets)
        else:
            _emb, predictions, _feat = model(images)
            total, reg, rank = criterion(predictions, targets)

        # ---- Backward -------------------------------------------- #
        if cfg.amp and device.type == "cuda":
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        # ---- Logging --------------------------------------------- #
        meters["total"].update(total.item(), B)
        meters["reg"].update(reg.item(),   B)
        meters["rank"].update(rank.item(), B)

        pbar.set_postfix(
            loss=f"{meters['total'].avg:.4f}",
            reg=f"{meters['reg'].avg:.4f}",
            rank=f"{meters['rank'].avg:.4f}",
        )

    pbar.close()

    current_lr = optimizer.param_groups[0]["lr"]
    return {
        "total": meters["total"].avg,
        "reg":   meters["reg"].avg,
        "rank":  meters["rank"].avg,
        "lr":    current_lr,
    }


# ------------------------------------------------------------------ #
# Main training function
# ------------------------------------------------------------------ #

def train_teacher(cfg: Config) -> Tuple[TeacherModel, EvalResult]:
    """
    Full teacher training pipeline.

    Parameters
    ----------
    cfg : Config

    Returns
    -------
    model       : TeacherModel  – best checkpoint loaded at the end
    best_result : EvalResult    – val-set metrics of the best checkpoint
    """
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    # ---- Data ------------------------------------------------------ #
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    # ---- Model ----------------------------------------------------- #
    model = build_teacher(cfg).to(device)

    if torch.cuda.device_count() > 1 and device.type == "cuda":
        print(f"[INFO] Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    print_model_info(model, name="Teacher")

    # Optionally resume from checkpoint
    start_epoch = 0
    if cfg.teacher_ckpt and os.path.isfile(cfg.teacher_ckpt):
        ckpt = load_checkpoint(cfg.teacher_ckpt, model, device=device)
        start_epoch = ckpt.get("epoch", 0)

    # ---- Optimiser + Schedule -------------------------------------- #
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.teacher_lr,
        weight_decay=cfg.teacher_weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_epochs=cfg.teacher_warmup_epochs,
        total_epochs=cfg.teacher_epochs,
    )

    scaler = GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    # ---- Loss ------------------------------------------------------- #
    criterion = TeacherTotalLoss(cfg)

    # ---- Checkpointing paths --------------------------------------- #
    ckpt_dir  = os.path.join(cfg.output_dir, "teacher")
    os.makedirs(ckpt_dir, exist_ok=True)
    last_ckpt  = os.path.join(ckpt_dir, "teacher_last.pth")
    best_ckpt  = os.path.join(ckpt_dir, "teacher_best.pth")

    best_srcc  = -1.0
    best_result: EvalResult | None = None

    # ---- Early stopping ------------------------------------------- #
    early_stopper = EarlyStopping(
        patience=cfg.early_stopping_patience,
        min_delta=cfg.early_stopping_min_delta,
        mode="max",
    )

    # ---- WandB (optional) ----------------------------------------- #
    if cfg.use_wandb:
        try:
            import wandb
            wandb.init(
                project=cfg.wandb_project,
                name=cfg.wandb_run_name or "teacher_train",
                config=vars(cfg),
            )
        except ImportError:
            print("[utils] wandb not installed; skipping.")
            cfg.use_wandb = False

    # ---------------------------------------------------------------- #
    # Training loop
    # ---------------------------------------------------------------- #
    print(f"\n{'='*60}")
    print(f"  Teacher training on {device}  |  epochs={cfg.teacher_epochs}")
    print(f"{'='*60}\n")

    epoch_bar = tqdm(
        range(start_epoch, cfg.teacher_epochs),
        desc="Epochs",
        unit="epoch",
        dynamic_ncols=True,
    )

    for epoch in epoch_bar:
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, cfg
        )
        scheduler.step()

        epoch_time = time.time() - t0
        epoch_bar.set_postfix(
            loss=f"{train_metrics['total']:.4f}",
            lr=f"{train_metrics['lr']:.1e}",
            time=f"{epoch_time:.1f}s",
        )

        if device.type == "cuda":
            print_gpu_memory()

        # ---- Periodic validation ----------------------------------- #
        if (epoch + 1) % cfg.eval_every_n_epochs == 0 or epoch == cfg.teacher_epochs - 1:
            result = evaluate_model(
                model, val_loader, device,
                image_size=cfg.image_size,
                amp=cfg.amp,
            )
            tqdm.write(
                f"\n  [VAL] Epoch {epoch+1}  "
                f"SRCC={result.srcc:.4f}  PLCC={result.plcc:.4f}  "
                f"MAE={result.mae:.4f}  "
                f"Inf={result.inference_ms:.1f}ms"
            )

            is_best = result.srcc > best_srcc
            if is_best:
                best_srcc   = result.srcc
                best_result = result
                tqdm.write(f"  ✓ New best SRCC = {best_srcc:.4f}")

            # Save checkpoint
            state = {
                "epoch":              epoch + 1,
                "model_state_dict": get_model_state_dict(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_srcc":          best_srcc,
                "config":             vars(cfg),
            }
            save_checkpoint(state, last_ckpt)
            save_checkpoint(state, last_ckpt, is_best=is_best, best_path=best_ckpt)

            if cfg.use_wandb:
                import wandb
                wandb.log({
                    "epoch":      epoch + 1,
                    "train/loss": train_metrics["total"],
                    "train/reg":  train_metrics["reg"],
                    "train/rank": train_metrics["rank"],
                    "val/srcc":   result.srcc,
                    "val/plcc":   result.plcc,
                    "val/mae":    result.mae,
                })

            # ---- Early stopping check ----------------------------- #
            if early_stopper(result.srcc):
                tqdm.write(
                    f"\n[EarlyStopping] Triggered at epoch {epoch+1}. "
                    f"Best SRCC = {best_srcc:.4f}"
                )
                break

    epoch_bar.close()

    # ---- Final evaluation on test set ------------------------------ #
    print("\n[teacher_train] Loading best checkpoint for test evaluation …")
    if os.path.isfile(best_ckpt):
        load_checkpoint(best_ckpt, model, device=device)

    test_result = evaluate_model(
        model, test_loader, device,
        image_size=cfg.image_size,
        amp=False,  # use full precision for final eval
    )
    print(f"\n[TEST] {test_result}\n")

    if cfg.use_wandb:
        import wandb
        wandb.log({
            "test/srcc": test_result.srcc,
            "test/plcc": test_result.plcc,
            "test/mae":  test_result.mae,
        })
        wandb.finish()

    return model, test_result


# ------------------------------------------------------------------ #
# CLI entry-point
# ------------------------------------------------------------------ #

def _parse_args() -> Config:
    """
    Parse command-line arguments and return a Config.
    Only the most commonly varied fields are exposed here;
    everything else uses Config defaults.
    """
    parser = argparse.ArgumentParser(
        description="Teacher pre-training for RG-KD-IQA"
    )
    parser.add_argument("--data_root",        type=str,   default="./data")
    parser.add_argument("--dataset_name",     type=str,   default="koniq10k")
    parser.add_argument("--csv_path",         type=str,   default="")
    parser.add_argument("--output_dir",       type=str,   default="./outputs")
    parser.add_argument("--teacher_ckpt",     type=str,   default="")
    parser.add_argument("--teacher_epochs",   type=int,   default=50)
    parser.add_argument("--teacher_lr",       type=float, default=1e-4)
    parser.add_argument("--teacher_batch_size", type=int, default=8)
    parser.add_argument("--embed_dim",        type=int,   default=512)
    parser.add_argument("--device",           type=str,   default="cuda")
    parser.add_argument("--amp",              action="store_true", default=True)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--use_wandb",        action="store_true", default=False)
    args = parser.parse_args()

    cfg = Config()
    for key, val in vars(args).items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
    return cfg


if __name__ == "__main__":
    cfg = _parse_args()
    train_teacher(cfg)