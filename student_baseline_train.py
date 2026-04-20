"""
student_baseline_train.py
--------------------------
Trains EfficientNet-B0 student with regression + ranking loss only.
No teacher, no KD, no graph distillation, no DataParallel.

Reuses:
    student.py       – StudentModel, build_student
    losses.py        – TeacherTotalLoss (reg + ranking)
    datasets.py      – build_dataloaders
    evaluate.py      – evaluate_model, EvalResult
    utils.py         – AverageMeter, EarlyStopping, get_cosine_schedule_with_warmup,
                       get_device, load_checkpoint, save_checkpoint,
                       print_model_info, set_seed
    config.py        – Config
"""

from __future__ import annotations

import os
import time
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from datasets import build_dataloaders
from evaluate import EvalResult, evaluate_model
from losses import TeacherTotalLoss          # reg + ranking, reused as-is
from student import StudentModel, build_student
from utils import (
    AverageMeter,
    EarlyStopping,
    get_cosine_schedule_with_warmup,
    get_device,
    load_checkpoint,
    print_model_info,
    save_checkpoint,
    set_seed,
)


# ------------------------------------------------------------------ #
# One training epoch
# ------------------------------------------------------------------ #

def train_one_epoch_baseline(
    model:     StudentModel,
    loader:    DataLoader,
    criterion: TeacherTotalLoss,
    optimizer: torch.optim.Optimizer,
    scaler:    GradScaler,
    device:    torch.device,
    epoch:     int,
    cfg:       Config,
) -> Dict[str, float]:
    """
    Single training epoch for the no-KD student baseline.

    Parameters
    ----------
    model     : StudentModel
    loader    : DataLoader  – yields {'image': (B,3,H,W), 'mos': (B,)}
    criterion : TeacherTotalLoss  – reg + ranking loss only
    optimizer : Optimizer
    scaler    : GradScaler
    device    : torch.device
    epoch     : int  – 0-indexed
    cfg       : Config

    Returns
    -------
    dict with float keys: 'total', 'reg', 'rank', 'lr'
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

    use_amp = cfg.amp and device.type == "cuda"

    for batch in pbar:
        images  = batch["image"].to(device, non_blocking=True)  # (B, 3, H, W)
        targets = batch["mos"].to(device, non_blocking=True)     # (B,)
        B = images.size(0)

        optimizer.zero_grad(set_to_none=True)

        # ---- Forward ----------------------------------------------- #
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _emb, predictions, _feat = model(images)  # (B,E), (B,1), (B,F)
            # criterion always in float32 to avoid AMP numerical issues
            # in the pairwise ranking loss (small differences, masking)
            total, reg, rank = criterion(predictions.float(), targets)
        else:
            _emb, predictions, _feat = model(images)
            total, reg, rank = criterion(predictions, targets)

        # ---- Backward ---------------------------------------------- #
        if use_amp:
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        # ---- Meters ------------------------------------------------- #
        meters["total"].update(total.item(), B)
        meters["reg"].update(reg.item(),     B)
        meters["rank"].update(rank.item(),   B)

        pbar.set_postfix(
            loss=f"{meters['total'].avg:.4f}",
            reg=f"{meters['reg'].avg:.4f}",
            rank=f"{meters['rank'].avg:.4f}",
        )

    pbar.close()

    return {
        "total": meters["total"].avg,
        "reg":   meters["reg"].avg,
        "rank":  meters["rank"].avg,
        "lr":    optimizer.param_groups[0]["lr"],
    }


# ------------------------------------------------------------------ #
# Main training function
# ------------------------------------------------------------------ #

def train_student_baseline(cfg: Config) -> Tuple[StudentModel, EvalResult]:
    """
    Full training pipeline for the no-KD student baseline.

    Parameters
    ----------
    cfg : Config
        Uses student_* fields for lr, epochs, batch size, warmup.
        Checkpoint saved to cfg.output_dir/student_baseline/

    Returns
    -------
    model       : StudentModel  – best checkpoint reloaded
    test_result : EvalResult
    """
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    # ---- Data ------------------------------------------------------- #
    # build_dataloaders reads cfg.teacher_batch_size for the train loader
    # so we temporarily point it to the student batch size
    _orig = cfg.teacher_batch_size
    cfg.teacher_batch_size = cfg.student_batch_size
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    cfg.teacher_batch_size = _orig

    # ---- Model ------------------------------------------------------ #
    model = build_student(cfg).to(device)
    print_model_info(model, name="Student Baseline (no KD)")

    # Optionally resume
    start_epoch = 0
    if cfg.student_ckpt and os.path.isfile(cfg.student_ckpt):
        ckpt = load_checkpoint(cfg.student_ckpt, model, device=device)
        start_epoch = ckpt.get("epoch", 0)

    # ---- Optimiser + Schedule --------------------------------------- #
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.student_lr,
        weight_decay=cfg.student_weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        warmup_epochs=cfg.student_warmup_epochs,
        total_epochs=cfg.student_epochs,
    )
    scaler = GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    # ---- Loss ------------------------------------------------------- #
    # TeacherTotalLoss = reg + ranking; lambda_graph not used here
    criterion = TeacherTotalLoss(cfg)

    # ---- Early stopping -------------------------------------------- #
    early_stopper = EarlyStopping(
        patience=cfg.early_stopping_patience,
        min_delta=cfg.early_stopping_min_delta,
        mode="max",
    )

    # ---- Checkpointing --------------------------------------------- #
    ckpt_dir  = os.path.join(cfg.output_dir, "student_baseline")
    os.makedirs(ckpt_dir, exist_ok=True)
    last_ckpt = os.path.join(ckpt_dir, "student_baseline_last.pth")
    best_ckpt = os.path.join(ckpt_dir, "student_baseline_best.pth")

    best_srcc:   float = -1.0
    best_result: Optional[EvalResult] = None

    # ---- WandB (optional) ------------------------------------------ #
    if cfg.use_wandb:
        try:
            import wandb
            wandb.init(
                project=cfg.wandb_project,
                name=cfg.wandb_run_name or "student_baseline",
                config=vars(cfg),
            )
        except ImportError:
            print("[student_baseline_train] wandb not installed; skipping.")
            cfg.use_wandb = False

    # ----------------------------------------------------------------- #
    # Training loop
    # ----------------------------------------------------------------- #
    print(f"\n{'='*60}")
    print(f"  Student Baseline (no KD) on {device}  |  epochs={cfg.student_epochs}")
    print(f"{'='*60}\n")

    epoch_bar = tqdm(
        range(start_epoch, cfg.student_epochs),
        desc="Epochs",
        unit="epoch",
        dynamic_ncols=True,
    )

    for epoch in epoch_bar:
        t0 = time.time()

        train_metrics = train_one_epoch_baseline(
            model, train_loader, criterion,
            optimizer, scaler, device, epoch, cfg,
        )
        scheduler.step()

        epoch_bar.set_postfix(
            loss=f"{train_metrics['total']:.4f}",
            lr=f"{train_metrics['lr']:.1e}",
            time=f"{time.time() - t0:.1f}s",
        )

        # ---- Periodic validation ------------------------------------ #
        do_eval = (
            (epoch + 1) % cfg.eval_every_n_epochs == 0
            or epoch == cfg.student_epochs - 1
        )
        if do_eval:
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

            state = {
                "epoch":                epoch + 1,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_srcc":            best_srcc,
                "config":               vars(cfg),
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

            if early_stopper(result.srcc):
                tqdm.write(
                    f"\n[EarlyStopping] Triggered at epoch {epoch+1}. "
                    f"Best SRCC = {best_srcc:.4f}"
                )
                break

    epoch_bar.close()

    # ---- Final test evaluation ------------------------------------- #
    print("\n[student_baseline] Loading best checkpoint for test evaluation …")
    if os.path.isfile(best_ckpt):
        load_checkpoint(best_ckpt, model, device=device)

    test_result = evaluate_model(
        model, test_loader, device,
        image_size=cfg.image_size,
        amp=False,
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