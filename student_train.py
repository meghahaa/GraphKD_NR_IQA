"""
student_train.py
----------------
Knowledge-distillation training loop for the EfficientNet student.

Pipeline
--------
1. Load the frozen (eval-mode) teacher from ``cfg.teacher_ckpt``.
2. Build the student model, optimiser, cosine-warmup LR schedule, and
   AMP GradScaler — identical plumbing to teacher_train.py.
3. Initialise a MemoryBank (when ``cfg.use_memory_bank=True``) to extend
   the relational graph context beyond a single small batch.
4. For each epoch:
     a. Teacher: frozen forward pass → (t_emb, t_pred, _)
     b. Student: trainable forward pass → (s_emb, s_pred, _)
     c. Compute StudentTotalLoss (reg + rank + graph alignment)
     d. Backward with AMP + gradient clipping
     e. Update MemoryBank with teacher embeddings
     f. tqdm inner progress bar shows per-batch metrics
5. Every ``cfg.eval_every_n_epochs`` epochs evaluate on the val set,
   checkpoint if SRCC improved, and check EarlyStopping.
6. On exit (early stop or epoch limit) reload best checkpoint and
   run final evaluation on the test set.

CLI
---
    python student_train.py --teacher_ckpt outputs/teacher/teacher_best.pth
    python student_train.py --help
"""

from __future__ import annotations

import argparse
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
from student_losses import MemoryBank, StudentTotalLoss
from student import StudentModel, build_student
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
# Load frozen teacher
# ------------------------------------------------------------------ #

def load_frozen_teacher(cfg: Config, device: torch.device) -> TeacherModel:
    """
    Instantiate the teacher, load its checkpoint, freeze all parameters,
    and set it to eval mode.

    Parameters
    ----------
    cfg    : Config  – teacher_ckpt path and model hyperparameters
    device : torch.device

    Returns
    -------
    teacher : TeacherModel  (frozen, eval mode)

    Raises
    ------
    FileNotFoundError if ``cfg.teacher_ckpt`` does not exist.
    """
    if not os.path.isfile(cfg.teacher_ckpt):
        raise FileNotFoundError(
            f"[student_train] Teacher checkpoint not found at '{cfg.teacher_ckpt}'.\n"
            "Please run teacher_train.py first and set cfg.teacher_ckpt."
        )

    teacher = build_teacher(cfg).to(device)
    load_checkpoint(cfg.teacher_ckpt, teacher, device=device)

    # Freeze completely
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    print(f"[student_train] Teacher loaded and frozen from '{cfg.teacher_ckpt}'")
    return teacher


# ------------------------------------------------------------------ #
# One training epoch
# ------------------------------------------------------------------ #

def train_one_epoch(
    teacher:   TeacherModel,
    student:   StudentModel,
    loader:    DataLoader,
    criterion: StudentTotalLoss,
    optimizer: torch.optim.Optimizer,
    scaler:    GradScaler,
    bank:      Optional[MemoryBank],
    device:    torch.device,
    epoch:     int,
    cfg:       Config,
    pbar:      Optional[tqdm] = None,
) -> Dict[str, float]:
    """
    Run a single student training epoch with a tqdm inner progress bar.

    Parameters
    ----------
    teacher   : TeacherModel     – frozen; provides soft targets + embeddings
    student   : StudentModel     – being trained
    loader    : DataLoader       – yields dicts: 'image' (B,3,H,W), 'mos' (B,)
    criterion : StudentTotalLoss
    optimizer : Optimizer
    scaler    : GradScaler       – AMP scaler (identity when cfg.amp=False)
    bank      : MemoryBank|None  – relational context memory bank
    device    : torch.device
    epoch     : int              – 0-indexed
    cfg       : Config

    Returns
    -------
    dict with float values for keys:
        'total', 'reg', 'rank', 'graph', 'lr'
    """
    student.train()
    teacher.eval()   # Always keep teacher in eval mode

    meters = {
        "total": AverageMeter("total"),
        "reg":   AverageMeter("reg"),
        "rank":  AverageMeter("rank"),
        "graph": AverageMeter("graph"),
    }

    use_amp = cfg.amp and device.type == "cuda"

    for batch in loader:
        images  = batch["image"].to(device, non_blocking=True)   # (B, 3, H, W)
        targets = batch["mos"].to(device, non_blocking=True)      # (B,)
        B = images.size(0)

        optimizer.zero_grad(set_to_none=True)

        # ---- Teacher forward (no grad) ----------------------------- #
        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    t_emb, t_pred, _ = teacher(images)   # (B,E), (B,1), _
            else:
                t_emb, t_pred, _ = teacher(images)

            t_emb  = t_emb.detach().float()    # ensure float32 for loss
            t_pred = t_pred.detach().float()
            t_scores = t_pred.squeeze(1)        # (B,)

        # ---- Student forward (with grad) --------------------------- #
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                s_emb, s_pred, _ = student(images)       # (B,E), (B,1), _
                total, reg, rank, graph = criterion(
                    s_pred.float(), targets,
                    t_emb, s_emb.float(),
                    t_scores, bank,
                )
        else:
            s_emb, s_pred, _ = student(images)
            total, reg, rank, graph = criterion(
                s_pred, targets,
                t_emb, s_emb,
                t_scores, bank,
            )

        # ---- Backward ---------------------------------------------- #
        if use_amp:
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=5.0)
            optimizer.step()

        # ---- Update memory bank ------------------------------------ #
        if bank is not None:
            bank.update(t_emb, t_scores)

        # ---- Meters + tqdm postfix --------------------------------- #
        meters["total"].update(total.item(), B)
        meters["reg"].update(reg.item(),    B)
        meters["rank"].update(rank.item(),  B)
        meters["graph"].update(graph.item(), B)

        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(
                loss=f"{meters['total'].avg:.4f}",
                reg=f"{meters['reg'].avg:.4f}",
                rank=f"{meters['rank'].avg:.4f}",
                graph=f"{meters['graph'].avg:.4f}",
            )

    current_lr = optimizer.param_groups[0]["lr"]

    return {
        "total": meters["total"].avg,
        "reg":   meters["reg"].avg,
        "rank":  meters["rank"].avg,
        "graph": meters["graph"].avg,
        "lr":    current_lr,
    }


# ------------------------------------------------------------------ #
# Main distillation training function
# ------------------------------------------------------------------ #

def train_student(cfg: Config) -> Tuple[StudentModel, EvalResult]:
    """
    Full knowledge-distillation training pipeline for the student.

    Parameters
    ----------
    cfg : Config

    Returns
    -------
    student     : StudentModel  – best checkpoint reloaded
    test_result : EvalResult    – metrics on the held-out test set
    """
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    # ---- Data ------------------------------------------------------ #
    # Reuse build_dataloaders but honour student batch size
    # We temporarily patch the batch size so the loader is built correctly
    _orig_batch = cfg.teacher_batch_size
    cfg.teacher_batch_size = cfg.student_batch_size   # datasets.py reads this field
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    cfg.teacher_batch_size = _orig_batch              # restore

    # ---- Teacher (frozen) ----------------------------------------- #
    teacher = load_frozen_teacher(cfg, device)
    print_model_info(teacher, name="Teacher (frozen)")

    # ---- Student --------------------------------------------------- #
    student = build_student(cfg).to(device)
    print_model_info(student, name="Student")

    start_epoch = 0
    if cfg.student_ckpt and os.path.isfile(cfg.student_ckpt):
        ckpt = load_checkpoint(cfg.student_ckpt, student, device=device)
        start_epoch = ckpt.get("epoch", 0)

    # ---- Memory Bank ----------------------------------------------- #
    bank: Optional[MemoryBank] = None
    if cfg.use_memory_bank:
        bank = MemoryBank(
            size=cfg.memory_bank_size,
            embed_dim=cfg.embed_dim,
            device=device,
        )
        print(
            f"[student_train] MemoryBank initialised "
            f"(size={cfg.memory_bank_size}, embed_dim={cfg.embed_dim})"
        )

    # ---- Optimiser + Schedule -------------------------------------- #
    optimizer = torch.optim.AdamW(
        student.parameters(),
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
    criterion = StudentTotalLoss(cfg)

    # ---- Early stopping ------------------------------------------- #
    early_stopper = EarlyStopping(
        patience=cfg.early_stopping_patience,
        min_delta=cfg.early_stopping_min_delta,
        mode="max",
    )

    # ---- Checkpointing paths -------------------------------------- #
    ckpt_dir  = os.path.join(cfg.output_dir, "student")
    os.makedirs(ckpt_dir, exist_ok=True)
    last_ckpt = os.path.join(ckpt_dir, "student_last.pth")
    best_ckpt = os.path.join(ckpt_dir, "student_best.pth")

    best_srcc:   float = -1.0
    best_result: Optional[EvalResult] = None

    # ---- WandB (optional) ----------------------------------------- #
    if cfg.use_wandb:
        try:
            import wandb
            wandb.init(
                project=cfg.wandb_project,
                name=cfg.wandb_run_name or "student_kd",
                config=vars(cfg),
            )
        except ImportError:
            print("[student_train] wandb not installed; skipping.")
            cfg.use_wandb = False

    # ---------------------------------------------------------------- #
    # Outer epoch loop
    # ---------------------------------------------------------------- #
    print(f"\n{'='*65}")
    print(
        f"  Student KD training on {device}  |  "
        f"epochs={cfg.student_epochs}  |  bank={cfg.use_memory_bank}"
    )
    print(f"{'='*65}\n")

    for epoch in range(start_epoch, cfg.student_epochs):
        pbar = tqdm(
        total=len(train_loader),
        desc=f"Epoch {epoch+1}/{cfg.student_epochs}",
        unit="batch",
        dynamic_ncols=True
        )

        t0 = time.time()

        train_metrics = train_one_epoch(
            teacher, student, train_loader,
            criterion, optimizer, scaler,
            bank, device, epoch, cfg,
            pbar=pbar  
        )

        pbar.close()
        scheduler.step()
        epoch_time = time.time() - t0

        tqdm.write(
        f"Epoch {epoch+1}/{cfg.student_epochs} "
        f"| Loss: {train_metrics['total']:.4f} "
        f"| Reg: {train_metrics['reg']:.4f} "
        f"| Rank: {train_metrics['rank']:.4f} "
        f"| Graph: {train_metrics['graph']:.4f} "
        f"| Time: {epoch_time:.1f}s"
        )

        if device.type == "cuda":
            tqdm.write(print_gpu_memory())

        # ---- Periodic validation ----------------------------------- #
        do_eval = (
            (epoch + 1) % cfg.eval_every_n_epochs == 0
            or epoch == cfg.student_epochs - 1
        )
        if do_eval:
            result = evaluate_model(
                student, val_loader, device,
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

            # Save checkpoints
            state = {
                "epoch":                epoch + 1,
                "model_state_dict":     student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_srcc":            best_srcc,
                "config":               vars(cfg),
            }
            save_checkpoint(state, last_ckpt)
            save_checkpoint(state, last_ckpt, is_best=is_best, best_path=best_ckpt)

            if cfg.use_wandb:
                import wandb
                wandb.log({
                    "epoch":       epoch + 1,
                    "train/loss":  train_metrics["total"],
                    "train/reg":   train_metrics["reg"],
                    "train/rank":  train_metrics["rank"],
                    "train/graph": train_metrics["graph"],
                    "val/srcc":    result.srcc,
                    "val/plcc":    result.plcc,
                    "val/mae":     result.mae,
                })

            # ---- Early stopping check ----------------------------- #
            if early_stopper(result.srcc):
                tqdm.write(
                    f"\n[EarlyStopping] Triggered at epoch {epoch+1}. "
                    f"Best SRCC = {best_srcc:.4f}"
                )
                break


    # ---- Final test evaluation ------------------------------------ #
    print("\n[student_train] Loading best checkpoint for test evaluation …")
    if os.path.isfile(best_ckpt):
        load_checkpoint(best_ckpt, student, device=device)

    test_result = evaluate_model(
        student, test_loader, device,
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

    return student, test_result


# ------------------------------------------------------------------ #
# CLI entry-point
# ------------------------------------------------------------------ #

def _parse_args() -> Config:
    """
    Parse command-line arguments and return a populated Config.
    Only the most-varied fields are surfaced; everything else uses defaults.
    """
    parser = argparse.ArgumentParser(
        description="Student KD training for RG-KD-IQA"
    )
    parser.add_argument("--data_root",          type=str,   default="./data")
    parser.add_argument("--dataset_name",       type=str,   default="koniq10k")
    parser.add_argument("--csv_path",           type=str,   default="")
    parser.add_argument("--output_dir",         type=str,   default="./outputs")
    parser.add_argument("--teacher_ckpt",       type=str,   required=True,
                        help="Path to teacher_best.pth (required)")
    parser.add_argument("--student_ckpt",       type=str,   default="",
                        help="Path to resume a student checkpoint")
    parser.add_argument("--student_epochs",     type=int,   default=50)
    parser.add_argument("--student_lr",         type=float, default=1e-4)
    parser.add_argument("--student_batch_size", type=int,   default=8)
    parser.add_argument("--student_warmup_epochs", type=int, default=5)
    parser.add_argument("--embed_dim",          type=int,   default=512)
    parser.add_argument("--lambda_reg_student", type=float, default=1.0)
    parser.add_argument("--lambda_rank_student",type=float, default=1.0)
    parser.add_argument("--lambda_graph",       type=float, default=0.5)
    parser.add_argument("--graph_loss_type",    type=str,   default="kl",
                        choices=["mse", "kl"])
    parser.add_argument("--temperature",        type=float, default=0.07)
    parser.add_argument("--use_memory_bank",    action="store_true", default=True)
    parser.add_argument("--memory_bank_size",   type=int,   default=1024)
    parser.add_argument("--knn_k",              type=int,   default=8)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--device",             type=str,   default="cuda")
    parser.add_argument("--amp",                action="store_true", default=True)
    parser.add_argument("--seed",               type=int,   default=42)
    parser.add_argument("--use_wandb",          action="store_true", default=False)
    args = parser.parse_args()

    cfg = Config()
    for key, val in vars(args).items():
        if hasattr(cfg, key):
            setattr(cfg, key, val)
    return cfg


if __name__ == "__main__":
    cfg = _parse_args()
    train_student(cfg)