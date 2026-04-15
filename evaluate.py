"""
evaluate.py
-----------
Common evaluation routines shared by teacher and student.

Metrics
-------
- SRCC  (Spearman Rank Correlation Coefficient) – measures rank preservation
- PLCC  (Pearson Linear Correlation Coefficient) – measures linearity
- Inference time (ms/image, averaged over the test set)
- Model size (parameters + estimated MB)

Public API
----------
- evaluate_model()  – main entry-point; returns EvalResult dataclass
- EvalResult        – named container for all metrics
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

from utils import count_parameters, model_size_mb


# ------------------------------------------------------------------ #
# Result container
# ------------------------------------------------------------------ #

@dataclass
class EvalResult:
    """
    Container for evaluation metrics.

    Attributes
    ----------
    srcc            : float  – Spearman rank correlation in [-1, 1]
    plcc            : float  – Pearson correlation in [-1, 1]
    mae             : float  – Mean absolute error
    rmse            : float  – Root mean squared error
    inference_ms    : float  – Mean inference time per image (ms)
    n_params_M      : float  – Trainable parameters in millions
    model_size_mb   : float  – Estimated model size in MB
    n_samples       : int    – Number of test samples evaluated
    """
    srcc: float
    plcc: float
    mae: float
    rmse: float
    inference_ms: float
    n_params_M: float
    model_size_mb: float
    n_samples: int

    def __str__(self) -> str:
        return (
            f"SRCC={self.srcc:.4f}  PLCC={self.plcc:.4f}  "
            f"MAE={self.mae:.4f}  RMSE={self.rmse:.4f}  "
            f"Inference={self.inference_ms:.2f}ms/img  "
            f"Params={self.n_params_M:.2f}M  "
            f"Size≈{self.model_size_mb:.1f}MB  "
            f"N={self.n_samples}"
        )


# ------------------------------------------------------------------ #
# Correlation helpers
# ------------------------------------------------------------------ #

def compute_srcc(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    Spearman Rank Correlation Coefficient.

    Parameters
    ----------
    preds   : (N,) float array – model predictions
    targets : (N,) float array – ground-truth MOS

    Returns
    -------
    float in [-1, 1]
    """
    corr, _ = spearmanr(preds, targets)
    return float(corr)


def compute_plcc(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    Pearson Linear Correlation Coefficient.

    Parameters
    ----------
    preds   : (N,) float array
    targets : (N,) float array

    Returns
    -------
    float in [-1, 1]
    """
    corr, _ = pearsonr(preds, targets)
    return float(corr)


# ------------------------------------------------------------------ #
# Inference timing
# ------------------------------------------------------------------ #

def _warmup_model(
    model: nn.Module,
    device: torch.device,
    image_size: int = 224,
    n_warmup: int = 5,
) -> None:
    """
    Run a few dummy forward passes to initialise CUDA kernels.

    Parameters
    ----------
    model      : nn.Module
    device     : torch.device
    image_size : int
    n_warmup   : int
    """
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, image_size, image_size, device=device)
        for _ in range(n_warmup):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()


def measure_inference_time(
    model: nn.Module,
    device: torch.device,
    image_size: int = 224,
    n_runs: int = 50,
) -> float:
    """
    Measure mean per-image inference latency (ms) using single-image batches.

    CUDA events are used for GPU timing; time.perf_counter for CPU.

    Parameters
    ----------
    model      : nn.Module – must support forward(x) returning anything
    device     : torch.device
    image_size : int
    n_runs     : int – number of timed forward passes

    Returns
    -------
    float – mean milliseconds per image
    """
    _warmup_model(model, device, image_size)

    model.eval()
    dummy = torch.zeros(1, 3, image_size, image_size, device=device)
    times = []

    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == "cuda":
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt   = torch.cuda.Event(enable_timing=True)
                start_evt.record()
                _ = model(dummy)
                end_evt.record()
                torch.cuda.synchronize()
                times.append(start_evt.elapsed_time(end_evt))   # ms
            else:
                t0 = time.perf_counter()
                _ = model(dummy)
                times.append((time.perf_counter() - t0) * 1000)  # s → ms

    return float(np.mean(times))


# ------------------------------------------------------------------ #
# Main evaluation function
# ------------------------------------------------------------------ #

def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    image_size: int = 224,
    amp: bool = False,
    use_predictions_fn: Optional[callable] = None,
) -> EvalResult:
    """
    Evaluate a model on a DataLoader and return all metrics.

    The function expects the model's forward() to return one of:
      - a Tensor of shape (B, 1)                      [regression model]
      - a tuple whose **first** element is (B, embed_dim) and
        **second** element is (B, 1)                  [teacher / student]

    If your model has a different output format, pass a custom
    ``use_predictions_fn`` that maps ``model(batch_images)`` → (B,) Tensor.

    Parameters
    ----------
    model              : nn.Module
    loader             : DataLoader – yields dicts with 'image' and 'mos'
    device             : torch.device
    image_size         : int – for timing measurement
    amp                : bool – use torch.autocast for inference
    use_predictions_fn : callable | None
        Signature: (model, images) → Tensor of shape (B,)
        When None, the default extraction logic is used.

    Returns
    -------
    EvalResult
    """
    model.eval()
    all_preds: list  = []
    all_targets: list = []

    with torch.no_grad():
        for batch in loader:
            images  = batch["image"].to(device)   # (B, 3, H, W)
            targets = batch["mos"].to(device)      # (B,)

            ctx = torch.autocast(device_type=device.type, dtype=torch.float16) \
                  if (amp and device.type == "cuda") else torch.no_grad()

            with ctx if amp else torch.no_grad():
                if use_predictions_fn is not None:
                    preds = use_predictions_fn(model, images)   # (B,)
                else:
                    out = model(images)
                    if isinstance(out, (tuple, list)):
                        # Second element is (B, 1) quality scores
                        preds = out[1].squeeze(1)               # (B,)
                    else:
                        preds = out.squeeze(1)                  # (B,)

            all_preds.append(preds.cpu().float().numpy())
            all_targets.append(targets.cpu().float().numpy())

    all_preds   = np.concatenate(all_preds)    # (N,)
    all_targets = np.concatenate(all_targets)  # (N,)

    srcc  = compute_srcc(all_preds, all_targets)
    plcc  = compute_plcc(all_preds, all_targets)
    mae   = float(np.mean(np.abs(all_preds - all_targets)))
    rmse  = float(np.sqrt(np.mean((all_preds - all_targets) ** 2)))

    inf_ms     = measure_inference_time(model, device, image_size)
    n_params_M = count_parameters(model) / 1e6
    size_mb    = model_size_mb(model)

    return EvalResult(
        srcc=srcc,
        plcc=plcc,
        mae=mae,
        rmse=rmse,
        inference_ms=inf_ms,
        n_params_M=n_params_M,
        model_size_mb=size_mb,
        n_samples=len(all_preds),
    )


# ------------------------------------------------------------------ #
# Pretty-print comparison helper
# ------------------------------------------------------------------ #

def compare_results(
    teacher_result: EvalResult,
    student_result: EvalResult,
) -> None:
    """
    Print a side-by-side table comparing teacher and student metrics.

    Parameters
    ----------
    teacher_result : EvalResult
    student_result : EvalResult
    """
    header = f"{'Metric':<22} {'Teacher':>12} {'Student':>12}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    rows = [
        ("SRCC",            f"{teacher_result.srcc:.4f}",            f"{student_result.srcc:.4f}"),
        ("PLCC",            f"{teacher_result.plcc:.4f}",            f"{student_result.plcc:.4f}"),
        ("MAE",             f"{teacher_result.mae:.4f}",             f"{student_result.mae:.4f}"),
        ("RMSE",            f"{teacher_result.rmse:.4f}",            f"{student_result.rmse:.4f}"),
        ("Inference (ms)",  f"{teacher_result.inference_ms:.2f}",    f"{student_result.inference_ms:.2f}"),
        ("Params (M)",      f"{teacher_result.n_params_M:.2f}",      f"{student_result.n_params_M:.2f}"),
        ("Size (MB)",       f"{teacher_result.model_size_mb:.1f}",   f"{student_result.model_size_mb:.1f}"),
        ("N samples",       f"{teacher_result.n_samples}",           f"{student_result.n_samples}"),
    ]

    for name, t_val, s_val in rows:
        print(f"{name:<22} {t_val:>12} {s_val:>12}")
    print("=" * len(header) + "\n")