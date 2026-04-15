"""
utils.py
--------
Shared utility functions for the RG-KD-IQA framework.

Contents
--------
- set_seed()                 – reproducibility
- get_device()               – resolve torch device
- count_parameters()         – model size helpers
- AverageMeter               – running mean tracker
- save_checkpoint()          – save model + optimiser state
- load_checkpoint()          – restore from checkpoint
- get_cosine_schedule()      – LR schedule with linear warm-up
- model_size_mb()            – disk size of a model's parameters
"""

from __future__ import annotations

import os
import random
import math
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


# ------------------------------------------------------------------ #
# Reproducibility
# ------------------------------------------------------------------ #

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch (CPU + CUDA).

    Parameters
    ----------
    seed : int
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------------ #
# Device
# ------------------------------------------------------------------ #

def get_device(device_str: str = "cuda") -> torch.device:
    """
    Resolve the compute device.

    Parameters
    ----------
    device_str : str – "cuda" or "cpu"

    Returns
    -------
    torch.device
    """
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[utils] CUDA not available – falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


# ------------------------------------------------------------------ #
# Parameter counting / model size
# ------------------------------------------------------------------ #

def count_parameters(model: nn.Module) -> int:
    """
    Return total number of trainable parameters.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    int – number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model: nn.Module) -> float:
    """
    Estimate model size in MB (parameters only, float32 assumed).

    Actual on-disk size can differ due to optimiser states, etc.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    float – megabytes
    """
    n_params = sum(p.numel() for p in model.parameters())
    # float32 = 4 bytes per element; float16 = 2 bytes
    return n_params * 4 / (1024 ** 2)


def print_model_info(model: nn.Module, name: str = "Model") -> None:
    """
    Pretty-print parameter count and estimated size.

    Parameters
    ----------
    model : nn.Module
    name  : str – label shown in output
    """
    trainable = count_parameters(model)
    total = sum(p.numel() for p in model.parameters())
    size = model_size_mb(model)
    print(
        f"[{name}]  trainable={trainable/1e6:.2f}M  "
        f"total={total/1e6:.2f}M  "
        f"size≈{size:.1f}MB"
    )


# ------------------------------------------------------------------ #
# Running mean tracker
# ------------------------------------------------------------------ #

class AverageMeter:
    """
    Computes and stores the running average of a scalar quantity.

    Usage
    -----
    meter = AverageMeter("loss")
    meter.update(loss_value, batch_size)
    print(meter.avg)
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        """
        Parameters
        ----------
        val : float – value to accumulate
        n   : int   – weight (typically batch size)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __repr__(self) -> str:
        return f"AverageMeter({self.name}: avg={self.avg:.4f})"


# ------------------------------------------------------------------ #
# Checkpointing
# ------------------------------------------------------------------ #

def save_checkpoint(
    state: Dict[str, Any],
    path: str,
    is_best: bool = False,
    best_path: Optional[str] = None,
) -> None:
    """
    Save a checkpoint dictionary to disk.

    Parameters
    ----------
    state     : dict  – must include at minimum 'model_state_dict'
    path      : str   – file path for the checkpoint
    is_best   : bool  – if True, also copy to ``best_path``
    best_path : str | None – destination for the best checkpoint
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save(state, path)
    if is_best and best_path is not None:
        import shutil
        shutil.copyfile(path, best_path)
        print(f"[utils] Best checkpoint saved to {best_path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Load a checkpoint and restore model (and optionally optimiser) state.

    Parameters
    ----------
    path      : str
    model     : nn.Module – target model
    optimizer : Optimizer | None – if given, restore optimiser state too
    device    : torch.device | None

    Returns
    -------
    dict – full checkpoint contents (e.g. 'epoch', 'best_srcc', …)
    """
    map_location = device if device is not None else torch.device("cpu")
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"[utils] Loaded checkpoint from '{path}' (epoch {ckpt.get('epoch', '?')})")
    return ckpt


# ------------------------------------------------------------------ #
# Learning-rate schedule
# ------------------------------------------------------------------ #

def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr_ratio: float = 0.01,
) -> LambdaLR:
    """
    Cosine annealing LR schedule with a linear warm-up phase.

    Schedule:
      epoch ∈ [0, warmup_epochs)  → LR rises linearly from 0 to base_lr
      epoch ∈ [warmup_epochs, T]  → cosine decay down to min_lr_ratio · base_lr

    Parameters
    ----------
    optimizer      : Optimizer
    warmup_epochs  : int
    total_epochs   : int
    min_lr_ratio   : float – floor for the cosine decay (as fraction of base LR)

    Returns
    -------
    LambdaLR scheduler (call .step() once per epoch)
    """

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(warmup_epochs, 1))
        progress = float(epoch - warmup_epochs) / float(
            max(total_epochs - warmup_epochs, 1)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# ------------------------------------------------------------------ #
# VRAM utilities
# ------------------------------------------------------------------ #

def print_gpu_memory() -> None:
    """Print current CUDA memory usage (allocated / reserved)."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024 ** 2
        reserv = torch.cuda.memory_reserved() / 1024 ** 2
        print(f"[GPU] allocated={alloc:.1f}MB  reserved={reserv:.1f}MB")
    else:
        print("[GPU] CUDA not available.")