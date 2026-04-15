"""
datasets.py
-----------
Dataset classes and DataLoader factories for No-Reference IQA.

Supported dataset layouts
--------------------------
1. KonIQ-10k  – CSV: image_name, MOS (1–5 scale)
2. LIVE        – CSV: image_path, dmos
3. CSIQ        – CSV: image, dst_img, dmos
4. Generic     – Any CSV with columns  ``image_path``  and  ``mos``

The module normalises every MOS column to [0, 1] so downstream code
never has to worry about differing scales.

Public API
----------
- IQADataset           – torch Dataset
- build_dataloaders()  – returns (train_loader, val_loader, test_loader)
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from config import Config


# ------------------------------------------------------------------ #
# Default augmentation pipelines
# ------------------------------------------------------------------ #

def get_train_transform(image_size: int = 224) -> transforms.Compose:
    """
    Returns the training augmentation pipeline.

    Parameters
    ----------
    image_size : int
        Target square size (H = W).

    Returns
    -------
    transforms.Compose
        Composed transform suitable for training.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_val_transform(image_size: int = 224) -> transforms.Compose:
    """
    Returns the deterministic validation/test transform.

    Parameters
    ----------
    image_size : int
        Target square size (H = W).

    Returns
    -------
    transforms.Compose
        Composed transform suitable for inference.
    """
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


# ------------------------------------------------------------------ #
# Dataset class
# ------------------------------------------------------------------ #

class IQADataset(Dataset):
    """
    Generic No-Reference IQA dataset.

    Reads a CSV that must contain at least two columns:
      - ``image_path`` : absolute or relative path to the image file.
      - ``mos``        : Mean Opinion Score (any scale; normalised to [0,1]).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ``image_path`` and ``mos``.
    transform : Callable | None
        torchvision transform applied to each image.
    data_root : str
        Optional prefix prepended to relative ``image_path`` values.

    Attributes
    ----------
    image_paths : List[str]   – absolute image paths
    mos         : np.ndarray  – normalised MOS in [0, 1], shape (N,)
    transform   : Callable
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform: Optional[Callable] = None,
        data_root: str = "",
    ) -> None:
        super().__init__()
        self.transform = transform
        self.data_root = data_root

        # Resolve paths
        self.image_paths: List[str] = [
            p if os.path.isabs(p) else os.path.join(data_root, p)
            for p in df["image_path"].tolist()
        ]

        # Normalise MOS to [0, 1]
        raw_mos = df["mos"].values.astype(np.float32)
        mos_min, mos_max = raw_mos.min(), raw_mos.max()
        if mos_max - mos_min < 1e-6:
            self.mos = raw_mos * 0.0  # edge case: constant scores
        else:
            self.mos = (raw_mos - mos_min) / (mos_max - mos_min)

    # -------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict with keys:
          ``image``  – FloatTensor of shape (3, H, W)
          ``mos``    – scalar FloatTensor in [0, 1]
          ``index``  – LongTensor scalar (sample index in dataset)
        """
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {
            "image": img,                                         # (3, H, W)
            "mos": torch.tensor(self.mos[idx], dtype=torch.float32),  # scalar
            "index": torch.tensor(idx, dtype=torch.long),         # scalar
        }


# ------------------------------------------------------------------ #
# CSV loaders per dataset
# ------------------------------------------------------------------ #

def _load_koniq10k(csv_path: str, data_root: str) -> pd.DataFrame:
    """
    Parse KonIQ-10k CSV into the standard (image_path, mos) format.

    KonIQ-10k default CSV columns: image_name, C1, C2, C3, C4, C5, MOS, MOS_std
    """
    df = pd.read_csv(csv_path)
    # Column names may vary; try common variants
    img_col = next(
        (c for c in df.columns if "image" in c.lower() or "name" in c.lower()), df.columns[0]
    )
    mos_col = next(
        (c for c in df.columns if "mos" in c.lower()), df.columns[-1]
    )
    df = df.rename(columns={img_col: "image_path", mos_col: "mos"})
    df["image_path"] = df["image_path"].apply(
        lambda x: os.path.join(data_root, "1024x768", str(x))
    )
    return df[["image_path", "mos"]]


def _load_generic(csv_path: str) -> pd.DataFrame:
    """
    Load a generic CSV that already has columns ``image_path`` and ``mos``.

    PLACEHOLDER: Extend this function to support your custom dataset format.
    """
    df = pd.read_csv(csv_path)
    assert "image_path" in df.columns and "mos" in df.columns, (
        "Generic CSV must contain 'image_path' and 'mos' columns. "
        f"Found: {list(df.columns)}"
    )
    return df[["image_path", "mos"]]


# ------------------------------------------------------------------ #
# Public builder
# ------------------------------------------------------------------ #

def build_dataloaders(cfg: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / validation / test DataLoaders from the config.

    Split strategy: stratified-like random split based on ``cfg.train_split``
    and ``cfg.val_split`` (remaining fraction = test).

    Parameters
    ----------
    cfg : Config
        Framework-wide configuration object.

    Returns
    -------
    train_loader : DataLoader  – shuffled, with training augmentations
    val_loader   : DataLoader  – deterministic
    test_loader  : DataLoader  – deterministic
    """
    # ---- Load raw DataFrame ---------------------------------------- #
    dataset_name = cfg.dataset_name.lower()
    if dataset_name == "koniq10k":
        df = _load_koniq10k(cfg.csv_path, cfg.data_root)
    else:
        # Generic CSV fallback
        df = _load_generic(cfg.csv_path)

    df = df.dropna().reset_index(drop=True)

    # ---- Splits ----------------------------------------------------- #
    n = len(df)
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)

    n_train = int(n * cfg.train_split)
    n_val = int(n * cfg.val_split)

    train_idx = idx[:n_train].tolist()
    val_idx = idx[n_train : n_train + n_val].tolist()
    test_idx = idx[n_train + n_val :].tolist()

    # ---- Build datasets --------------------------------------------- #
    train_transform = get_train_transform(cfg.image_size)
    val_transform = get_val_transform(cfg.image_size)

    full_dataset = IQADataset(df, transform=None, data_root=cfg.data_root)

    class _SubsetWithTransform(Dataset):
        """Wraps a list of indices and applies a given transform."""

        def __init__(self, base: IQADataset, indices: List[int], transform: Callable) -> None:
            self.base = base
            self.indices = indices
            self.transform = transform

        def __len__(self) -> int:
            return len(self.indices)

        def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
            real_idx = self.indices[i]
            img = Image.open(self.base.image_paths[real_idx]).convert("RGB")
            img = self.transform(img)
            return {
                "image": img,
                "mos": torch.tensor(self.base.mos[real_idx], dtype=torch.float32),
                "index": torch.tensor(real_idx, dtype=torch.long),
            }

    train_ds = _SubsetWithTransform(full_dataset, train_idx, train_transform)
    val_ds = _SubsetWithTransform(full_dataset, val_idx, val_transform)
    test_ds = _SubsetWithTransform(full_dataset, test_idx, val_transform)

    # ---- DataLoaders ------------------------------------------------ #
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.teacher_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    print(
        f"[datasets] train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}"
    )
    return train_loader, val_loader, test_loader