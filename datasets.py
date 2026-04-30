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

def get_train_transform(image_size: int = 224, patch_resize: int = 384) -> transforms.Compose:
    """
    Returns the training transform for a SINGLE patch/view.
    Called once per patch in the multi-patch pipeline.

    Parameters
    ----------
    image_size   : int – final crop size (H = W)
    patch_resize : int – resize before random crop (should be > image_size)

    Returns
    -------
    transforms.Compose
    """
    return transforms.Compose(
        [
            transforms.Resize((patch_resize, patch_resize)),
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


def get_val_transform(image_size: int = 224, patch_resize: int = 384) -> transforms.Compose:
    """
    Returns a deterministic center-crop transform for val/test.
    Used for the global view; deterministic crops handled separately.

    Parameters
    ----------
    image_size   : int
    patch_resize : int

    Returns
    -------
    transforms.Compose
    """
    return transforms.Compose(
        [
            transforms.Resize((patch_resize, patch_resize)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_patch_transforms(
    image_size: int = 224,
    patch_resize: int = 384,
    num_patches: int = 4,
    is_train: bool = True,
) -> List[transforms.Compose]:
    """
    Returns a list of transforms, one per patch view.

    Training   : num_patches independent random crops from patch_resize
    Validation : num_patches deterministic crops (4 corners + center if needed)

    Parameters
    ----------
    image_size   : int
    patch_resize : int
    num_patches  : int
    is_train     : bool

    Returns
    -------
    List of transforms.Compose, length = num_patches
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if is_train:
        # Each patch gets its own independent random crop transform
        patch_tfms = []
        for _ in range(num_patches):
            patch_tfms.append(
                transforms.Compose([
                    transforms.Resize((patch_resize, patch_resize)),
                    transforms.RandomCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                    transforms.ToTensor(),
                    normalize,
                ])
            )
        return patch_tfms
    else:
        # Deterministic: evenly spaced crops covering the spatial extent
        # Positions: top-left, top-right, bottom-left, bottom-right, center
        positions = [
            transforms.FiveCrop(image_size),   # returns tuple of 5 PIL images
        ]
        # We'll handle FiveCrop slicing in __getitem__ directly
        # Return a single transform; dataset handles the splitting
        return [
            transforms.Compose([
                transforms.Resize((patch_resize, patch_resize)),
                transforms.ToTensor(),
                normalize,
            ])
        ]


# ------------------------------------------------------------------ #
# Dataset class
# ------------------------------------------------------------------ #

class IQADataset(Dataset):
    """
    Generic No-Reference IQA dataset.

    Reads a CSV that must contain at least two columns:
      - ``image_path`` : absolute or relative path to the image file.
      - ``mos``        : Mean Opinion Score (any scale; normalised to [0,1] later). higher = better quality

    Has utility for dmos column , inverted to mos by 1 - dmos, so that higher is always better for consistency across datasets. 
    Set is_mos=False in constructor to enable this inversion.

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
        is_mos: bool = True,
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
        
        if not is_mos:
            self.mos = 1.0 - self.mos  # Invert if lower scores are better (e.g., DMOS)

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

class MultiPatchDataset(Dataset):
    """
    Public dataset class for cross-dataset evaluation using the
    same multi-patch pipeline as the main training/val loaders.

    Each __getitem__ returns:
        image : (num_patches+1, 3, H, W)  — global view + patch crops
        mos   : scalar float in [0, 1]
        index : int

    Parameters
    ----------
    df            : pd.DataFrame  – must have 'image_path' and 'mos' columns
    image_size    : int           – final crop size (e.g. 224)
    patch_resize  : int           – resize before cropping (e.g. 384)
    num_patches   : int           – number of patch crops (e.g. 4)
    is_train      : bool          – True for random crops, False for deterministic
    data_root     : str           – prepended to relative image paths
    """

    def __init__(
        self,
        df:           pd.DataFrame,
        image_size:   int  = 224,
        patch_resize: int  = 384,
        num_patches:  int  = 4,
        is_train:     bool = False,
        data_root:    str  = "",
    ) -> None:
        super().__init__()
        self.image_size   = image_size
        self.patch_resize = patch_resize
        self.num_patches  = num_patches
        self.is_train     = is_train

        # Resolve paths
        self.image_paths: List[str] = [
            p if os.path.isabs(p) else os.path.join(data_root, p)
            for p in df["image_path"].tolist()
        ]

        # Normalise MOS to [0, 1]
        raw_mos = df["mos"].values.astype(np.float32)
        mos_min, mos_max = raw_mos.min(), raw_mos.max()
        if mos_max - mos_min < 1e-6:
            self.mos = raw_mos * 0.0
        else:
            self.mos = (raw_mos - mos_min) / (mos_max - mos_min)

        # Shared normalize transform
        self._normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Global view transform
        if is_train:
            self._global_tfm = get_train_transform(image_size, patch_resize)
        else:
            self._global_tfm = get_val_transform(image_size, patch_resize)

        # Patch transforms
        self._patch_tfms = get_patch_transforms(
            image_size, patch_resize, num_patches, is_train
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict with keys:
            image : (num_patches+1, 3, H, W)
            mos   : scalar FloatTensor in [0, 1]
            index : scalar LongTensor
        """
        img = Image.open(self.image_paths[idx]).convert("RGB")

        # Global view
        global_view = self._global_tfm(img)              # (3, H, W)

        # Patch views
        if self.is_train:
            patch_views = [tfm(img) for tfm in self._patch_tfms]
        else:
            resized     = transforms.Resize((self.patch_resize, self.patch_resize))(img)
            five_crops  = transforms.FiveCrop(self.image_size)(resized)
            patch_views = [
                self._normalize(c) for c in five_crops[:self.num_patches]
            ]

        # Stack → (num_patches+1, 3, H, W)
        all_views = torch.stack([global_view] + patch_views, dim=0)

        return {
            "image": all_views,
            "mos":   torch.tensor(self.mos[idx], dtype=torch.float32),
            "index": torch.tensor(idx, dtype=torch.long),
        }
    

def build_cross_dataset_loader(
    csv_path:     str,
    image_size:   int  = 224,
    patch_resize: int  = 384,
    num_patches:  int  = 4,
    data_root:    str  = "",
    batch_size:   int  = 16,
    num_workers:  int  = 2,
) -> DataLoader:
    """
    Build a DataLoader for cross-dataset evaluation using the
    full multi-patch pipeline. CSV must have 'image_path' and 'mos' columns.

    Parameters
    ----------
    csv_path     : str  – path to metadata CSV
    image_size   : int  – crop size (default 224)
    patch_resize : int  – resize before cropping (default 384)
    num_patches  : int  – number of patch crops (default 4)
    data_root    : str  – optional prefix for relative image paths
    batch_size   : int
    num_workers  : int

    Returns
    -------
    DataLoader  – yields {'image': (B, P+1, 3, H, W), 'mos': (B,), 'index': (B,)}
    """
    df = pd.read_csv(csv_path).dropna().reset_index(drop=True)
    assert "image_path" in df.columns and "mos" in df.columns, (
        f"CSV must have 'image_path' and 'mos' columns. Found: {list(df.columns)}"
    )

    dataset = MultiPatchDataset(
        df           = df,
        image_size   = image_size,
        patch_resize = patch_resize,
        num_patches  = num_patches,
        is_train     = False,   # always deterministic for evaluation
        data_root    = data_root,
    )

    print(f"[cross-dataset] {csv_path}  →  {len(dataset)} samples")

    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )


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

def _load_kadid10k(csv_path: str, data_root: str) -> pd.DataFrame:
    """
    Parse KADID-10k CSV into the standard (image_path, mos) format.

    KADID-10k default CSV columns: ref_img, dis_img, dmos
    """
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"image": "image_path", "dmos": "mos"})
    df["image_path"] = df["image_path"].apply(
        lambda x: os.path.join(data_root, "images", str(x))
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
    elif dataset_name == "kadid10k":
        df = _load_kadid10k(cfg.csv_path, cfg.data_root)
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

    full_dataset = IQADataset(df, transform=None, data_root=cfg.data_root)

    # ---- Build datasets ----------------------------------------- #
    train_ds = MultiPatchDataset(
        df           = df.iloc[train_idx].reset_index(drop=True),
        image_size   = cfg.image_size,
        patch_resize = cfg.patch_resize,
        num_patches  = cfg.num_patches,
        is_train     = True,
        data_root    = cfg.data_root,
    )
    val_ds = MultiPatchDataset(
        df           = df.iloc[val_idx].reset_index(drop=True),
        image_size   = cfg.image_size,
        patch_resize = cfg.patch_resize,
        num_patches  = cfg.num_patches,
        is_train     = False,
        data_root    = cfg.data_root,
    )
    test_ds = MultiPatchDataset(
        df           = df.iloc[test_idx].reset_index(drop=True),
        image_size   = cfg.image_size,
        patch_resize = cfg.patch_resize,
        num_patches  = cfg.num_patches,
        is_train     = False,
        data_root    = cfg.data_root,
    )

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