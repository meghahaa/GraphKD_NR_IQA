"""
config.py
---------
Central configuration for the RG-KD-IQA framework.
All hyperparameters, paths, and flags live here so that
every other module imports from a single source of truth.

Usage:
    from config import Config
    cfg = Config()
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class Config:
    # ------------------------------------------------------------------ #
    # Paths
    # ------------------------------------------------------------------ #
    data_root: str = "./data"                  # Root folder for all datasets
    dataset_name: str = "koniq10k"             # One of: koniq10k | live | csiq | tid2013
    csv_path: str = ""                         # Override: explicit CSV with (image_path, mos) columns
    output_dir: str = "./outputs"              # Checkpoints, logs, results

    teacher_ckpt: str = ""                     # Path to saved teacher checkpoint
    student_ckpt: str = ""                     # Path to saved student checkpoint

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    teacher_backbone: str = "swin_tiny_patch4_window7_224"   # timm model name
    student_backbone: str = "efficientnet_b0"                 # timm model name
    embed_dim: int = 512           # Projection head output dimension
    pretrained: bool = True        # Load ImageNet weights for backbones

    # ------------------------------------------------------------------ #
    # Training – Teacher
    # ------------------------------------------------------------------ #
    teacher_epochs: int = 50
    teacher_lr: float = 1e-4
    teacher_weight_decay: float = 1e-4
    teacher_batch_size: int = 8        # Keep small for ≤ 4 GB VRAM
    teacher_warmup_epochs: int = 5

    # Loss weights (teacher)
    lambda_rank_teacher: float = 1.0   # Weight for ranking loss
    lambda_reg_teacher: float = 1.0    # Weight for regression (MOS) loss
    margin: float = 0.1               # Margin for ranking loss

    # ------------------------------------------------------------------ #
    # Training – Student
    # ------------------------------------------------------------------ #
    student_epochs: int = 50
    student_lr: float = 1e-4
    student_weight_decay: float = 1e-4
    student_batch_size: int = 8
    student_warmup_epochs: int = 5

    lambda_rank_student: float = 1.0
    lambda_reg_student: float = 1.0
    lambda_graph: float = 0.5          # Weight for graph-alignment distillation loss
    graph_loss_type: str = "kl"        # "mse" | "kl"  – graph alignment loss variant
    temperature: float = 0.07          # Softmax temperature for KL graph alignment

    # Early stopping (shared by teacher and student)
    early_stopping_patience: int = 10  # Stop after this many evals without improvement
    early_stopping_min_delta: float = 1e-4  # Minimum SRCC improvement to count

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    image_size: int = 224
    train_split: float = 0.8          # Fraction used for training
    val_split: float = 0.1
    # Remaining fraction → test
    num_workers: int = 2              # DataLoader workers (keep low on Colab/Kaggle)

    # ------------------------------------------------------------------ #
    # Memory Bank / k-NN graph (for relational distillation)
    # ------------------------------------------------------------------ #
    use_memory_bank: bool = True
    memory_bank_size: int = 1024      # Number of feature vectors stored
    knn_k: int = 8                    # Neighbours per node in the k-NN graph

    # ------------------------------------------------------------------ #
    # Multi-patch sampling
    # ------------------------------------------------------------------ #
    num_patches: int = 4          # number of random patches per image
    patch_resize: int = 384       # resize before random cropping
    # patch crop size = image_size (224) — controlled by existing image_size field

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    eval_batch_size: int = 16
    eval_every_n_epochs: int = 5      # Run val-set eval this often during training

    # ------------------------------------------------------------------ #
    # Hardware
    # ------------------------------------------------------------------ #
    device: str = "cuda"              # "cuda" | "cpu"
    amp: bool = True                  # Automatic mixed precision (saves VRAM)
    seed: int = 42

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    use_wandb: bool = False
    wandb_project: str = "rg-kd-iqa"
    wandb_run_name: str = ""

    def __post_init__(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        if not self.csv_path:
            # Default CSV location inferred from dataset name
            self.csv_path = os.path.join(
                self.data_root, self.dataset_name, "metadata.csv"
            )