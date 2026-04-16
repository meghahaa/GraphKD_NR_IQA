"""
student_model.py
----------------
Lightweight EfficientNet-B0 student for No-Reference IQA.

Architecture  (mirrors teacher_model.py conventions exactly)
------------
  Backbone  : efficientnet_b0  (from timm, num_classes=0 → pooled features)
              Output → (B, 1280)
  Proj Head : Linear(1280 → embed_dim) + BN + ReLU + Linear → L2-normalise
              (same ProjectionHead class reused from teacher_model.py)
  Reg Head  : Linear(embed_dim → 128) + ReLU + Linear(128 → 1) + Sigmoid
              (same RegressionHead class reused from teacher_model.py)

Forward returns the same three-tensor tuple as TeacherModel so that all
downstream loss functions and evaluate.py work without modification:
  - embeddings        : (B, embed_dim)        L2-normalised
  - predictions       : (B, 1)                MOS score in [0, 1]
  - backbone_features : (B, backbone_out_dim) raw pooled features

The student is intentionally kept lightweight:
  EfficientNet-B0 ≈ 5.3 M parameters  vs  Swin-Tiny ≈ 28 M parameters.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import timm

from config import Config
# Reuse the identical head classes from the teacher to keep consistency
from teacher import ProjectionHead, RegressionHead


class StudentModel(nn.Module):
    """
    Full student network: EfficientNet-B0 + ProjectionHead + RegressionHead.

    Parameters
    ----------
    cfg : Config

    Attributes
    ----------
    backbone          : timm EfficientNet-B0 (num_classes=0, pooled output)
    backbone_out_dim  : int  – 1280 for efficientnet_b0
    proj_head         : ProjectionHead – shared design with teacher
    reg_head          : RegressionHead – shared design with teacher
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()

        # ---- Backbone ------------------------------------------------ #
        self.backbone = timm.create_model(
            cfg.student_backbone,
            pretrained=cfg.pretrained,
            num_classes=0,   # Remove classifier; returns pooled feature vector
        )

        # Infer output dimension via a dry-run (works for any timm backbone)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, cfg.image_size, cfg.image_size)
            feat  = self.backbone(dummy)
        self.backbone_out_dim: int = feat.shape[-1]   # 1280 for efficientnet_b0

        # ---- Heads (identical design to teacher) --------------------- #
        self.proj_head = ProjectionHead(self.backbone_out_dim, cfg.embed_dim)
        self.reg_head  = RegressionHead(cfg.embed_dim)

    # ------------------------------------------------------------------ #

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.

        Parameters
        ----------
        x : Tensor of shape (B, 3, H, W)
            Batch of RGB images, normalised to ImageNet stats.

        Returns
        -------
        embeddings        : (B, embed_dim)         L2-normalised projected features
        predictions       : (B, 1)                 MOS prediction in [0, 1]
        backbone_features : (B, backbone_out_dim)  raw pooled backbone output
        """
        backbone_features = self.backbone(x)           # (B, 1280)
        embeddings        = self.proj_head(backbone_features)  # (B, embed_dim)
        predictions       = self.reg_head(embeddings)  # (B, 1)
        return embeddings, predictions, backbone_features

    # ------------------------------------------------------------------ #

    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return only L2-normalised embeddings (inference / eval helper).

        Parameters
        ----------
        x : (B, 3, H, W)

        Returns
        -------
        (B, embed_dim)
        """
        with torch.no_grad():
            backbone_features = self.backbone(x)
            embeddings        = self.proj_head(backbone_features)
        return embeddings

    def predict_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return only MOS predictions (inference helper).

        Parameters
        ----------
        x : (B, 3, H, W)

        Returns
        -------
        (B, 1)  quality scores in [0, 1]
        """
        with torch.no_grad():
            backbone_features = self.backbone(x)
            embeddings        = self.proj_head(backbone_features)
            scores            = self.reg_head(embeddings)
        return scores


def build_student(cfg: Config) -> StudentModel:
    """
    Factory: instantiate and return a StudentModel (on CPU).

    Parameters
    ----------
    cfg : Config

    Returns
    -------
    StudentModel
    """
    model = StudentModel(cfg)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(
        f"[student_model] backbone={cfg.student_backbone} "
        f"| backbone_out={model.backbone_out_dim} "
        f"| embed_dim={cfg.embed_dim} "
        f"| params={n_params:.2f}M"
    )
    return model