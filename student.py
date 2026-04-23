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
from teacher import ProjectionHead, RegressionHead, AttentionPool


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
        self.attn_pool = AttentionPool(cfg.embed_dim)  
        self.reg_head  = RegressionHead(cfg.embed_dim)

    # ------------------------------------------------------------------ #

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Multi-patch forward pass.

        Parameters
        ----------
        x : (B, P, 3, H, W)

        Returns
        -------
        embeddings       : (B, embed_dim)     attention-pooled L2-normalised
        predictions      : (B, 1)             MOS prediction in [0, 1]
        patch_embeddings : (B, P, embed_dim)  per-patch before pooling
        """
        B, P, C, H, W = x.shape
        x_flat        = x.view(B * P, C, H, W)
        feat_flat     = self.backbone(x_flat)              # (B*P, backbone_out_dim)
        emb_flat      = self.proj_head(feat_flat)          # (B*P, embed_dim)
        patch_embeddings = emb_flat.view(B, P, -1)         # (B, P, embed_dim)
        embeddings, _ = self.attn_pool(patch_embeddings)   # (B, embed_dim)
        predictions   = self.reg_head(embeddings)          # (B, 1)
        return embeddings, predictions, patch_embeddings

    # ------------------------------------------------------------------ #

    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, P, 3, H, W)

        Returns
        -------
        (B, embed_dim)  attention-pooled L2-normalised embeddings
        """
        with torch.no_grad():
            embeddings, _, _ = self.forward(x)
        return embeddings

    def predict_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, P, 3, H, W)

        Returns
        -------
        (B, 1)
        """
        with torch.no_grad():
            _, predictions, _ = self.forward(x)
        return predictions


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