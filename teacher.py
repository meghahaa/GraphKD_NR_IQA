"""
teacher_model.py
----------------
Swin Transformer Tiny teacher for No-Reference IQA.

Architecture
------------
  Backbone  : swin_tiny_patch4_window7_224  (from timm)
              Output feature map → GlobalAvgPool → 768-d vector
  Proj Head : Linear(768 → embed_dim) + BN + ReLU  (used for ranking / distil.)
  Reg Head  : Linear(embed_dim → 1) + Sigmoid      (predicts normalised MOS)

Forward pass returns three tensors so that loss functions can pick what they need:
  - embeddings  (B, embed_dim)   – used for ranking & graph distillation
  - predictions (B, 1)           – MOS predictions in [0, 1]
  - backbone_features (B, 768)   – raw backbone output (optionally used)
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import timm

from config import Config


class ProjectionHead(nn.Module):
    """
    Two-layer projection MLP: Linear → BN → ReLU → Linear → L2-normalise.

    Parameters
    ----------
    in_dim  : int – input dimensionality (backbone output size)
    out_dim : int – embedding dimensionality (``cfg.embed_dim``)

    Input
    -----
    x : Tensor of shape (B, in_dim)

    Output
    ------
    z : Tensor of shape (B, out_dim), L2-normalised
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, in_dim)

        Returns
        -------
        (B, out_dim)  L2-normalised embeddings
        """
        z = self.net(x)                       # (B, out_dim)
        z = nn.functional.normalize(z, dim=1) # L2 norm along feature dim
        return z


class RegressionHead(nn.Module):
    """
    Lightweight MOS regressor: Linear → ReLU → Linear → Sigmoid.

    Parameters
    ----------
    in_dim : int – dimensionality of the projected embedding

    Input
    -----
    z : Tensor of shape (B, in_dim)

    Output
    ------
    score : Tensor of shape (B, 1), values in [0, 1]
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : (B, in_dim)

        Returns
        -------
        (B, 1) quality scores in [0, 1]
        """
        return self.net(z)


class TeacherModel(nn.Module):
    """
    Full teacher network: Swin-Tiny + ProjectionHead + RegressionHead.

    Parameters
    ----------
    cfg : Config

    Attributes
    ----------
    backbone          : timm Swin-Tiny (features_only style)
    backbone_out_dim  : int – backbone output channels (768 for swin_tiny)
    proj_head         : ProjectionHead
    reg_head          : RegressionHead
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()

        # ---- Backbone ------------------------------------------------ #
        self.backbone = timm.create_model(
            cfg.teacher_backbone,
            pretrained=cfg.pretrained,
            num_classes=0,          # Remove classification head; returns pooled features
        )

        # Infer backbone output dimension by a dry run
        with torch.no_grad():
            dummy = torch.zeros(1, 3, cfg.image_size, cfg.image_size)
            feat = self.backbone(dummy)
        self.backbone_out_dim: int = feat.shape[-1]  # e.g. 768 for swin_tiny

        # ---- Heads --------------------------------------------------- #
        self.proj_head = ProjectionHead(self.backbone_out_dim, cfg.embed_dim)
        self.reg_head = RegressionHead(cfg.embed_dim)

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
        embeddings        : (B, embed_dim)  L2-normalised projected features
        predictions       : (B, 1)          MOS prediction in [0, 1]
        backbone_features : (B, backbone_out_dim)  raw pooled backbone output
        """
        backbone_features = self.backbone(x)          # (B, 768)
        embeddings = self.proj_head(backbone_features) # (B, embed_dim)
        predictions = self.reg_head(embeddings)        # (B, 1)
        return embeddings, predictions, backbone_features

    # ------------------------------------------------------------------ #

    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience: return only L2-normalised embeddings.

        Parameters
        ----------
        x : (B, 3, H, W)

        Returns
        -------
        (B, embed_dim)
        """
        with torch.no_grad():
            backbone_features = self.backbone(x)
            embeddings = self.proj_head(backbone_features)
        return embeddings

    def predict_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience: return only MOS predictions.

        Parameters
        ----------
        x : (B, 3, H, W)

        Returns
        -------
        (B, 1) quality scores in [0, 1]
        """
        with torch.no_grad():
            backbone_features = self.backbone(x)
            embeddings = self.proj_head(backbone_features)
            scores = self.reg_head(embeddings)
        return scores


def build_teacher(cfg: Config) -> TeacherModel:
    """
    Factory function – instantiates and returns a TeacherModel.

    Parameters
    ----------
    cfg : Config

    Returns
    -------
    model : TeacherModel (on CPU; move to device externally)
    """
    model = TeacherModel(cfg)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(
        f"[teacher_model] backbone={cfg.teacher_backbone} "
        f"| backbone_out={model.backbone_out_dim} "
        f"| embed_dim={cfg.embed_dim} "
        f"| params={n_params:.2f}M"
    )
    return model