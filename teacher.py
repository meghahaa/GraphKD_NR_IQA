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

class AttentionPool(nn.Module):
    """
    Learned weighted aggregation across patch embeddings.

    Given P patch embeddings, computes a single image-level embedding
    as a weighted sum where weights are produced by a small MLP scorer.

    Architecture:
        score_i = MLP(z_i)          scalar logit per patch
        w_i     = softmax(scores)   normalised attention weight
        out     = Σ w_i · z_i       weighted sum

    Parameters
    ----------
    embed_dim : int – dimensionality of patch embeddings (= cfg.embed_dim)

    Input
    -----
    z : Tensor of shape (B, P, embed_dim)
        P patch embeddings per image, already L2-normalised.

    Output
    ------
    pooled : Tensor of shape (B, embed_dim)
             Attention-weighted sum, re-normalised to unit L2 norm.
    weights : Tensor of shape (B, P)
              Softmax attention weights (useful for visualisation).
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        # Small MLP: embed_dim → embed_dim//2 → 1
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        z : (B, P, embed_dim)

        Returns
        -------
        pooled  : (B, embed_dim)  L2-normalised aggregated embedding
        weights : (B, P)          attention weights
        """
        logits  = self.scorer(z).squeeze(-1)          # (B, P)
        weights = torch.softmax(logits, dim=1)         # (B, P)
        pooled  = torch.bmm(
            weights.unsqueeze(1), z
        ).squeeze(1)                                   # (B, embed_dim)
        pooled  = nn.functional.normalize(pooled, dim=1)  # L2 norm
        return pooled, weights


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
        self.proj_head   = ProjectionHead(self.backbone_out_dim, cfg.embed_dim)
        self.attn_pool   = AttentionPool(cfg.embed_dim)
        self.reg_head    = RegressionHead(cfg.embed_dim)
        # ------------------------------------------------------------------ #


    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Multi-patch forward pass.

        Parameters
        ----------
        x : Tensor of shape (B, P, 3, H, W)
            B images, each with P patch views (P = num_patches + 1).

        Returns
        -------
        embeddings        : (B, embed_dim)  attention-pooled L2-normalised embedding
        predictions       : (B, 1)          MOS prediction in [0, 1]
        patch_embeddings  : (B, P, embed_dim) per-patch embeddings before pooling
        """
        B, P, C, H, W = x.shape

        # Flatten batch and patch dims → process all patches in one forward pass
        x_flat = x.view(B * P, C, H, W)                   # (B*P, 3, H, W)
        feat_flat = self.backbone(x_flat)                   # (B*P, backbone_out_dim)
        emb_flat  = self.proj_head(feat_flat)               # (B*P, embed_dim)

        # Reshape back to (B, P, embed_dim)
        patch_embeddings = emb_flat.view(B, P, -1)          # (B, P, embed_dim)

        # Attention pooling across patches → single image embedding
        embeddings, _ = self.attn_pool(patch_embeddings)    # (B, embed_dim)

        # MOS prediction from pooled embedding
        predictions = self.reg_head(embeddings)             # (B, 1)

        return embeddings, predictions, patch_embeddings
    
    # ------------------------------------------------------------------ #

     extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (defB, P, 3, H, W)

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