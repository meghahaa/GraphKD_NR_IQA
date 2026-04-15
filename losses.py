"""
teacher_losses.py
-----------------
Loss functions used during **teacher pre-training**.

Three losses are defined:

1. ``MOSRegressionLoss``   – supervised regression to ground-truth MOS labels
                             (Huber / smooth-L1 by default, configurable).

2. ``PairwiseRankingLoss`` – margin-based ranking loss that penalises violations
                             of the ground-truth ordering between image pairs.
                             All O(B²) within-batch pairs are used.

3. ``TeacherTotalLoss``    – weighted sum of the two above.  This is the loss
                             actually optimised during teacher training.

Mathematical reference
----------------------
Regression:
    L_reg = SmoothL1( ŷ_i, y_i )          (Huber with δ=1 by default)

Ranking (pairwise margin loss):
    For each pair (i, j) where  y_i > y_j + ε:
        L_rank += max(0, margin − (ŷ_i − ŷ_j))
    Averaged over all valid pairs.

Total:
    L_total = λ_reg · L_reg + λ_rank · L_rank
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


# ------------------------------------------------------------------ #
# 1. Regression Loss
# ------------------------------------------------------------------ #

class MOSRegressionLoss(nn.Module):
    """
    Smooth-L1 (Huber) regression loss between predicted and true MOS.

    Parameters
    ----------
    beta : float
        Huber threshold (smooth-L1 parameter). Default 1.0.

    Inputs
    ------
    predictions : Tensor of shape (B, 1)  – model MOS predictions in [0, 1]
    targets     : Tensor of shape (B,)    – ground-truth MOS in [0, 1]

    Output
    ------
    loss : scalar Tensor
    """

    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()
        self.beta = beta

    def forward(
        self,
        predictions: torch.Tensor,   # (B, 1)
        targets: torch.Tensor,        # (B,)
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        predictions : (B, 1)
        targets     : (B,)

        Returns
        -------
        scalar Tensor – mean Huber loss over the batch
        """
        preds = predictions.squeeze(1)                         # (B,)
        return F.smooth_l1_loss(preds, targets, beta=self.beta)


# ------------------------------------------------------------------ #
# 2. Pairwise Ranking Loss
# ------------------------------------------------------------------ #

class PairwiseRankingLoss(nn.Module):
    """
    Margin-based pairwise ranking loss.

    For every ordered pair (i, j) in the batch where  y_i > y_j + ε  we
    enforce  ŷ_i − ŷ_j ≥ margin.  Pairs where the true MOS difference is
    below ``min_diff_threshold`` are ignored (near-ties add noise).

    Parameters
    ----------
    margin              : float – desired score separation (default 0.1)
    min_diff_threshold  : float – minimum |y_i − y_j| to form a valid pair
                                  (default 0.05)

    Inputs
    ------
    predictions : Tensor of shape (B, 1)  – model quality scores in [0, 1]
    targets     : Tensor of shape (B,)    – ground-truth MOS in [0, 1]

    Output
    ------
    loss : scalar Tensor (0 if no valid pairs exist)
    """

    def __init__(
        self,
        margin: float = 0.1,
        min_diff_threshold: float = 0.05,
    ) -> None:
        super().__init__()
        self.margin = margin
        self.min_diff_threshold = min_diff_threshold

    def forward(
        self,
        predictions: torch.Tensor,   # (B, 1)
        targets: torch.Tensor,        # (B,)
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        predictions : (B, 1)
        targets     : (B,)

        Returns
        -------
        scalar Tensor – mean margin ranking loss
        """
        preds = predictions.squeeze(1)   # (B,)
        B = preds.size(0)

        # Build all-pair difference matrices  →  (B, B)
        # mos_diff[i, j]  = y_i − y_j
        # pred_diff[i, j] = ŷ_i − ŷ_j
        mos_diff  = targets.unsqueeze(0) - targets.unsqueeze(1)   # (B, B) wait – fix order
        # We want row-i > col-j, so:
        mos_diff  = targets.unsqueeze(1) - targets.unsqueeze(0)   # (B, B): [i,j] = y_i - y_j
        pred_diff = preds.unsqueeze(1)   - preds.unsqueeze(0)     # (B, B): [i,j] = ŷ_i - ŷ_j

        # Valid pairs: y_i > y_j + threshold  →  mos_diff[i,j] > threshold
        valid_mask = mos_diff > self.min_diff_threshold            # (B, B) bool

        # Exclude diagonal (same sample)
        diag_mask = ~torch.eye(B, dtype=torch.bool, device=preds.device)
        valid_mask = valid_mask & diag_mask

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=preds.device, requires_grad=True)

        # Ranking violations: max(0, margin − (ŷ_i − ŷ_j))
        violations = F.relu(self.margin - pred_diff)               # (B, B)
        loss = violations[valid_mask].mean()
        return loss


# ------------------------------------------------------------------ #
# 3. Combined Teacher Loss
# ------------------------------------------------------------------ #

class TeacherTotalLoss(nn.Module):
    """
    Weighted combination of regression and pairwise ranking losses.

    L_total = λ_reg · L_reg + λ_rank · L_rank

    Parameters
    ----------
    cfg : Config
        Uses ``cfg.lambda_reg_teacher``, ``cfg.lambda_rank_teacher``,
        and ``cfg.margin``.

    Inputs
    ------
    predictions : Tensor of shape (B, 1)
    targets     : Tensor of shape (B,)

    Output
    ------
    total_loss  : scalar Tensor
    reg_loss    : scalar Tensor  (for logging)
    rank_loss   : scalar Tensor  (for logging)
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.lambda_reg  = cfg.lambda_reg_teacher
        self.lambda_rank = cfg.lambda_rank_teacher

        self.reg_loss  = MOSRegressionLoss(beta=1.0)
        self.rank_loss = PairwiseRankingLoss(
            margin=cfg.margin,
            min_diff_threshold=0.05,
        )

    def forward(
        self,
        predictions: torch.Tensor,   # (B, 1)
        targets: torch.Tensor,        # (B,)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        predictions : (B, 1) – model quality scores
        targets     : (B,)   – ground-truth MOS

        Returns
        -------
        total_loss : scalar Tensor
        reg_loss   : scalar Tensor
        rank_loss  : scalar Tensor
        """
        reg  = self.reg_loss(predictions, targets)
        rank = self.rank_loss(predictions, targets)
        total = self.lambda_reg * reg + self.lambda_rank * rank
        return total, reg, rank