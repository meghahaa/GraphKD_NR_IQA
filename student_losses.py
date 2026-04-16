"""
student_losses.py
-----------------
Knowledge-distillation loss functions for student training.

Three complementary knowledge transfer mechanisms are implemented:

1. ``MOSRegressionLoss``      – score-level supervision to ground-truth MOS
                                (re-exported from teacher_losses for convenience)

2. ``PairwiseRankingLoss``    – pairwise ranking constraints that enforce the
                                student to respect the teacher's quality ordering
                                (re-exported from teacher_losses for convenience)

3. ``GraphAlignmentLoss``     – relational structure distillation.
                                The batch (optionally augmented by a memory bank)
                                is treated as a graph:
                                  • Nodes  = image embeddings
                                  • Edges  = pairwise feature similarity /
                                             quality distance / ranking indicator
                                The student is trained to match the teacher's
                                graph affinity matrix via MSE or KL divergence.

4. ``MemoryBank``             – fixed-size FIFO queue of (teacher_embedding,
                                teacher_score) pairs accumulated across batches.
                                Enables approximating large-context k-NN graphs
                                even with tiny batch sizes (4–8).

5. ``StudentTotalLoss``       – weighted combination of all three losses:
                                L = λ_reg · L_reg
                                  + λ_rank · L_rank
                                  + λ_graph · L_graph

Mathematical details
--------------------
Graph affinity matrix  A  of size (N, N) for N nodes:

    A[i, j] = σ(sim(f_i, f_j))     where σ = softmax over row j
                                    and   sim = cosine similarity

The student is trained to minimise:
    MSE mode : ||A_student − A_teacher||²_F
    KL  mode : KL( softmax(A_teacher / τ) || softmax(A_student / τ) )
               summed over all pairs, averaged over rows.

When a memory bank is used, the N nodes in the graph are:
    [current batch embeddings]  +  [memory bank embeddings]
giving N = B + memory_bank_size for the full relational context.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
# Re-export so student_train.py only needs to import from here
from losses import MOSRegressionLoss, PairwiseRankingLoss


# ------------------------------------------------------------------ #
# Memory Bank
# ------------------------------------------------------------------ #

class MemoryBank:
    """
    Fixed-size FIFO queue that stores teacher embeddings and quality scores
    accumulated across training batches.

    Enables building large-context relational graphs without increasing
    the per-batch GPU memory footprint.

    Parameters
    ----------
    size      : int – maximum number of stored vectors (e.g. 1024)
    embed_dim : int – dimensionality of stored embeddings
    device    : torch.device

    Attributes
    ----------
    embeddings : Tensor of shape (size, embed_dim)  – circular buffer
    scores     : Tensor of shape (size,)             – corresponding MOS scores
    ptr        : int  – write pointer (next slot to overwrite)
    is_full    : bool – True once the buffer has been written at least once fully
    """

    def __init__(
        self,
        size: int,
        embed_dim: int,
        device: torch.device,
    ) -> None:
        self.size      = size
        self.embed_dim = embed_dim
        self.device    = device

        # Pre-allocate on CPU, move on demand
        self.embeddings: torch.Tensor = torch.zeros(size, embed_dim)
        self.scores:     torch.Tensor = torch.zeros(size)
        self.ptr:  int  = 0
        self.is_full: bool = False

    # -------------------------------------------------------------- #

    def update(
        self,
        embeddings: torch.Tensor,   # (B, embed_dim)  L2-normalised
        scores:     torch.Tensor,   # (B,)            MOS in [0, 1]
    ) -> None:
        """
        Write a new batch of embeddings and scores into the circular buffer.

        Parameters
        ----------
        embeddings : (B, embed_dim)  detached teacher embeddings
        scores     : (B,)            corresponding teacher MOS predictions
        """
        B = embeddings.size(0)
        embeddings = embeddings.detach().cpu()
        scores     = scores.detach().cpu()

        # Handle wrap-around
        if self.ptr + B <= self.size:
            self.embeddings[self.ptr : self.ptr + B] = embeddings
            self.scores[self.ptr : self.ptr + B]     = scores
        else:
            # Split across the boundary
            tail = self.size - self.ptr
            self.embeddings[self.ptr :]  = embeddings[:tail]
            self.scores[self.ptr :]      = scores[:tail]
            self.embeddings[: B - tail]  = embeddings[tail:]
            self.scores[: B - tail]      = scores[tail:]

        self.ptr = (self.ptr + B) % self.size
        if self.ptr == 0 or self.ptr < B:
            self.is_full = True

    def get(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return all currently filled embeddings and scores.

        Parameters
        ----------
        device : torch.device

        Returns
        -------
        embeddings : (M, embed_dim)  where M = size if full, else ptr
        scores     : (M,)
        """
        n = self.size if self.is_full else max(self.ptr, 1)
        return (
            self.embeddings[:n].to(device),   # (M, embed_dim)
            self.scores[:n].to(device),        # (M,)
        )

    def __len__(self) -> int:
        return self.size if self.is_full else self.ptr


# ------------------------------------------------------------------ #
# Graph Affinity helpers
# ------------------------------------------------------------------ #

def _build_affinity_matrix(
    embeddings: torch.Tensor,   # (N, embed_dim)  L2-normalised
    scores:     torch.Tensor,   # (N,)
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Build a row-softmax affinity matrix from embeddings.

    The raw similarity is cosine similarity (embeddings are L2-normed,
    so dot-product == cosine similarity).

    Parameters
    ----------
    embeddings  : (N, embed_dim)  L2-normalised feature vectors
    scores      : (N,)            quality scores (used for sign; kept for
                                  extensibility to asymmetric kernels)
    temperature : float           softmax temperature τ

    Returns
    -------
    A : (N, N) row-stochastic affinity matrix
        A[i, j] = exp(sim(i,j)/τ) / Σ_k exp(sim(i,k)/τ)
    """
    # Cosine similarity matrix  (N, N);  embeddings already L2-normalised
    sim = torch.mm(embeddings, embeddings.t())   # (N, N)
    # Zero out diagonal (self-similarity)
    sim.fill_diagonal_(float("-inf"))
    A = F.softmax(sim / temperature, dim=1)      # (N, N) row-stochastic
    return A


# ------------------------------------------------------------------ #
# Graph Alignment Loss
# ------------------------------------------------------------------ #

class GraphAlignmentLoss(nn.Module):
    """
    Relational graph-structure distillation loss.

    Given teacher and student embedding sets (optionally augmented with
    memory bank vectors), build an affinity graph for each and minimise
    the divergence between them.

    Two variants are supported (``cfg.graph_loss_type``):
      "mse" – element-wise MSE between affinity matrices
      "kl"  – KL(A_teacher || A_student), averaged over rows

    Parameters
    ----------
    cfg : Config
        Uses: embed_dim, temperature, graph_loss_type,
              use_memory_bank, memory_bank_size, knn_k.

    Inputs  (via forward)
    ------
    t_emb   : (B, embed_dim)  teacher embeddings for current batch
    s_emb   : (B, embed_dim)  student embeddings for current batch
    t_scores: (B,)            teacher quality predictions (for memory bank)
    bank    : MemoryBank | None
              If provided, memory bank vectors are concatenated to the graph.

    Output
    ------
    loss : scalar Tensor
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.temperature     = cfg.temperature
        self.graph_loss_type = cfg.graph_loss_type.lower()
        assert self.graph_loss_type in ("mse", "kl"), (
            f"graph_loss_type must be 'mse' or 'kl', got '{self.graph_loss_type}'"
        )

    def forward(
        self,
        t_emb:    torch.Tensor,            # (B, embed_dim)
        s_emb:    torch.Tensor,            # (B, embed_dim)
        t_scores: torch.Tensor,            # (B,)
        bank:     Optional[MemoryBank] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        t_emb    : (B, embed_dim)  teacher embeddings (L2-normalised)
        s_emb    : (B, embed_dim)  student embeddings (L2-normalised)
        t_scores : (B,)            teacher quality predictions
        bank     : MemoryBank | None

        Returns
        -------
        scalar Tensor – graph alignment loss
        """
        device = t_emb.device

        # ---- Optionally augment with memory bank -------------------- #
        if bank is not None and len(bank) > 0:
            bank_emb, bank_scores = bank.get(device)       # (M, E), (M,)

            # Concatenate batch + bank;  L2-renormalise student side
            t_full = torch.cat([t_emb, bank_emb], dim=0)  # (B+M, E)
            s_full = torch.cat([
                s_emb,
                bank_emb,           # teacher bank used as anchor for student too
            ], dim=0)               # (B+M, E)
            scores_full = torch.cat([t_scores, bank_scores], dim=0)  # (B+M,)
        else:
            t_full      = t_emb       # (B, E)
            s_full      = s_emb       # (B, E)
            scores_full = t_scores    # (B,)

        # Re-normalise (guard against accumulated floating-point drift)
        t_full = F.normalize(t_full, dim=1)
        s_full = F.normalize(s_full, dim=1)

        # ---- Affinity matrices -------------------------------------- #
        A_t = _build_affinity_matrix(t_full, scores_full, self.temperature)  # (N, N)
        A_s = _build_affinity_matrix(s_full, scores_full, self.temperature)  # (N, N)

        # ---- Divergence -------------------------------------------- #
        if self.graph_loss_type == "mse":
            loss = F.mse_loss(A_s, A_t)
        else:  # "kl"
            # KL(P || Q) where P = teacher (target), Q = student
            # F.kl_div expects log-probabilities for Q
            loss = F.kl_div(
                A_s.clamp(min=1e-8).log(),   # log Q
                A_t.clamp(min=1e-8),         # P
                reduction="batchmean",
            )

        return loss


# ------------------------------------------------------------------ #
# Combined Student Distillation Loss
# ------------------------------------------------------------------ #

class StudentTotalLoss(nn.Module):
    """
    Weighted combination of all three distillation objectives:

        L = λ_reg  · L_reg(s_pred, y_mos)
          + λ_rank · L_rank(s_pred, y_mos)       [teacher ordering enforced]
          + λ_graph· L_graph(t_emb, s_emb, ...)  [relational structure]

    Note: ranking loss operates on STUDENT predictions vs ground-truth MOS
    (not teacher predictions) so the student learns the true ordering, not
    just mimicking the teacher's possibly imperfect ranks.

    Parameters
    ----------
    cfg : Config

    Inputs  (via forward)
    ------
    s_pred   : (B, 1)          student quality predictions
    targets  : (B,)            ground-truth MOS
    t_emb    : (B, embed_dim)  teacher embeddings (detached)
    s_emb    : (B, embed_dim)  student embeddings
    t_scores : (B,)            teacher quality predictions (for memory bank)
    bank     : MemoryBank | None

    Outputs
    -------
    total_loss  : scalar Tensor
    reg_loss    : scalar Tensor  (for logging)
    rank_loss   : scalar Tensor  (for logging)
    graph_loss  : scalar Tensor  (for logging)
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.lambda_reg   = cfg.lambda_reg_student
        self.lambda_rank  = cfg.lambda_rank_student
        self.lambda_graph = cfg.lambda_graph

        self.reg_loss   = MOSRegressionLoss(beta=1.0)
        self.rank_loss  = PairwiseRankingLoss(
            margin=cfg.margin,
            min_diff_threshold=0.05,
        )
        self.graph_loss = GraphAlignmentLoss(cfg)

    def forward(
        self,
        s_pred:   torch.Tensor,            # (B, 1)
        targets:  torch.Tensor,            # (B,)
        t_emb:    torch.Tensor,            # (B, embed_dim)  teacher, detached
        s_emb:    torch.Tensor,            # (B, embed_dim)  student
        t_scores: torch.Tensor,            # (B,)
        bank:     Optional[MemoryBank] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        s_pred   : (B, 1)
        targets  : (B,)
        t_emb    : (B, embed_dim)  detached teacher embeddings
        s_emb    : (B, embed_dim)  student embeddings (gradient-bearing)
        t_scores : (B,)            teacher scores (for memory bank augmentation)
        bank     : MemoryBank | None

        Returns
        -------
        total_loss : scalar Tensor
        reg_loss   : scalar Tensor
        rank_loss  : scalar Tensor
        graph_loss : scalar Tensor
        """
        reg   = self.reg_loss(s_pred, targets)
        rank  = self.rank_loss(s_pred, targets)
        graph = self.graph_loss(t_emb, s_emb, t_scores, bank)

        total = (
            self.lambda_reg   * reg
            + self.lambda_rank  * rank
            + self.lambda_graph * graph
        )
        return total, reg, rank, graph