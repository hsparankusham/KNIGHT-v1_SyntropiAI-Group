"""Custom loss functions for KNIGHT v1 training.

Provides masked reconstruction losses for pretraining, correlation-aware losses
for perturbation prediction, focal loss for imbalanced cell state classification,
and contrastive losses for embedding regularisation.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Masked reconstruction loss
# ---------------------------------------------------------------------------

class MaskedMSELoss(nn.Module):
    """Mean-squared error computed only over masked (corrupted) positions.

    During masked gene-expression pretraining the model is asked to reconstruct
    the original expression values for a random subset of genes.  This loss
    ignores all unmasked positions so that the gradient signal comes exclusively
    from the reconstruction targets.

    Parameters
    ----------
    reduction : str
        ``"mean"`` (default) averages over all masked elements; ``"sum"``
        returns the total squared error; ``"none"`` returns per-element losses.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.reduction = reduction

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked MSE.

        Parameters
        ----------
        predicted : Tensor [B, G]
            Model predictions for all genes.
        target : Tensor [B, G]
            Ground-truth expression values.
        mask : Tensor [B, G]
            Boolean or binary mask — ``True``/``1`` for positions that were
            masked and should contribute to the loss.

        Returns
        -------
        Tensor
            Scalar loss (or per-element if ``reduction="none"``).
        """
        mask = mask.float()
        sq_error = (predicted - target) ** 2 * mask

        if self.reduction == "none":
            return sq_error
        if self.reduction == "sum":
            return sq_error.sum()

        n_masked = mask.sum().clamp(min=1.0)
        return sq_error.sum() / n_masked


# ---------------------------------------------------------------------------
# Pearson correlation loss
# ---------------------------------------------------------------------------

class PearsonCorrelationLoss(nn.Module):
    """1 - mean Pearson correlation, computed gene-wise across a batch.

    The Pearson correlation is computed independently for each gene (column)
    across the batch dimension, then averaged.  Minimising this loss
    encourages the model to capture per-gene variance structure.

    Parameters
    ----------
    eps : float
        Small constant added to standard deviations to avoid division by zero.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute 1 - mean gene-wise Pearson *r*.

        Parameters
        ----------
        predicted, target : Tensor [B, G]
            Predicted and ground-truth expression matrices.

        Returns
        -------
        Tensor
            Scalar loss in [0, 2].
        """
        # Centre each gene across the batch
        pred_centered = predicted - predicted.mean(dim=0, keepdim=True)
        tgt_centered = target - target.mean(dim=0, keepdim=True)

        # Covariance per gene
        cov = (pred_centered * tgt_centered).sum(dim=0)
        pred_std = pred_centered.pow(2).sum(dim=0).sqrt().clamp(min=self.eps)
        tgt_std = tgt_centered.pow(2).sum(dim=0).sqrt().clamp(min=self.eps)

        pearson_r = cov / (pred_std * tgt_std)          # [G]
        mean_r = pearson_r.mean()
        return 1.0 - mean_r


# ---------------------------------------------------------------------------
# Combined perturbation loss
# ---------------------------------------------------------------------------

class CombinedPerturbationLoss(nn.Module):
    """Weighted combination of MSE and Pearson-correlation loss.

    Used when fine-tuning for perturbation-response prediction, where we want
    both pointwise accuracy (MSE) and correct correlation structure (Pearson).

    Parameters
    ----------
    mse_weight : float
        Weight for the MSE component (default ``1.0``).
    pearson_weight : float
        Weight for the Pearson-correlation component (default ``1.0``).
    eps : float
        Epsilon forwarded to :class:`PearsonCorrelationLoss`.
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        pearson_weight: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.mse_weight = mse_weight
        self.pearson_weight = pearson_weight
        self.mse_loss = nn.MSELoss()
        self.pearson_loss = PearsonCorrelationLoss(eps=eps)

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined loss.

        Returns
        -------
        loss : Tensor
            Weighted scalar loss.
        components : dict
            Individual ``"mse"`` and ``"pearson"`` values for logging.
        """
        mse = self.mse_loss(predicted, target)
        pearson = self.pearson_loss(predicted, target)
        loss = self.mse_weight * mse + self.pearson_weight * pearson
        components = {
            "mse": mse.item(),
            "pearson": pearson.item(),
        }
        return loss, components


# ---------------------------------------------------------------------------
# Focal loss for imbalanced classification
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal loss for multi-class classification with class imbalance.

    Focal loss down-weights well-classified examples so that the model
    concentrates on hard, misclassified samples — particularly useful when
    rare neuroimmune cell states are heavily outnumbered by common ones.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

    Parameters
    ----------
    gamma : float
        Focusing parameter (default ``2.0``).  ``gamma=0`` recovers standard
        cross-entropy.
    alpha : Tensor | None
        Optional per-class weights of shape ``[C]``.
    reduction : str
        ``"mean"`` (default), ``"sum"``, or ``"none"``.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha: Optional[torch.Tensor] = None

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss.

        Parameters
        ----------
        logits : Tensor [B, C]
            Unnormalised class scores.
        targets : Tensor [B]
            Integer class labels.

        Returns
        -------
        Tensor
            Scalar (or per-sample if ``reduction="none"``).
        """
        log_probs = F.log_softmax(logits, dim=-1)                     # [B, C]
        probs = log_probs.exp()

        # Gather the probability / log-prob for the correct class
        targets_onehot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        pt = (probs * targets_onehot).sum(dim=-1)                     # [B]
        log_pt = (log_probs * targets_onehot).sum(dim=-1)             # [B]

        focal_weight = (1.0 - pt).pow(self.gamma)
        loss = -focal_weight * log_pt                                 # [B]

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            loss = loss * alpha_t

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ---------------------------------------------------------------------------
# Contrastive batch loss for pretraining regularisation
# ---------------------------------------------------------------------------

class ContrastiveBatchLoss(nn.Module):
    """Supervised contrastive loss over cell-type labels within a batch.

    Pulls embeddings of cells that share the same cell type together while
    pushing embeddings of different cell types apart.  Can be used as an
    auxiliary regulariser during masked-expression pretraining when cell-type
    metadata is available.

    This implements the *SupCon* formulation from Khosla et al., NeurIPS 2020.

    Parameters
    ----------
    temperature : float
        Scaling temperature for the softmax (default ``0.07``).
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")
        self.temperature = temperature

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute supervised contrastive loss.

        Parameters
        ----------
        embeddings : Tensor [B, D]
            L2-normalised cell embeddings.
        labels : Tensor [B]
            Integer cell-type labels.

        Returns
        -------
        Tensor
            Scalar loss.
        """
        device = embeddings.device
        batch_size = embeddings.size(0)
        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # L2-normalise (in case caller did not)
        embeddings = F.normalize(embeddings, p=2, dim=-1)

        # Pairwise cosine similarity matrix  [B, B]
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Mask: same label → positive pair (excluding self)
        labels = labels.view(-1, 1)
        positive_mask = (labels == labels.T).float()                  # [B, B]
        self_mask = torch.eye(batch_size, device=device)
        positive_mask = positive_mask - self_mask                     # exclude self

        # For numerical stability, subtract the max from each row
        logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        # Denominator: sum over all negatives + positives (excluding self)
        neg_mask = 1.0 - self_mask
        exp_logits = torch.exp(logits) * neg_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp(min=1e-12))

        # Mean log-probability over positive pairs
        n_positives = positive_mask.sum(dim=1).clamp(min=1.0)
        mean_log_prob = (positive_mask * log_prob).sum(dim=1) / n_positives

        loss = -mean_log_prob.mean()
        return loss
