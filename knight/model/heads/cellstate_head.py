"""Cell-state classification heads for KNIGHT v1.

This module provides two cell-state prediction heads:

* :class:`CellStateHead` -- flat multi-class classification over all cell
  states (e.g. 40 microglial states).
* :class:`HierarchicalCellStateHead` -- two-level hierarchical classification
  that first predicts a coarse cell type (7 classes, e.g. microglia,
  astrocyte, neuron ...) and then predicts a fine-grained cell state within
  the predicted coarse type (up to 40 fine states total).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CellStateHeadConfig:
    """Hyperparameters for :class:`CellStateHead`."""

    d_model: int = 512
    n_classes: int = 40
    dropout: float = 0.1


@dataclass
class HierarchicalCellStateHeadConfig:
    """Hyperparameters for :class:`HierarchicalCellStateHead`."""

    d_model: int = 512
    n_coarse_classes: int = 7
    n_fine_classes: int = 40
    dropout: float = 0.1
    coarse_to_fine_map: Dict[int, List[int]] | None = None
    """Optional mapping from coarse class index to the list of valid fine
    class indices.  When provided, fine logits for irrelevant classes are
    masked to ``-inf`` before softmax."""


# ---------------------------------------------------------------------------
# Flat Cell-State Head
# ---------------------------------------------------------------------------


class CellStateHead(nn.Module):
    """MLP head for flat cell-state classification.

    Architecture::

        d_model -> LayerNorm -> Linear(d_model, d_model//2) -> GELU
        -> Dropout -> Linear(d_model//2, n_classes)

    Parameters
    ----------
    d_model:
        Dimension of the incoming cell embeddings.
    n_classes:
        Number of cell-state classes.
    dropout:
        Dropout probability.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_classes: int = 40,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(self, cell_embeddings: torch.Tensor) -> torch.Tensor:
        """Predict cell-state logits.

        Parameters
        ----------
        cell_embeddings:
            Shape ``(batch, d_model)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, n_classes)``.
        """
        return self.head(cell_embeddings)


# ---------------------------------------------------------------------------
# Hierarchical Cell-State Head
# ---------------------------------------------------------------------------


class HierarchicalCellStateHead(nn.Module):
    """Two-level hierarchical cell-state classifier.

    **Level 1 (coarse):** Predict the broad cell type (e.g. microglia,
    astrocyte, neuron, oligodendrocyte, OPC, endothelial, pericyte).

    **Level 2 (fine):** Predict the specific activation / disease state
    within that cell type (e.g. DAM-1, DAM-2, homeostatic microglia ...).

    When a ``coarse_to_fine_map`` is provided the fine logits for states
    that do not belong to the predicted coarse class are masked to ``-inf``
    so the fine softmax is only computed over valid sub-states.

    Parameters
    ----------
    d_model:
        Dimension of the incoming cell embeddings.
    n_coarse_classes:
        Number of coarse cell-type classes (default 7).
    n_fine_classes:
        Total number of fine cell-state classes across all coarse types
        (default 40).
    dropout:
        Dropout probability.
    coarse_to_fine_map:
        Optional dict ``{coarse_idx: [fine_idx, ...]}``.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_coarse_classes: int = 7,
        n_fine_classes: int = 40,
        dropout: float = 0.1,
        coarse_to_fine_map: Dict[int, List[int]] | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_coarse_classes = n_coarse_classes
        self.n_fine_classes = n_fine_classes

        # -- Coarse head ----------------------------------------------------
        self.coarse_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_coarse_classes),
        )

        # -- Fine head ------------------------------------------------------
        # Receives the cell embedding concatenated with a coarse-class
        # embedding so it is aware of the coarse prediction.
        self.coarse_embedding = nn.Embedding(n_coarse_classes, d_model // 4)
        fine_input_dim = d_model + d_model // 4

        self.fine_head = nn.Sequential(
            nn.LayerNorm(fine_input_dim),
            nn.Linear(fine_input_dim, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_fine_classes),
        )

        # -- Coarse-to-fine masking -----------------------------------------
        self.coarse_to_fine_map = coarse_to_fine_map
        if coarse_to_fine_map is not None:
            # Pre-compute a boolean mask matrix (n_coarse, n_fine) where True
            # means the fine class is valid for that coarse class.
            mask = torch.zeros(n_coarse_classes, n_fine_classes, dtype=torch.bool)
            for coarse_idx, fine_indices in coarse_to_fine_map.items():
                for fi in fine_indices:
                    mask[coarse_idx, fi] = True
            self.register_buffer("_fine_mask", mask)
        else:
            self.register_buffer(
                "_fine_mask",
                torch.ones(n_coarse_classes, n_fine_classes, dtype=torch.bool),
            )

    # --------------------------------------------------------------------- #

    def _mask_fine_logits(
        self,
        fine_logits: torch.Tensor,
        coarse_preds: torch.Tensor,
    ) -> torch.Tensor:
        """Zero out (set to ``-inf``) fine logits for irrelevant states.

        Parameters
        ----------
        fine_logits:
            Raw fine logits ``(batch, n_fine_classes)``.
        coarse_preds:
            Predicted coarse class indices ``(batch,)``.

        Returns
        -------
        torch.Tensor
            Masked fine logits ``(batch, n_fine_classes)``.
        """
        # Gather the valid-fine mask for each sample's coarse prediction.
        valid = self._fine_mask[coarse_preds]  # (B, n_fine)
        return fine_logits.masked_fill(~valid, float("-inf"))

    # --------------------------------------------------------------------- #

    def forward(
        self,
        cell_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Two-level hierarchical prediction.

        Parameters
        ----------
        cell_embeddings:
            Shape ``(batch, d_model)``.

        Returns
        -------
        coarse_logits:
            Shape ``(batch, n_coarse_classes)``.
        fine_logits:
            Shape ``(batch, n_fine_classes)``.  If a ``coarse_to_fine_map``
            was provided, irrelevant positions are ``-inf``.
        """
        # Coarse prediction
        coarse_logits = self.coarse_head(cell_embeddings)  # (B, C)
        coarse_preds = coarse_logits.argmax(dim=-1)  # (B,)

        # Condition fine head on coarse prediction
        coarse_emb = self.coarse_embedding(coarse_preds)  # (B, D//4)
        fine_input = torch.cat([cell_embeddings, coarse_emb], dim=-1)  # (B, D + D//4)
        fine_logits = self.fine_head(fine_input)  # (B, F)

        # Mask invalid fine states
        fine_logits = self._mask_fine_logits(fine_logits, coarse_preds)

        return coarse_logits, fine_logits
