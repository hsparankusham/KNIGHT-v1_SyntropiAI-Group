"""Perturbation response prediction head for KNIGHT v1.

This module predicts how a cell's gene expression profile changes in response
to a perturbation (CRISPRi gene knockout, small-molecule drug, etc.).  The
architecture uses cross-attention between the cell state embedding and a
perturbation embedding, then decodes the attended representation into a
predicted delta-expression vector over all genes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PerturbationHeadConfig:
    """Hyperparameters for :class:`PerturbationHead`."""

    d_model: int = 512
    n_genes: int = 60_697
    n_perturbations: int = 10_000
    """Size of the perturbation vocabulary (gene knockouts + drugs)."""

    d_perturbation: int = 256
    """Dimension of the perturbation embedding space."""

    n_cross_attn_heads: int = 4
    """Number of heads in the cross-attention layer."""

    n_decoder_layers: int = 2
    """Number of MLP layers in the delta-expression decoder."""

    dropout: float = 0.1


@dataclass
class PerturbationEncoderConfig:
    """Hyperparameters for :class:`PerturbationEncoder`."""

    n_perturbations: int = 10_000
    d_perturbation: int = 256
    n_genes: int = 60_697
    """Gene vocab size -- used when encoding gene-level knockouts."""

    dropout: float = 0.1


# ---------------------------------------------------------------------------
# Perturbation Encoder
# ---------------------------------------------------------------------------


class PerturbationEncoder(nn.Module):
    """Encode perturbation identity into a dense vector.

    Supports two perturbation modalities:

    * **Gene knockout (CRISPRi):** The perturbation ID corresponds to a gene
      in the same vocabulary used by the main encoder.  A separate embedding
      layer (optionally tied to the encoder gene embeddings) maps the gene
      ID to a perturbation vector.
    * **Drug / compound:** An independent embedding table maps drug IDs to
      vectors.

    Both pathways produce a vector of size ``d_perturbation``.

    Parameters
    ----------
    n_perturbations:
        Total number of perturbation tokens (genes + drugs).
    d_perturbation:
        Output embedding dimension.
    n_genes:
        Gene vocabulary size (used to distinguish gene knockouts from drug
        perturbations by ID range).
    dropout:
        Dropout applied after the embedding lookup.
    """

    def __init__(
        self,
        n_perturbations: int = 10_000,
        d_perturbation: int = 256,
        n_genes: int = 60_697,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_genes = n_genes
        self.d_perturbation = d_perturbation

        # Unified embedding table covering both gene knockouts and drugs.
        self.embedding = nn.Embedding(n_perturbations, d_perturbation)
        self.norm = nn.LayerNorm(d_perturbation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, perturbation_ids: torch.Tensor) -> torch.Tensor:
        """Encode perturbation IDs.

        Parameters
        ----------
        perturbation_ids:
            Integer IDs of shape ``(batch,)`` or ``(batch, n_perts)`` for
            combinatorial perturbations.

        Returns
        -------
        torch.Tensor
            Perturbation embeddings.  If input is 1-D the output is
            ``(batch, d_perturbation)``.  If input is 2-D (combinatorial)
            the individual perturbation embeddings are mean-pooled to
            produce ``(batch, d_perturbation)``.
        """
        emb = self.embedding(perturbation_ids)  # (..., D_p)
        emb = self.norm(emb)
        emb = self.dropout(emb)

        # Mean-pool combinatorial perturbations.
        if emb.dim() == 3:
            emb = emb.mean(dim=1)  # (B, D_p)

        return emb


# ---------------------------------------------------------------------------
# Cross-Attention Block
# ---------------------------------------------------------------------------


class _CrossAttentionBlock(nn.Module):
    """Single cross-attention layer: cell state attends to perturbation."""

    def __init__(self, d_model: int, d_perturbation: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        # Project perturbation embedding to d_model for cross-attention.
        self.perturbation_proj = nn.Linear(d_perturbation, d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        cell_emb: torch.Tensor,
        pert_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-attend cell embedding to perturbation embedding.

        Parameters
        ----------
        cell_emb:
            ``(batch, d_model)`` -- treated as a length-1 query sequence.
        pert_emb:
            ``(batch, d_perturbation)`` -- treated as a length-1 KV sequence.

        Returns
        -------
        torch.Tensor
            Updated cell embedding ``(batch, d_model)``.
        """
        # Unsqueeze to create sequence dimension.
        query = cell_emb.unsqueeze(1)  # (B, 1, D)
        kv = self.perturbation_proj(pert_emb).unsqueeze(1)  # (B, 1, D)

        # Cross-attention + residual
        attn_out, _ = self.cross_attn(query, kv, kv)
        query = self.norm1(query + attn_out)

        # FFN + residual
        ffn_out = self.ffn(query)
        out = self.norm2(query + ffn_out)

        return out.squeeze(1)  # (B, D)


# ---------------------------------------------------------------------------
# Perturbation Head
# ---------------------------------------------------------------------------


class PerturbationHead(nn.Module):
    """Predict post-perturbation delta expression from cell embeddings.

    The head works in three stages:

    1. **Perturbation encoding:** Convert perturbation IDs into dense
       embeddings via :class:`PerturbationEncoder`.
    2. **Cross-attention:** Let the cell-state embedding attend to the
       perturbation embedding so the model can contextualise the
       cell's response.
    3. **Delta decoder:** An MLP projects the attended embedding to a
       per-gene delta-expression prediction.

    Parameters
    ----------
    d_model:
        Dimension of the cell embeddings from the encoder.
    n_genes:
        Number of genes in the expression vector (output dimension).
    n_perturbations:
        Size of the perturbation vocabulary.
    d_perturbation:
        Dimension of the perturbation embedding space.
    n_cross_attn_heads:
        Number of attention heads in the cross-attention layer.
    n_decoder_layers:
        Depth of the delta-expression decoder MLP.
    dropout:
        Dropout probability.
    """

    def __init__(
        self,
        d_model: int = 512,
        n_genes: int = 60_697,
        n_perturbations: int = 10_000,
        d_perturbation: int = 256,
        n_cross_attn_heads: int = 4,
        n_decoder_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_genes = n_genes

        # -- Perturbation encoder -------------------------------------------
        self.perturbation_encoder = PerturbationEncoder(
            n_perturbations=n_perturbations,
            d_perturbation=d_perturbation,
            n_genes=n_genes,
            dropout=dropout,
        )

        # -- Cross-attention ------------------------------------------------
        self.cross_attention = _CrossAttentionBlock(
            d_model=d_model,
            d_perturbation=d_perturbation,
            n_heads=n_cross_attn_heads,
            dropout=dropout,
        )

        # -- Delta expression decoder ---------------------------------------
        decoder_layers = []
        in_dim = d_model
        for i in range(n_decoder_layers - 1):
            out_dim = max(d_model, n_genes // (2 ** (n_decoder_layers - 1 - i)))
            decoder_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim
        # Final projection to n_genes (no activation -- raw delta values).
        decoder_layers.append(nn.Linear(in_dim, n_genes))

        self.delta_decoder = nn.Sequential(*decoder_layers)

    # --------------------------------------------------------------------- #

    def forward(
        self,
        cell_embeddings: torch.Tensor,
        perturbation_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Predict delta expression after perturbation.

        Parameters
        ----------
        cell_embeddings:
            Cell-level embeddings from the encoder, shape ``(batch, d_model)``.
        perturbation_ids:
            Integer perturbation IDs, shape ``(batch,)`` or ``(batch, n_perts)``
            for combinatorial perturbations.

        Returns
        -------
        torch.Tensor
            Predicted change in gene expression, shape ``(batch, n_genes)``.
            Positive values indicate up-regulation, negative values indicate
            down-regulation relative to the unperturbed state.
        """
        # 1. Encode the perturbation identity.
        pert_emb = self.perturbation_encoder(perturbation_ids)  # (B, D_p)

        # 2. Cross-attend cell state to perturbation.
        attended = self.cross_attention(cell_embeddings, pert_emb)  # (B, D)

        # 3. Decode to per-gene delta expression.
        predicted_delta = self.delta_decoder(attended)  # (B, n_genes)

        return predicted_delta
