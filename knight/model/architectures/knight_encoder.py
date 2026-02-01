"""KNIGHT v1 Core Encoder -- Transformer encoder for single-cell gene expression.

This module implements the core encoder backbone for the KNIGHT (Knowledge-driven
Neuroimmune Inference of Gene-expression Heterogeneity in Tissue) foundation model.
The encoder ingests per-cell gene expression profiles (gene token IDs + continuous
expression values) and produces a fixed-size cell embedding via CLS-token pooling.

The architecture is designed to be initialised from pretrained scGPT / Geneformer
checkpoints and then fine-tuned on brain neuroimmune cell-state tasks.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class KnightEncoderConfig:
    """Hyperparameters for :class:`KnightEncoder`."""

    n_genes: int = 60_697
    """Size of the gene token vocabulary (including special tokens)."""

    d_model: int = 512
    """Hidden dimension throughout the encoder."""

    n_layers: int = 12
    """Number of Transformer encoder layers."""

    n_heads: int = 8
    """Number of attention heads."""

    d_ff: int | None = None
    """Feed-forward intermediate dimension.  Defaults to ``4 * d_model``."""

    dropout: float = 0.1
    """Dropout probability applied to attention weights and MLP layers."""

    max_seq_len: int = 3_000
    """Maximum number of expressed genes per cell (sequence length)."""

    use_flash_attn: bool = False
    """Whether to use Flash Attention 2 (requires ``flash_attn`` package)."""

    expression_encoder_mode: str = "direct"
    """How continuous expression values are projected: ``'direct'`` (MLP) or
    ``'binned'`` (discretise then embed)."""

    n_expression_bins: int = 51
    """Number of bins when ``expression_encoder_mode='binned'``."""

    cls_token_id: int = 0
    """Token ID reserved for the CLS token in the gene vocabulary."""

    pad_token_id: int = 1
    """Token ID reserved for padding."""

    def __post_init__(self) -> None:
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------


class _LearnablePositionalEncoding(nn.Module):
    """Learnable positional encoding for variable-length gene sequences."""

    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, seq_len: int) -> torch.Tensor:
        """Return positional embeddings ``(1, seq_len, d_model)``."""
        positions = torch.arange(seq_len, device=self.pe.weight.device)
        return self.pe(positions).unsqueeze(0)


class _ExpressionValueMLP(nn.Module):
    """Project scalar expression values to *d_model* via a small MLP."""

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, seq_len)`` scalars to ``(batch, seq_len, d_model)``."""
        return self.net(values.unsqueeze(-1))


class _ExpressionValueBinned(nn.Module):
    """Discretise expression values into bins and embed."""

    def __init__(self, n_bins: int, d_model: int) -> None:
        super().__init__()
        self.n_bins = n_bins
        self.embedding = nn.Embedding(n_bins, d_model)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Map ``(batch, seq_len)`` scalars to ``(batch, seq_len, d_model)``.

        Values are expected to be normalised to [0, 1] before binning.
        """
        bin_ids = torch.clamp(
            (values * (self.n_bins - 1)).long(), min=0, max=self.n_bins - 1
        )
        return self.embedding(bin_ids)


# ---------------------------------------------------------------------------
# Core Encoder
# ---------------------------------------------------------------------------


class KnightEncoder(nn.Module):
    """Transformer encoder for single-cell gene expression profiles.

    The encoder converts a variable-length set of (gene_id, expression_value)
    pairs into a single cell-level embedding vector of dimension *d_model*
    using CLS-token pooling.

    Parameters
    ----------
    config:
        A :class:`KnightEncoderConfig` instance (or a plain ``dict`` that will
        be unpacked into one).
    """

    def __init__(self, config: KnightEncoderConfig | dict | None = None, **kwargs: Any) -> None:
        super().__init__()

        if config is None:
            config = KnightEncoderConfig(**kwargs)
        elif isinstance(config, dict):
            config = KnightEncoderConfig(**config)
        self.config = config

        # -- Gene token embedding -------------------------------------------
        self.gene_embedding = nn.Embedding(
            config.n_genes, config.d_model, padding_idx=config.pad_token_id
        )

        # -- Expression value encoder ----------------------------------------
        if config.expression_encoder_mode == "direct":
            self.expression_encoder = _ExpressionValueMLP(config.d_model, config.dropout)
        elif config.expression_encoder_mode == "binned":
            self.expression_encoder = _ExpressionValueBinned(
                config.n_expression_bins, config.d_model
            )
        else:
            raise ValueError(
                f"Unknown expression_encoder_mode: {config.expression_encoder_mode!r}"
            )

        # -- Positional encoding --------------------------------------------
        # +1 for CLS token that is prepended.
        self.positional_encoding = _LearnablePositionalEncoding(
            config.max_seq_len + 1, config.d_model
        )

        # -- CLS token embedding --------------------------------------------
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)

        # -- Input projection layernorm + dropout ---------------------------
        self.input_norm = nn.LayerNorm(config.d_model)
        self.input_dropout = nn.Dropout(config.dropout)

        # -- Transformer encoder stack --------------------------------------
        if config.use_flash_attn:
            logger.info("Flash Attention requested -- building custom layers.")
            # Fall back to standard PyTorch impl; users can monkey-patch with
            # flash_attn.modules.mha if available.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,  # type: ignore[arg-type]
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN Transformer for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.n_layers
        )

        # -- Output projection (optional) -----------------------------------
        self.output_norm = nn.LayerNorm(config.d_model)

        self._init_weights()

    # --------------------------------------------------------------------- #

    def _init_weights(self) -> None:
        """Initialise weights following the standard Transformer recipe."""
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() >= 2:
                nn.init.trunc_normal_(param, std=0.02)
            elif "bias" in name:
                nn.init.zeros_(param)

    # --------------------------------------------------------------------- #

    def _build_input_embeddings(
        self,
        gene_ids: torch.Tensor,
        expression_values: torch.Tensor,
    ) -> torch.Tensor:
        """Combine gene token embeddings with expression value embeddings.

        Parameters
        ----------
        gene_ids:
            Integer gene token IDs, shape ``(batch, seq_len)``.
        expression_values:
            Continuous expression values, shape ``(batch, seq_len)``.

        Returns
        -------
        torch.Tensor
            Combined embeddings ``(batch, seq_len, d_model)``.
        """
        gene_emb = self.gene_embedding(gene_ids)  # (B, S, D)
        expr_emb = self.expression_encoder(expression_values)  # (B, S, D)
        return gene_emb + expr_emb

    # --------------------------------------------------------------------- #

    def forward(
        self,
        gene_ids: torch.Tensor,
        expression_values: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a batch of single-cell gene expression profiles.

        Parameters
        ----------
        gene_ids:
            Gene token IDs, shape ``(batch, seq_len)``.
        expression_values:
            Continuous expression values, shape ``(batch, seq_len)``.
        padding_mask:
            Boolean mask where ``True`` indicates padding positions,
            shape ``(batch, seq_len)``.  A column of ``False`` is prepended
            automatically for the CLS token.

        Returns
        -------
        torch.Tensor
            Cell-level embeddings, shape ``(batch, d_model)``, obtained via
            CLS-token pooling (position 0).
        """
        B, S = gene_ids.shape

        # 1. Input embeddings ------------------------------------------------
        x = self._build_input_embeddings(gene_ids, expression_values)  # (B, S, D)

        # 2. Prepend CLS token -----------------------------------------------
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, S+1, D)

        # 3. Positional encoding ----------------------------------------------
        x = x + self.positional_encoding(S + 1)

        # 4. LayerNorm + Dropout ----------------------------------------------
        x = self.input_norm(x)
        x = self.input_dropout(x)

        # 5. Padding mask (prepend False for CLS) -----------------------------
        if padding_mask is not None:
            cls_pad = torch.zeros(
                B, 1, dtype=torch.bool, device=padding_mask.device
            )
            padding_mask = torch.cat([cls_pad, padding_mask], dim=1)  # (B, S+1)

        # 6. Transformer encoder stack ----------------------------------------
        x = self.transformer(x, src_key_padding_mask=padding_mask)  # (B, S+1, D)

        # 7. CLS pooling ------------------------------------------------------
        cell_embedding = self.output_norm(x[:, 0, :])  # (B, D)

        return cell_embedding

    # --------------------------------------------------------------------- #

    @property
    def d_model(self) -> int:
        return self.config.d_model

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Compact Variant
# ---------------------------------------------------------------------------


class KnightMinEncoder(KnightEncoder):
    """Compact ~100M-parameter variant of :class:`KnightEncoder` for rapid
    prototyping and ablation studies.

    Default configuration: ``d_model=256, n_layers=6, n_heads=4``.
    """

    def __init__(
        self,
        n_genes: int = 60_697,
        dropout: float = 0.1,
        use_flash_attn: bool = False,
        **overrides: Any,
    ) -> None:
        config_kwargs: Dict[str, Any] = dict(
            n_genes=n_genes,
            d_model=256,
            n_layers=6,
            n_heads=4,
            d_ff=1024,
            dropout=dropout,
            use_flash_attn=use_flash_attn,
        )
        config_kwargs.update(overrides)
        super().__init__(config=KnightEncoderConfig(**config_kwargs))


# ---------------------------------------------------------------------------
# Pretrained Backbone Loading
# ---------------------------------------------------------------------------


def load_pretrained_backbone(
    checkpoint_path: str | Path,
    model_config: KnightEncoderConfig | dict | None = None,
    *,
    strict: bool = False,
    map_location: str | torch.device = "cpu",
) -> KnightEncoder:
    """Instantiate a :class:`KnightEncoder` and load pretrained weights.

    This helper handles the typical weight-key mismatches that arise when
    loading scGPT or Geneformer checkpoints into the KNIGHT encoder
    (e.g. different embedding layer names).

    Parameters
    ----------
    checkpoint_path:
        Path to a ``.pt`` / ``.bin`` checkpoint file.
    model_config:
        Encoder configuration.  If ``None``, sensible defaults are used.
    strict:
        Whether to enforce that every key in the checkpoint matches the model.
    map_location:
        Device mapping for ``torch.load``.

    Returns
    -------
    KnightEncoder
        Model with loaded weights (on the specified device).
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if model_config is None:
        model_config = KnightEncoderConfig()
    elif isinstance(model_config, dict):
        model_config = KnightEncoderConfig(**model_config)

    model = KnightEncoder(config=model_config)

    state_dict = torch.load(checkpoint_path, map_location=map_location, weights_only=True)

    # -- Handle common scGPT / Geneformer key remappings --------------------
    key_map: Dict[str, str] = {
        "encoder.embedding.weight": "gene_embedding.weight",
        "encoder.pos_embedding.weight": "positional_encoding.pe.weight",
        "transformer_encoder.": "transformer.",
    }

    remapped: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in key_map.items():
            if new_key.startswith(old_prefix) or old_prefix in new_key:
                new_key = new_key.replace(old_prefix, new_prefix)
        remapped[new_key] = value

    missing, unexpected = model.load_state_dict(remapped, strict=strict)
    if missing:
        logger.warning("Missing keys when loading checkpoint: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys when loading checkpoint: %s", unexpected)

    logger.info(
        "Loaded pretrained backbone from %s (%d params).",
        checkpoint_path,
        model.num_parameters,
    )
    return model
