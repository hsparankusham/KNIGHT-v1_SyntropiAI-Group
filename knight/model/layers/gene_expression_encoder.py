"""Custom layers for encoding gene expression data in KNIGHT v1.

This module provides the foundational building blocks that sit between raw
single-cell RNA-seq data and the Transformer encoder:

* :class:`ContinuousValueEncoder` -- maps continuous expression values to
  dense vectors via either binned embedding or direct MLP projection.
* :class:`GeneTokenizer` -- converts gene names (symbols) to integer IDs
  using a fixed vocabulary file.
* :class:`ExpressionMasker` -- implements masked gene expression modelling
  (analogous to MLM in NLP) for self-supervised pretraining.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ContinuousValueEncoder
# ---------------------------------------------------------------------------


@dataclass
class ContinuousValueEncoderConfig:
    """Configuration for :class:`ContinuousValueEncoder`."""

    d_model: int = 512
    mode: str = "direct"
    """Encoding mode: ``'direct'`` (MLP projection) or ``'binned'``
    (discretise then embed)."""

    n_bins: int = 51
    """Number of bins when ``mode='binned'``.  The range [0, 1] is split
    into ``n_bins`` equal-width intervals."""

    dropout: float = 0.1


class ContinuousValueEncoder(nn.Module):
    """Encode continuous gene expression values into dense d_model vectors.

    Two modes are supported:

    * **direct** -- A small MLP maps each scalar value ``(batch, n_genes, 1)``
      to ``(batch, n_genes, d_model)``.  This preserves the full continuous
      information and is the default for fine-tuning.
    * **binned** -- Values (assumed normalised to [0, 1]) are discretised
      into ``n_bins`` equal-width bins and looked up in an embedding table.
      This mirrors the approach used by Geneformer and can be useful for
      compatibility with pretrained checkpoints.

    Parameters
    ----------
    d_model:
        Output embedding dimension.
    mode:
        ``'direct'`` or ``'binned'``.
    n_bins:
        Number of bins (only used when ``mode='binned'``).
    dropout:
        Dropout probability applied inside the MLP (direct mode).
    """

    def __init__(
        self,
        d_model: int = 512,
        mode: str = "direct",
        n_bins: int = 51,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.mode = mode
        self.n_bins = n_bins

        if mode == "direct":
            self.encoder = nn.Sequential(
                nn.Linear(1, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, d_model),
                nn.LayerNorm(d_model),
            )
        elif mode == "binned":
            self.bin_embedding = nn.Embedding(n_bins, d_model)
            self.norm = nn.LayerNorm(d_model)
        else:
            raise ValueError(f"Unknown ContinuousValueEncoder mode: {mode!r}")

    # --------------------------------------------------------------------- #

    def _discretise(self, values: torch.Tensor) -> torch.Tensor:
        """Convert normalised [0, 1] values to integer bin indices."""
        bin_ids = torch.clamp(
            (values * (self.n_bins - 1)).long(), min=0, max=self.n_bins - 1
        )
        return bin_ids

    # --------------------------------------------------------------------- #

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Encode expression values.

        Parameters
        ----------
        values:
            Continuous expression values of shape ``(batch, n_genes)``.
            For binned mode these should be normalised to [0, 1].

        Returns
        -------
        torch.Tensor
            Encoded values of shape ``(batch, n_genes, d_model)``.
        """
        if self.mode == "direct":
            return self.encoder(values.unsqueeze(-1))  # (B, G, 1) -> (B, G, D)
        else:
            bin_ids = self._discretise(values)  # (B, G)
            emb = self.bin_embedding(bin_ids)  # (B, G, D)
            return self.norm(emb)


# ---------------------------------------------------------------------------
# GeneTokenizer
# ---------------------------------------------------------------------------


class GeneTokenizer:
    """Convert gene names (symbols) to integer token IDs.

    The tokenizer maintains a deterministic mapping from gene symbol strings
    to integer IDs.  Special tokens are reserved at the start of the
    vocabulary:

    * ``[CLS]`` -- index 0
    * ``[PAD]`` -- index 1
    * ``[MASK]`` -- index 2
    * ``[UNK]`` -- index 3

    Parameters
    ----------
    gene_vocab_path:
        Path to a JSON file mapping gene symbols to integer IDs.
        If ``None`` the tokenizer is initialised empty and must be
        populated via :meth:`build_vocab`.
    """

    SPECIAL_TOKENS: Dict[str, int] = {
        "[CLS]": 0,
        "[PAD]": 1,
        "[MASK]": 2,
        "[UNK]": 3,
    }

    NUM_SPECIAL: int = len(SPECIAL_TOKENS)

    def __init__(self, gene_vocab_path: str | Path | None = None) -> None:
        self._token_to_id: Dict[str, int] = dict(self.SPECIAL_TOKENS)
        self._id_to_token: Dict[int, str] = {v: k for k, v in self._token_to_id.items()}

        if gene_vocab_path is not None:
            self._load_vocab(Path(gene_vocab_path))

    # --------------------------------------------------------------------- #

    def _load_vocab(self, path: Path) -> None:
        """Load a vocabulary JSON file."""
        if not path.exists():
            raise FileNotFoundError(f"Gene vocabulary not found: {path}")

        with open(path, "r") as fh:
            vocab: Dict[str, int] = json.load(fh)

        # Merge with special tokens (special tokens always win).
        for gene, idx in vocab.items():
            if gene not in self.SPECIAL_TOKENS:
                self._token_to_id[gene] = idx
                self._id_to_token[idx] = gene

        logger.info(
            "Loaded gene vocabulary with %d tokens (incl. %d special) from %s.",
            len(self._token_to_id),
            self.NUM_SPECIAL,
            path,
        )

    # --------------------------------------------------------------------- #

    @classmethod
    def build_vocab(
        cls,
        gene_list: List[str],
        save_path: str | Path | None = None,
    ) -> "GeneTokenizer":
        """Create a new tokenizer from a list of gene symbols.

        Parameters
        ----------
        gene_list:
            Ordered list of gene symbols (e.g. from an AnnData ``.var_names``).
        save_path:
            If provided, the vocabulary is saved as JSON to this path.

        Returns
        -------
        GeneTokenizer
            A fully populated tokenizer.
        """
        tokenizer = cls()
        offset = cls.NUM_SPECIAL
        for i, gene in enumerate(gene_list):
            gene = gene.strip()
            if gene and gene not in tokenizer._token_to_id:
                idx = offset + i
                tokenizer._token_to_id[gene] = idx
                tokenizer._id_to_token[idx] = gene

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as fh:
                json.dump(tokenizer._token_to_id, fh, indent=2)
            logger.info("Saved gene vocabulary (%d tokens) to %s.", len(tokenizer), save_path)

        return tokenizer

    # --------------------------------------------------------------------- #

    def encode(self, gene_names: List[str]) -> torch.Tensor:
        """Convert a list of gene symbols to a 1-D tensor of integer IDs.

        Unknown genes are mapped to the ``[UNK]`` token.

        Parameters
        ----------
        gene_names:
            Gene symbols to encode.

        Returns
        -------
        torch.Tensor
            Integer IDs, shape ``(len(gene_names),)``.
        """
        unk_id = self.SPECIAL_TOKENS["[UNK]"]
        ids = [self._token_to_id.get(g.strip(), unk_id) for g in gene_names]
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> List[str]:
        """Convert a tensor of integer IDs back to gene symbols."""
        return [self._id_to_token.get(int(i), "[UNK]") for i in ids]

    # --------------------------------------------------------------------- #

    @property
    def vocab_size(self) -> int:
        return len(self._token_to_id)

    @property
    def cls_token_id(self) -> int:
        return self.SPECIAL_TOKENS["[CLS]"]

    @property
    def pad_token_id(self) -> int:
        return self.SPECIAL_TOKENS["[PAD]"]

    @property
    def mask_token_id(self) -> int:
        return self.SPECIAL_TOKENS["[MASK]"]

    def __len__(self) -> int:
        return self.vocab_size

    def __contains__(self, gene: str) -> bool:
        return gene in self._token_to_id

    def __repr__(self) -> str:
        return f"GeneTokenizer(vocab_size={self.vocab_size})"


# ---------------------------------------------------------------------------
# ExpressionMasker
# ---------------------------------------------------------------------------


class ExpressionMasker(nn.Module):
    """Masked gene expression modelling (self-supervised pretraining).

    Analogous to masked language modelling (MLM) in BERT, this module
    randomly masks a fraction of gene expression values so the model must
    predict the original values from context.

    Masking strategy (following BERT conventions):

    * With probability ``mask_ratio``, a gene is selected for masking.
    * Of the selected genes:
      - 80% have their expression value set to 0.0 (equivalent to [MASK]).
      - 10% have their expression value replaced with a random value.
      - 10% are left unchanged (but still included in the loss).

    Parameters
    ----------
    mask_ratio:
        Fraction of genes to mask per cell.
    replace_with_zero_prob:
        Probability of replacing a masked gene's value with 0.
    random_replace_prob:
        Probability of replacing a masked gene's value with a random value.
    """

    def __init__(
        self,
        mask_ratio: float = 0.15,
        replace_with_zero_prob: float = 0.8,
        random_replace_prob: float = 0.1,
    ) -> None:
        super().__init__()
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")
        self.mask_ratio = mask_ratio
        self.replace_with_zero_prob = replace_with_zero_prob
        self.random_replace_prob = random_replace_prob
        # Remaining probability = keep original.
        self.keep_prob = 1.0 - replace_with_zero_prob - random_replace_prob

    # --------------------------------------------------------------------- #

    def forward(
        self,
        expression_values: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply random masking to expression values.

        Parameters
        ----------
        expression_values:
            Original expression values, shape ``(batch, n_genes)``.
        padding_mask:
            Boolean mask where ``True`` marks padding positions (these are
            never masked), shape ``(batch, n_genes)``.

        Returns
        -------
        masked_values:
            Expression values with masking applied, same shape as input.
        mask_indices:
            Boolean tensor indicating which positions were selected for
            masking (i.e. must be predicted), shape ``(batch, n_genes)``.
        original_values:
            The unmodified expression values (for computing the loss).
        """
        B, G = expression_values.shape
        device = expression_values.device

        original_values = expression_values.clone()

        # 1. Decide which positions to mask.
        rand = torch.rand(B, G, device=device)
        mask_indices = rand < self.mask_ratio

        # Never mask padding positions.
        if padding_mask is not None:
            mask_indices = mask_indices & ~padding_mask

        # 2. Within masked positions, decide the replacement strategy.
        strategy_rand = torch.rand(B, G, device=device)

        # Positions that get zeroed out (analogous to [MASK]).
        zero_mask = mask_indices & (strategy_rand < self.replace_with_zero_prob)

        # Positions that get a random replacement value.
        random_mask = mask_indices & (
            (strategy_rand >= self.replace_with_zero_prob)
            & (strategy_rand < self.replace_with_zero_prob + self.random_replace_prob)
        )
        # Remaining masked positions keep their original value.

        # 3. Apply replacements.
        masked_values = expression_values.clone()
        masked_values[zero_mask] = 0.0
        if random_mask.any():
            # Random values sampled uniformly from the observed range in the batch.
            val_min = expression_values.min()
            val_max = expression_values.max()
            random_vals = torch.rand(random_mask.sum(), device=device) * (val_max - val_min) + val_min
            masked_values[random_mask] = random_vals

        return masked_values, mask_indices, original_values
