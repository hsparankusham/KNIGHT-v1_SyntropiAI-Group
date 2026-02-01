"""KNIGHT v1 Full Model -- Encoder + dynamic task heads.

This module composes the :class:`KnightEncoder` backbone with one or more
task-specific prediction heads (cell-state classification, perturbation
response, etc.).  Heads are registered dynamically from a configuration dict
so that the same model class can serve pretraining, fine-tuning, and
inference across all KNIGHT tasks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

import torch
import torch.nn as nn

from knight.model.architectures.knight_encoder import (
    KnightEncoder,
    KnightEncoderConfig,
    KnightMinEncoder,
    load_pretrained_backbone,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Head registry -- maps string names to concrete head classes lazily so that
# head modules are only imported when actually needed.
# ---------------------------------------------------------------------------

_HEAD_REGISTRY: Dict[str, str] = {
    "cellstate": "knight.model.heads.cellstate_head.CellStateHead",
    "hierarchical_cellstate": "knight.model.heads.cellstate_head.HierarchicalCellStateHead",
    "perturbation": "knight.model.heads.perturbation_head.PerturbationHead",
}


def _import_head_class(dotted_path: str) -> Type[nn.Module]:
    """Import a head class from its fully-qualified dotted path."""
    module_path, cls_name = dotted_path.rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class HeadConfig:
    """Configuration for a single task head."""

    head_type: str
    """Registered head name (e.g. ``'cellstate'``, ``'perturbation'``)."""

    head_kwargs: Dict[str, Any] = field(default_factory=dict)
    """Keyword arguments forwarded to the head constructor."""


@dataclass
class KnightConfig:
    """Top-level configuration for :class:`KnightModel`."""

    # -- Encoder hyperparameters --------------------------------------------
    n_genes: int = 60_697
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    d_ff: int | None = None
    dropout: float = 0.1
    max_seq_len: int = 3_000
    use_flash_attn: bool = False
    expression_encoder_mode: str = "direct"
    n_expression_bins: int = 51

    # -- Head configurations ------------------------------------------------
    head_configs: Dict[str, HeadConfig] = field(default_factory=dict)
    """Mapping from head name to :class:`HeadConfig`."""

    # -- Pretrained backbone ------------------------------------------------
    pretrained_checkpoint: str | None = None
    """Optional path to a pretrained scGPT / Geneformer checkpoint."""

    # -- Training strategy --------------------------------------------------
    freeze_encoder_at_init: bool = False
    """If ``True`` the encoder parameters are frozen after construction
    (useful for head-only fine-tuning warmup)."""

    def to_encoder_config(self) -> KnightEncoderConfig:
        """Extract the encoder-specific subset of hyperparameters."""
        return KnightEncoderConfig(
            n_genes=self.n_genes,
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            dropout=self.dropout,
            max_seq_len=self.max_seq_len,
            use_flash_attn=self.use_flash_attn,
            expression_encoder_mode=self.expression_encoder_mode,
            n_expression_bins=self.n_expression_bins,
        )


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------


class KnightModel(nn.Module):
    """Full KNIGHT model: encoder backbone + pluggable task heads.

    Parameters
    ----------
    config:
        A :class:`KnightConfig` instance describing the encoder and heads.

    Example
    -------
    >>> cfg = KnightConfig(
    ...     d_model=256, n_layers=6, n_heads=4,
    ...     head_configs={
    ...         "cellstate": HeadConfig("cellstate", {"d_model": 256, "n_classes": 40}),
    ...     },
    ... )
    >>> model = KnightModel(cfg)
    >>> out = model(batch, task="cellstate")
    """

    def __init__(self, config: KnightConfig) -> None:
        super().__init__()
        self.config = config

        # -- Build encoder ---------------------------------------------------
        if config.pretrained_checkpoint is not None:
            self.encoder = load_pretrained_backbone(
                config.pretrained_checkpoint,
                config.to_encoder_config(),
            )
        else:
            self.encoder = KnightEncoder(config=config.to_encoder_config())

        # -- Build heads dynamically -----------------------------------------
        self.heads = nn.ModuleDict()
        for head_name, head_cfg in config.head_configs.items():
            self._register_head(head_name, head_cfg)

        # -- Optional encoder freeze -----------------------------------------
        if config.freeze_encoder_at_init:
            self.freeze_encoder()

        logger.info(
            "KnightModel built: %d encoder params, %d head(s) [%s].",
            self.encoder.num_parameters,
            len(self.heads),
            ", ".join(self.heads.keys()),
        )

    # ------------------------------------------------------------------ #
    # Head management
    # ------------------------------------------------------------------ #

    def _register_head(self, name: str, head_cfg: HeadConfig) -> None:
        """Instantiate and register a task head."""
        if head_cfg.head_type not in _HEAD_REGISTRY:
            raise ValueError(
                f"Unknown head type {head_cfg.head_type!r}. "
                f"Available: {list(_HEAD_REGISTRY.keys())}"
            )
        cls = _import_head_class(_HEAD_REGISTRY[head_cfg.head_type])
        head = cls(**head_cfg.head_kwargs)
        self.heads[name] = head
        logger.info("Registered head %r (%s).", name, head_cfg.head_type)

    def add_head(self, name: str, head_cfg: HeadConfig) -> None:
        """Add a new task head after construction."""
        self._register_head(name, head_cfg)

    def remove_head(self, name: str) -> None:
        """Remove a task head by name."""
        if name in self.heads:
            del self.heads[name]

    # ------------------------------------------------------------------ #
    # Encoder freeze / unfreeze
    # ------------------------------------------------------------------ #

    def freeze_encoder(self) -> None:
        """Freeze all encoder parameters (for head-only fine-tuning)."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder frozen (%d params).", self.encoder.num_parameters)

    def unfreeze_encoder(self) -> None:
        """Unfreeze all encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        logger.info("Encoder unfrozen (%d params).", self.encoder.num_parameters)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def get_cell_embeddings(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run the encoder only and return cell embeddings.

        Parameters
        ----------
        batch:
            Dictionary with keys ``gene_ids``, ``expression_values``, and
            optionally ``padding_mask``.

        Returns
        -------
        torch.Tensor
            Cell embeddings of shape ``(batch_size, d_model)``.
        """
        return self.encoder(
            gene_ids=batch["gene_ids"],
            expression_values=batch["expression_values"],
            padding_mask=batch.get("padding_mask"),
        )

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        task: str = "cellstate",
    ) -> torch.Tensor | Tuple[torch.Tensor, ...]:
        """Full forward pass: encoder then the requested task head.

        Parameters
        ----------
        batch:
            Dictionary containing at minimum ``gene_ids`` and
            ``expression_values``.  Additional keys may be required by
            specific heads (e.g. ``perturbation_ids`` for the perturbation
            head).
        task:
            Name of the registered head to route through.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            Output of the selected task head.
        """
        if task not in self.heads:
            raise ValueError(
                f"Task head {task!r} not registered. Available: {list(self.heads.keys())}"
            )

        cell_embeddings = self.get_cell_embeddings(batch)
        head = self.heads[task]

        # Perturbation head requires extra inputs
        if task == "perturbation" and "perturbation_ids" in batch:
            return head(cell_embeddings, batch["perturbation_ids"])

        return head(cell_embeddings)

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:  # pragma: no cover
        head_str = ", ".join(f"{k}: {type(v).__name__}" for k, v in self.heads.items())
        return (
            f"KnightModel(\n"
            f"  encoder={self.encoder.__class__.__name__}"
            f"(d={self.config.d_model}, L={self.config.n_layers}, H={self.config.n_heads}),\n"
            f"  heads={{{head_str}}},\n"
            f"  params={self.num_parameters:,}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_knight_model(config: KnightConfig | dict) -> KnightModel:
    """Factory function that builds a :class:`KnightModel` from a config.

    Parameters
    ----------
    config:
        Either a :class:`KnightConfig` dataclass or a plain dictionary.
        When a dict is provided, top-level keys map to ``KnightConfig``
        fields and nested ``head_configs`` values are converted to
        :class:`HeadConfig` instances automatically.

    Returns
    -------
    KnightModel
    """
    if isinstance(config, dict):
        raw_heads = config.pop("head_configs", {})
        head_configs: Dict[str, HeadConfig] = {}
        for name, hcfg in raw_heads.items():
            if isinstance(hcfg, dict):
                head_configs[name] = HeadConfig(**hcfg)
            else:
                head_configs[name] = hcfg
        config = KnightConfig(**config, head_configs=head_configs)

    return KnightModel(config)
