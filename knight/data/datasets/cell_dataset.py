"""PyTorch datasets for KNIGHT training."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class CellExpressionDataset(Dataset):
    """Dataset yielding single-cell expression profiles for pretraining.

    Each sample is a cell represented by its gene IDs and expression values,
    suitable for masked expression modeling.
    """

    def __init__(
        self,
        adata: ad.AnnData,
        gene_vocab: dict[str, int],
        max_genes: int = 4000,
    ) -> None:
        """Initialize dataset.

        Args:
            adata: AnnData with cells x genes expression matrix.
            gene_vocab: Mapping from gene symbol to integer ID.
            max_genes: Maximum number of genes per cell (top by expression).
        """
        self.n_cells = adata.n_obs
        self.max_genes = max_genes

        # Map gene names to vocab IDs
        gene_names = list(adata.var_names)
        self.gene_ids = np.array([gene_vocab.get(g, 0) for g in gene_names])

        # Store expression matrix (sparse → dense chunks on access)
        self.X = adata.X
        self._is_sparse = hasattr(self.X, "toarray")

        logger.info(
            "CellExpressionDataset: %d cells, %d genes, max_genes=%d",
            self.n_cells,
            len(gene_names),
            max_genes,
        )

    def __len__(self) -> int:
        return self.n_cells

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Get expression vector for this cell
        row = self.X[idx]
        if self._is_sparse:
            row = row.toarray().squeeze()
        else:
            row = np.asarray(row).squeeze()

        # Select top-expressed genes (nonzero first, then by value)
        nonzero_mask = row > 0
        nonzero_idx = np.where(nonzero_mask)[0]

        if len(nonzero_idx) > self.max_genes:
            top_idx = nonzero_idx[np.argsort(row[nonzero_idx])[-self.max_genes :]]
        else:
            top_idx = nonzero_idx

        # Sort by gene ID for consistent ordering
        top_idx = top_idx[np.argsort(self.gene_ids[top_idx])]

        gene_ids = self.gene_ids[top_idx]
        values = row[top_idx]

        # Pad to max_genes
        n_genes = len(gene_ids)
        padded_ids = np.zeros(self.max_genes, dtype=np.int64)
        padded_vals = np.zeros(self.max_genes, dtype=np.float32)
        padding_mask = np.ones(self.max_genes, dtype=bool)

        padded_ids[:n_genes] = gene_ids
        padded_vals[:n_genes] = values
        padding_mask[:n_genes] = False

        return {
            "gene_ids": torch.from_numpy(padded_ids),
            "expression_values": torch.from_numpy(padded_vals),
            "padding_mask": torch.from_numpy(padding_mask),
            "n_genes": torch.tensor(n_genes, dtype=torch.long),
        }


class CellStateDataset(CellExpressionDataset):
    """Dataset for cell state classification fine-tuning.

    Extends CellExpressionDataset with cell state labels.
    """

    def __init__(
        self,
        adata: ad.AnnData,
        gene_vocab: dict[str, int],
        label_key: str = "cell_type",
        label_map: dict[str, int] | None = None,
        max_genes: int = 4000,
    ) -> None:
        super().__init__(adata, gene_vocab, max_genes)

        labels = adata.obs[label_key].values
        if label_map is None:
            unique_labels = sorted(set(labels))
            label_map = {l: i for i, l in enumerate(unique_labels)}

        self.label_map = label_map
        self.labels = np.array([label_map.get(l, -1) for l in labels])
        self.n_classes = len(label_map)

        logger.info(
            "CellStateDataset: %d classes, label_key='%s'",
            self.n_classes,
            label_key,
        )

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = super().__getitem__(idx)
        sample["label"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample


class PerturbationDataset(Dataset):
    """Dataset for perturbation response prediction.

    Each sample is a (control_cell, perturbation_id, delta_expression) tuple.
    """

    def __init__(
        self,
        control_adata: ad.AnnData,
        perturbed_adata: ad.AnnData,
        gene_vocab: dict[str, int],
        perturbation_key: str = "perturbation",
        max_genes: int = 4000,
    ) -> None:
        self.control_dataset = CellExpressionDataset(control_adata, gene_vocab, max_genes)

        # Compute delta expression (perturbed - control mean per perturbation)
        self.perturbation_ids = perturbed_adata.obs[perturbation_key].values
        unique_perts = sorted(set(self.perturbation_ids))
        self.pert_vocab = {p: i for i, p in enumerate(unique_perts)}

        # Store perturbed expression
        self.perturbed_X = perturbed_adata.X
        self._perturbed_sparse = hasattr(self.perturbed_X, "toarray")

        self.n_samples = perturbed_adata.n_obs
        logger.info(
            "PerturbationDataset: %d samples, %d perturbations",
            self.n_samples,
            len(unique_perts),
        )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Use a random control cell as baseline
        ctrl_idx = np.random.randint(len(self.control_dataset))
        control_sample = self.control_dataset[ctrl_idx]

        pert_id = self.pert_vocab[self.perturbation_ids[idx]]

        row = self.perturbed_X[idx]
        if self._perturbed_sparse:
            row = row.toarray().squeeze()
        else:
            row = np.asarray(row).squeeze()

        return {
            **control_sample,
            "perturbation_id": torch.tensor(pert_id, dtype=torch.long),
            "target_expression": torch.from_numpy(row.astype(np.float32)),
        }
