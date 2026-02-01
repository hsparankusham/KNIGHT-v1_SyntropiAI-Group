"""scRNA-seq expression preprocessing pipeline.

Provides a scanpy-based pipeline for quality control, normalization, feature
selection, and dimensionality reduction of single-nucleus / single-cell
RNA-seq data.  Designed for the KNIGHT v1 neuroimmune foundation model but
general enough for any brain scRNA-seq experiment.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import scanpy as sc
from anndata import AnnData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    # Cell / gene filtering
    "min_genes_per_cell": 200,
    "max_genes_per_cell": 8000,
    "min_cells_per_gene": 10,
    # QC thresholds
    "max_pct_mito": 5.0,
    "max_pct_ribo": 30.0,
    # Normalization
    "target_sum": 1e4,
    # HVG selection
    "n_top_genes": 3000,
    "hvg_flavor": "seurat_v3",
    "hvg_batch_key": None,
    # Dimensionality reduction
    "n_pcs": 50,
    # Neighbors
    "n_neighbors": 30,
    "batch_key": None,  # used for batch-aware neighbor computation
}


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Merge user config with defaults."""
    merged = dict(DEFAULT_CONFIG)
    if config:
        merged.update(config)
    return merged


# ---------------------------------------------------------------------------
# QC helpers
# ---------------------------------------------------------------------------

_MITO_PREFIXES = ("MT-", "mt-")
_RIBO_PREFIXES = ("RPS", "RPL", "rps", "rpl")


def _annotate_qc(adata: AnnData) -> None:
    """Add mitochondrial and ribosomal percentage columns to ``adata.obs``."""
    adata.var["mt"] = adata.var_names.str.startswith(_MITO_PREFIXES)
    adata.var["ribo"] = adata.var_names.str.startswith(_RIBO_PREFIXES)
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt", "ribo"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )


def compute_qc_report(adata: AnnData) -> dict[str, Any]:
    """Compute a summary QC report for *adata*.

    Parameters
    ----------
    adata:
        Annotated data matrix (raw counts expected).

    Returns
    -------
    dict
        Dictionary with keys such as ``n_cells``, ``n_genes``,
        ``median_genes_per_cell``, ``median_counts_per_cell``,
        ``pct_mito_median``, ``pct_ribo_median``, etc.
    """
    _annotate_qc(adata)

    report: dict[str, Any] = {
        "n_cells": adata.n_obs,
        "n_genes": adata.n_vars,
        "median_genes_per_cell": float(np.median(adata.obs["n_genes_by_counts"])),
        "mean_genes_per_cell": float(np.mean(adata.obs["n_genes_by_counts"])),
        "median_counts_per_cell": float(np.median(adata.obs["total_counts"])),
        "mean_counts_per_cell": float(np.mean(adata.obs["total_counts"])),
        "pct_mito_median": float(np.median(adata.obs["pct_counts_mt"])),
        "pct_mito_mean": float(np.mean(adata.obs["pct_counts_mt"])),
        "pct_ribo_median": float(np.median(adata.obs["pct_counts_ribo"])),
        "pct_ribo_mean": float(np.mean(adata.obs["pct_counts_ribo"])),
    }
    logger.info("QC report: %s", report)
    return report


# ---------------------------------------------------------------------------
# Main preprocessing function
# ---------------------------------------------------------------------------


def preprocess_expression(
    adata: AnnData,
    config: dict[str, Any] | None = None,
) -> AnnData:
    """Run the full scRNA-seq preprocessing pipeline.

    The pipeline performs the following steps in order:

    1. Filter cells by gene count (``min_genes_per_cell``, ``max_genes_per_cell``).
    2. Filter genes by cell count (``min_cells_per_gene``).
    3. Compute QC metrics (mitochondrial %, ribosomal %).
    4. Filter cells exceeding QC thresholds.
    5. Preserve raw counts in ``adata.raw``.
    6. Normalize to ``target_sum`` per cell and apply ``log1p``.
    7. Select highly variable genes (HVGs).
    8. Subset to HVGs.
    9. PCA (``n_pcs`` components).
    10. Compute neighbors (optionally batch-aware via ``bbknn``).

    Parameters
    ----------
    adata:
        Annotated data matrix with raw counts in ``adata.X``.
    config:
        Dictionary of preprocessing parameters.  Missing keys fall back
        to :data:`DEFAULT_CONFIG`.

    Returns
    -------
    AnnData
        Processed AnnData object.  The ``.raw`` attribute stores the
        full-gene normalized counts prior to HVG subsetting.
    """
    cfg = _resolve_config(config)
    logger.info("Starting expression preprocessing (%d cells, %d genes)", adata.n_obs, adata.n_vars)

    # ------------------------------------------------------------------
    # 1. Filter cells by gene counts
    # ------------------------------------------------------------------
    sc.pp.filter_cells(adata, min_genes=cfg["min_genes_per_cell"])
    logger.info("After min_genes filter: %d cells", adata.n_obs)

    if cfg["max_genes_per_cell"] is not None:
        adata = adata[adata.obs["n_genes"] <= cfg["max_genes_per_cell"]].copy()
        logger.info("After max_genes filter: %d cells", adata.n_obs)

    # ------------------------------------------------------------------
    # 2. Filter genes by cell counts
    # ------------------------------------------------------------------
    sc.pp.filter_genes(adata, min_cells=cfg["min_cells_per_gene"])
    logger.info("After min_cells filter: %d genes", adata.n_vars)

    # ------------------------------------------------------------------
    # 3. QC metrics
    # ------------------------------------------------------------------
    _annotate_qc(adata)

    # ------------------------------------------------------------------
    # 4. QC filtering
    # ------------------------------------------------------------------
    n_before = adata.n_obs
    adata = adata[adata.obs["pct_counts_mt"] < cfg["max_pct_mito"]].copy()
    logger.info(
        "Mito filter (<%g%%): removed %d cells (%d remain)",
        cfg["max_pct_mito"],
        n_before - adata.n_obs,
        adata.n_obs,
    )

    n_before = adata.n_obs
    adata = adata[adata.obs["pct_counts_ribo"] < cfg["max_pct_ribo"]].copy()
    logger.info(
        "Ribo filter (<%g%%): removed %d cells (%d remain)",
        cfg["max_pct_ribo"],
        n_before - adata.n_obs,
        adata.n_obs,
    )

    # ------------------------------------------------------------------
    # 5. Normalize and log-transform
    # ------------------------------------------------------------------
    sc.pp.normalize_total(adata, target_sum=cfg["target_sum"])
    sc.pp.log1p(adata)

    # Store full-gene normalised counts before HVG subsetting
    adata.raw = adata.copy()
    logger.info("Normalization complete (target_sum=%g)", cfg["target_sum"])

    # ------------------------------------------------------------------
    # 6. Highly variable gene selection
    # ------------------------------------------------------------------
    hvg_kwargs: dict[str, Any] = {
        "n_top_genes": cfg["n_top_genes"],
        "flavor": cfg["hvg_flavor"],
    }
    if cfg["hvg_batch_key"] is not None:
        hvg_kwargs["batch_key"] = cfg["hvg_batch_key"]

    # seurat_v3 requires raw counts; use the raw layer
    if cfg["hvg_flavor"] == "seurat_v3":
        hvg_kwargs["layer"] = None  # scanpy will handle it
        # seurat_v3 needs counts; re-compute on raw
        sc.pp.highly_variable_genes(adata, **hvg_kwargs)
    else:
        sc.pp.highly_variable_genes(adata, **hvg_kwargs)

    n_hvg = int(adata.var["highly_variable"].sum())
    logger.info("Selected %d highly variable genes", n_hvg)
    adata = adata[:, adata.var["highly_variable"]].copy()

    # ------------------------------------------------------------------
    # 7. PCA
    # ------------------------------------------------------------------
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=cfg["n_pcs"], svd_solver="arpack")
    logger.info("PCA done (%d components)", cfg["n_pcs"])

    # ------------------------------------------------------------------
    # 8. Neighbors (optionally batch-aware)
    # ------------------------------------------------------------------
    batch_key = cfg["batch_key"]
    if batch_key is not None:
        try:
            import bbknn  # type: ignore[import-untyped]

            bbknn.bbknn(adata, batch_key=batch_key, n_pcs=cfg["n_pcs"])
            logger.info("Batch-aware neighbors via bbknn (batch_key=%s)", batch_key)
        except ImportError:
            logger.warning(
                "bbknn not installed; computing standard neighbors instead."
            )
            sc.pp.neighbors(adata, n_neighbors=cfg["n_neighbors"], n_pcs=cfg["n_pcs"])
    else:
        sc.pp.neighbors(adata, n_neighbors=cfg["n_neighbors"], n_pcs=cfg["n_pcs"])
        logger.info("Neighbors computed (n=%d)", cfg["n_neighbors"])

    logger.info(
        "Expression preprocessing complete: %d cells x %d genes",
        adata.n_obs,
        adata.n_vars,
    )
    return adata
