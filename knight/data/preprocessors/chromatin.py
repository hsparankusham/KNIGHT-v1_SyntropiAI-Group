"""scATAC-seq chromatin accessibility preprocessing pipeline.

Provides TF-IDF normalization, LSI dimensionality reduction, and peak
annotation for single-cell ATAC-seq data.  Follows muon / episcanpy
conventions and is designed for the KNIGHT v1 neuroimmune foundation model.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse, diags as sp_diags

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    # Peak filtering
    "min_cells_per_peak": 20,
    "min_peaks_per_cell": 500,
    "max_peaks_per_cell": 50_000,
    # TF-IDF
    "tfidf_scale_factor": 1e4,
    # LSI
    "n_lsi_components": 50,
    "drop_first_lsi": True,  # first LSI component often correlates with depth
    # Peak annotation distance thresholds (bp)
    "promoter_upstream": 2000,
    "promoter_downstream": 500,
}


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_CONFIG)
    if config:
        merged.update(config)
    return merged


# ---------------------------------------------------------------------------
# TF-IDF normalization
# ---------------------------------------------------------------------------


def _tfidf(adata: AnnData, scale_factor: float = 1e4) -> None:
    """In-place TF-IDF normalization on the count matrix.

    Uses the log-TF-IDF variant recommended for scATAC-seq:
        log(1 + TF * IDF * scale_factor)
    where TF = counts / total_per_cell and IDF = n_cells / n_cells_per_peak.
    """
    X = adata.X
    if issparse(X):
        X = X.astype(np.float64)
    else:
        X = np.array(X, dtype=np.float64)

    # Term frequency: normalize each cell to sum = 1
    cell_totals = np.asarray(X.sum(axis=1)).ravel()
    cell_totals[cell_totals == 0] = 1.0
    if issparse(X):
        inv_totals = sp_diags(1.0 / cell_totals)
        tf = inv_totals @ X
    else:
        tf = X / cell_totals[:, None]

    # Inverse document frequency
    n_cells = X.shape[0]
    peak_counts = np.asarray((X > 0).sum(axis=0)).ravel().astype(np.float64)
    peak_counts[peak_counts == 0] = 1.0
    idf = np.log1p(n_cells / peak_counts)

    # TF-IDF with log scaling
    if issparse(tf):
        idf_diag = sp_diags(idf)
        X_tfidf = tf @ idf_diag
        X_tfidf.data *= scale_factor
        X_tfidf = X_tfidf.log1p()
    else:
        X_tfidf = np.log1p(tf * idf[None, :] * scale_factor)

    adata.X = X_tfidf
    adata.uns["tfidf_scale_factor"] = scale_factor
    logger.info("TF-IDF normalization complete (scale_factor=%g)", scale_factor)


# ---------------------------------------------------------------------------
# LSI dimensionality reduction
# ---------------------------------------------------------------------------


def _lsi(
    adata: AnnData,
    n_components: int = 50,
    drop_first: bool = True,
) -> None:
    """Compute Latent Semantic Indexing (truncated SVD) on the TF-IDF matrix.

    Stores results in ``adata.obsm["X_lsi"]`` and
    ``adata.uns["lsi"]["stdev"]``.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.utils.extmath import randomized_svd

    n_total = n_components + (1 if drop_first else 0)
    svd = TruncatedSVD(n_components=n_total, algorithm="arpack")
    X_lsi = svd.fit_transform(adata.X)

    if drop_first:
        X_lsi = X_lsi[:, 1:]
        stdev = svd.singular_values_[1:]
        logger.info("Dropped first LSI component (depth-correlated)")
    else:
        stdev = svd.singular_values_

    adata.obsm["X_lsi"] = X_lsi
    adata.uns["lsi"] = {
        "stdev": stdev,
        "variance_ratio": svd.explained_variance_ratio_[
            1 if drop_first else 0 :
        ],
    }
    logger.info("LSI complete: %d components", X_lsi.shape[1])


# ---------------------------------------------------------------------------
# Peak annotation
# ---------------------------------------------------------------------------


def _parse_peak_name(peak: str) -> tuple[str, int, int]:
    """Parse a peak name like ``chr1-12345-67890`` or ``chr1:12345-67890``."""
    peak = peak.replace(":", "-")
    parts = peak.split("-")
    chrom = parts[0]
    start = int(parts[1])
    end = int(parts[2])
    return chrom, start, end


def annotate_peaks(
    adata_atac: AnnData,
    gene_annotation_df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> AnnData:
    """Annotate peaks as promoter, intergenic, or gene-body / enhancer.

    Adds columns to ``adata_atac.var``:

    - ``peak_type``: one of ``"promoter"``, ``"gene_body"``, ``"intergenic"``
    - ``nearest_gene``: gene symbol of the nearest TSS
    - ``distance_to_tss``: signed distance (bp) from peak center to TSS

    Parameters
    ----------
    adata_atac:
        AnnData with peak names as ``var_names`` in the format
        ``chr-start-end``.
    gene_annotation_df:
        DataFrame with at least columns ``chrom``, ``start``, ``end``,
        ``strand``, and ``gene_name``.  One row per gene / transcript.
    config:
        Overrides for promoter distance thresholds.

    Returns
    -------
    AnnData
        The same object, modified in place, also returned for chaining.
    """
    cfg = _resolve_config(config)
    up = cfg["promoter_upstream"]
    down = cfg["promoter_downstream"]

    gene_df = gene_annotation_df.copy()
    # Compute TSS
    gene_df["tss"] = np.where(
        gene_df["strand"] == "+", gene_df["start"], gene_df["end"]
    )

    peak_types: list[str] = []
    nearest_genes: list[str] = []
    distances: list[int] = []

    for peak_name in adata_atac.var_names:
        chrom, p_start, p_end = _parse_peak_name(peak_name)
        center = (p_start + p_end) // 2

        # Restrict to same chromosome
        chrom_genes = gene_df[gene_df["chrom"] == chrom]
        if chrom_genes.empty:
            peak_types.append("intergenic")
            nearest_genes.append("")
            distances.append(0)
            continue

        # Distance from peak center to each TSS
        dists = (chrom_genes["tss"].values - center).astype(np.int64)
        abs_dists = np.abs(dists)
        idx = int(np.argmin(abs_dists))
        min_dist = int(dists[idx])
        gene_name = chrom_genes.iloc[idx]["gene_name"]

        nearest_genes.append(gene_name)
        distances.append(min_dist)

        abs_d = abs(min_dist)
        strand = chrom_genes.iloc[idx]["strand"]

        # Determine if peak is in promoter region relative to TSS
        if strand == "+":
            in_promoter = -up <= min_dist <= down
        else:
            in_promoter = -down <= min_dist <= up

        if in_promoter:
            peak_types.append("promoter")
        elif (
            chrom_genes.iloc[idx]["start"] <= center <= chrom_genes.iloc[idx]["end"]
        ):
            peak_types.append("gene_body")
        else:
            peak_types.append("intergenic")

    adata_atac.var["peak_type"] = peak_types
    adata_atac.var["nearest_gene"] = nearest_genes
    adata_atac.var["distance_to_tss"] = distances

    counts = pd.Series(peak_types).value_counts().to_dict()
    logger.info("Peak annotation: %s", counts)
    return adata_atac


# ---------------------------------------------------------------------------
# Main preprocessing
# ---------------------------------------------------------------------------


def preprocess_chromatin(
    adata_atac: AnnData,
    config: dict[str, Any] | None = None,
) -> AnnData:
    """Run the full scATAC-seq preprocessing pipeline.

    Steps
    -----
    1. Filter peaks by minimum cell count.
    2. Filter cells by peak count range.
    3. TF-IDF normalization.
    4. LSI dimensionality reduction.
    5. Neighbor graph on LSI space.

    Parameters
    ----------
    adata_atac:
        AnnData with a binary or integer count matrix (cells x peaks).
    config:
        Preprocessing parameters.  Missing keys use :data:`DEFAULT_CONFIG`.

    Returns
    -------
    AnnData
        Processed AnnData with ``adata.obsm["X_lsi"]`` and a neighbor
        graph stored.
    """
    cfg = _resolve_config(config)
    logger.info(
        "Starting chromatin preprocessing (%d cells, %d peaks)",
        adata_atac.n_obs,
        adata_atac.n_vars,
    )

    # ------------------------------------------------------------------
    # 1. Filter peaks
    # ------------------------------------------------------------------
    import scanpy as sc

    sc.pp.filter_genes(adata_atac, min_cells=cfg["min_cells_per_peak"])
    logger.info("After peak filter (min_cells=%d): %d peaks", cfg["min_cells_per_peak"], adata_atac.n_vars)

    # ------------------------------------------------------------------
    # 2. Filter cells
    # ------------------------------------------------------------------
    sc.pp.filter_cells(adata_atac, min_genes=cfg["min_peaks_per_cell"])
    logger.info("After cell filter (min_peaks=%d): %d cells", cfg["min_peaks_per_cell"], adata_atac.n_obs)

    if cfg["max_peaks_per_cell"] is not None:
        peak_counts = np.asarray(adata_atac.X.sum(axis=1)).ravel()
        keep = peak_counts <= cfg["max_peaks_per_cell"]
        adata_atac = adata_atac[keep].copy()
        logger.info(
            "After max_peaks filter: %d cells", adata_atac.n_obs
        )

    # ------------------------------------------------------------------
    # 3. TF-IDF
    # ------------------------------------------------------------------
    _tfidf(adata_atac, scale_factor=cfg["tfidf_scale_factor"])

    # ------------------------------------------------------------------
    # 4. LSI
    # ------------------------------------------------------------------
    _lsi(
        adata_atac,
        n_components=cfg["n_lsi_components"],
        drop_first=cfg["drop_first_lsi"],
    )

    # ------------------------------------------------------------------
    # 5. Neighbors on LSI
    # ------------------------------------------------------------------
    sc.pp.neighbors(adata_atac, use_rep="X_lsi", n_neighbors=30)
    logger.info("Neighbor graph computed on LSI space")

    logger.info(
        "Chromatin preprocessing complete: %d cells x %d peaks",
        adata_atac.n_obs,
        adata_atac.n_vars,
    )
    return adata_atac
