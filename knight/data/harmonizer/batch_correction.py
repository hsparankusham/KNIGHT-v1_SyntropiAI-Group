"""Cross-dataset batch correction and harmonization.

Provides multiple integration strategies (scVI, Harmony, Scanorama) for
combining single-cell datasets from different studies, donors, and
sequencing platforms.  Includes integration quality metrics (kBET, LISI,
ASW, graph connectivity) to evaluate and compare methods.

Designed for the KNIGHT v1 neuroimmune foundation model pipeline.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.sparse import issparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    # General
    "batch_key": "dataset",
    "label_key": "cell_type",
    # Harmony
    "harmony_n_pcs": 50,
    "harmony_max_iter": 20,
    # scVI
    "scvi_n_latent": 30,
    "scvi_n_layers": 2,
    "scvi_max_epochs": 200,
    "scvi_early_stopping": True,
    # Scanorama
    "scanorama_knn": 20,
    # Neighbors (post-integration)
    "n_neighbors": 30,
}


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_CONFIG)
    if config:
        merged.update(config)
    return merged


# ---------------------------------------------------------------------------
# Integration backends
# ---------------------------------------------------------------------------


def _integrate_harmony(adata: AnnData, cfg: dict[str, Any]) -> AnnData:
    """Harmony integration on PCA space."""
    try:
        import harmonypy  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "harmonypy is required for Harmony integration. "
            "Install with: pip install harmonypy"
        ) from exc

    batch_key = cfg["batch_key"]
    n_pcs = cfg["harmony_n_pcs"]

    if "X_pca" not in adata.obsm:
        logger.info("Running PCA before Harmony (%d components)", n_pcs)
        sc.pp.pca(adata, n_comps=n_pcs, svd_solver="arpack")

    logger.info("Running Harmony (batch_key=%s, max_iter=%d)", batch_key, cfg["harmony_max_iter"])
    ho = harmonypy.run_harmony(
        adata.obsm["X_pca"][:, :n_pcs],
        adata.obs,
        batch_key,
        max_iter_harmony=cfg["harmony_max_iter"],
    )
    adata.obsm["X_pca_harmony"] = ho.Z_corr.T
    sc.pp.neighbors(adata, use_rep="X_pca_harmony", n_neighbors=cfg["n_neighbors"])
    logger.info("Harmony integration complete")
    return adata


def _integrate_scvi(adata: AnnData, cfg: dict[str, Any]) -> AnnData:
    """scVI latent-space integration."""
    try:
        import scvi as scvi_module  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "scvi-tools is required for scVI integration. "
            "Install with: pip install scvi-tools"
        ) from exc

    batch_key = cfg["batch_key"]

    logger.info(
        "Setting up scVI model (n_latent=%d, n_layers=%d, batch_key=%s)",
        cfg["scvi_n_latent"],
        cfg["scvi_n_layers"],
        batch_key,
    )
    scvi_module.model.SCVI.setup_anndata(adata, batch_key=batch_key)
    model = scvi_module.model.SCVI(
        adata,
        n_latent=cfg["scvi_n_latent"],
        n_layers=cfg["scvi_n_layers"],
    )
    model.train(
        max_epochs=cfg["scvi_max_epochs"],
        early_stopping=cfg["scvi_early_stopping"],
    )

    adata.obsm["X_scVI"] = model.get_latent_representation()
    sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=cfg["n_neighbors"])
    logger.info("scVI integration complete (%d latent dims)", cfg["scvi_n_latent"])
    return adata


def _integrate_scanorama(adata: AnnData, cfg: dict[str, Any]) -> AnnData:
    """Scanorama integration."""
    try:
        import scanorama  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "scanorama is required. Install with: pip install scanorama"
        ) from exc

    batch_key = cfg["batch_key"]
    batches = adata.obs[batch_key].unique().tolist()
    adatas_split = [adata[adata.obs[batch_key] == b].copy() for b in batches]

    logger.info("Running Scanorama on %d batches (knn=%d)", len(batches), cfg["scanorama_knn"])
    corrected, _ = scanorama.correct_scanpy(adatas_split, knn=cfg["scanorama_knn"])

    # Reassemble
    import anndata as ad

    adata_out = ad.concat(corrected, join="outer")
    adata_out.obsm["X_scanorama"] = adata_out.X.toarray() if issparse(adata_out.X) else adata_out.X.copy()
    sc.pp.neighbors(adata_out, use_rep="X_scanorama", n_neighbors=cfg["n_neighbors"])
    logger.info("Scanorama integration complete")
    return adata_out


_METHODS = {
    "harmony": _integrate_harmony,
    "scvi": _integrate_scvi,
    "scanorama": _integrate_scanorama,
}


# ---------------------------------------------------------------------------
# Integration anchor selection
# ---------------------------------------------------------------------------


def select_integration_anchors(
    adatas: list[AnnData],
    label_key: str = "cell_type",
    min_overlap: int = 2,
) -> list[str]:
    """Identify cell types shared across multiple datasets.

    Parameters
    ----------
    adatas:
        List of AnnData objects, each with *label_key* in ``.obs``.
    label_key:
        Column in ``.obs`` containing cell type labels.
    min_overlap:
        Minimum number of datasets a cell type must appear in to be
        considered an anchor.

    Returns
    -------
    list[str]
        Sorted list of anchor cell type labels.
    """
    from collections import Counter

    type_counter: Counter[str] = Counter()
    for ad in adatas:
        if label_key not in ad.obs.columns:
            logger.warning("label_key %r not found in an AnnData; skipping.", label_key)
            continue
        unique_types = ad.obs[label_key].dropna().unique().tolist()
        for ct in unique_types:
            type_counter[ct] += 1

    anchors = sorted(ct for ct, count in type_counter.items() if count >= min_overlap)
    logger.info(
        "Found %d anchor cell types shared across >= %d datasets: %s",
        len(anchors),
        min_overlap,
        anchors,
    )
    return anchors


# ---------------------------------------------------------------------------
# Integration quality metrics
# ---------------------------------------------------------------------------


def compute_batch_metrics(
    adata: AnnData,
    batch_key: str,
    label_key: str,
) -> dict[str, float]:
    """Compute integration quality metrics.

    Parameters
    ----------
    adata:
        Integrated AnnData with a neighbor graph and embeddings.
    batch_key:
        Column in ``.obs`` identifying the batch / dataset.
    label_key:
        Column in ``.obs`` identifying cell type labels.

    Returns
    -------
    dict[str, float]
        Dictionary with the following metrics:

        - **kbet_acceptance_rate**: fraction of cells whose local
          neighborhood passes the kBET chi-squared test (higher is
          better for mixing).
        - **ilisi**: integration LISI -- median inverse Simpson index
          over batches (higher means better mixing).
        - **clisi**: cell-type LISI -- median over labels (lower means
          cell types remain compact).
        - **asw_batch**: average silhouette width computed per batch
          label (closer to 0 is better for batch mixing).
        - **asw_bio**: average silhouette width computed per biological
          label (higher is better for bio conservation).
        - **graph_connectivity**: fraction of cells whose same-label
          neighbors are in the largest connected component per label.
    """
    metrics: dict[str, float] = {}

    # Choose embedding
    if "X_scVI" in adata.obsm:
        use_rep = "X_scVI"
    elif "X_pca_harmony" in adata.obsm:
        use_rep = "X_pca_harmony"
    elif "X_scanorama" in adata.obsm:
        use_rep = "X_scanorama"
    elif "X_pca" in adata.obsm:
        use_rep = "X_pca"
    else:
        logger.warning("No embedding found; skipping metric computation.")
        return metrics

    X_emb = adata.obsm[use_rep]
    logger.info("Computing batch metrics using %s", use_rep)

    # -- Silhouette scores ------------------------------------------------
    try:
        from sklearn.metrics import silhouette_score

        metrics["asw_batch"] = float(
            silhouette_score(X_emb, adata.obs[batch_key], sample_size=min(5000, adata.n_obs))
        )
        metrics["asw_bio"] = float(
            silhouette_score(X_emb, adata.obs[label_key], sample_size=min(5000, adata.n_obs))
        )
        logger.info("ASW batch=%.4f, bio=%.4f", metrics["asw_batch"], metrics["asw_bio"])
    except Exception as exc:
        logger.warning("Silhouette computation failed: %s", exc)

    # -- LISI (integration & cell-type) -----------------------------------
    try:
        from scib.metrics import ilisi_graph, clisi_graph  # type: ignore[import-untyped]

        metrics["ilisi"] = float(
            ilisi_graph(adata, batch_key=batch_key, type_="embed", use_rep=use_rep)
        )
        metrics["clisi"] = float(
            clisi_graph(adata, label_key=label_key, type_="embed", use_rep=use_rep)
        )
        logger.info("iLISI=%.4f, cLISI=%.4f", metrics["ilisi"], metrics["clisi"])
    except ImportError:
        logger.info("scib not installed; computing LISI manually.")
        metrics.update(_lisi_manual(adata, batch_key, label_key))
    except Exception as exc:
        logger.warning("LISI computation failed: %s", exc)

    # -- kBET acceptance rate ---------------------------------------------
    try:
        metrics["kbet_acceptance_rate"] = _kbet(adata, batch_key, use_rep)
        logger.info("kBET acceptance rate=%.4f", metrics["kbet_acceptance_rate"])
    except Exception as exc:
        logger.warning("kBET computation failed: %s", exc)

    # -- Graph connectivity -----------------------------------------------
    try:
        metrics["graph_connectivity"] = _graph_connectivity(adata, label_key)
        logger.info("Graph connectivity=%.4f", metrics["graph_connectivity"])
    except Exception as exc:
        logger.warning("Graph connectivity computation failed: %s", exc)

    return metrics


def _lisi_manual(
    adata: AnnData, batch_key: str, label_key: str
) -> dict[str, float]:
    """Lightweight LISI approximation when scib is not available.

    Uses the k-nearest neighbor graph already stored in ``adata.obsp``.
    """
    from scipy.sparse import csr_matrix

    results: dict[str, float] = {}
    connectivities = adata.obsp.get("connectivities")
    if connectivities is None:
        return results

    if not isinstance(connectivities, csr_matrix):
        connectivities = csr_matrix(connectivities)

    for key, metric_name in [(batch_key, "ilisi"), (label_key, "clisi")]:
        labels = adata.obs[key].values
        categories = pd.Categorical(labels).codes
        n_cats = len(set(categories))
        scores = []
        for i in range(adata.n_obs):
            row = connectivities[i]
            neigh_idx = row.indices
            if len(neigh_idx) == 0:
                continue
            neigh_labels = categories[neigh_idx]
            freq = np.bincount(neigh_labels, minlength=n_cats).astype(np.float64)
            freq /= freq.sum()
            # Inverse Simpson index
            simpson = np.sum(freq**2)
            scores.append(1.0 / simpson if simpson > 0 else 1.0)
        results[metric_name] = float(np.median(scores)) if scores else 0.0

    return results


def _kbet(
    adata: AnnData,
    batch_key: str,
    use_rep: str,
    k: int = 25,
    n_samples: int = 2000,
) -> float:
    """Approximate kBET acceptance rate using chi-squared test on local
    batch composition vs global batch frequencies.
    """
    from scipy.stats import chi2

    X_emb = adata.obsm[use_rep]
    batches = pd.Categorical(adata.obs[batch_key])
    codes = batches.codes
    n_batches = len(batches.categories)
    global_freq = np.bincount(codes, minlength=n_batches).astype(np.float64)
    global_freq /= global_freq.sum()

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(adata.n_obs, size=min(n_samples, adata.n_obs), replace=False)

    # Use pre-computed neighbor graph if available
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X_emb)
    _, indices = nn.kneighbors(X_emb[sample_idx])

    dof = n_batches - 1
    alpha = 0.05
    accepted = 0
    for neigh_idx in indices:
        observed = np.bincount(codes[neigh_idx], minlength=n_batches).astype(np.float64)
        expected = global_freq * k
        expected[expected == 0] = 1e-10
        chi2_stat = float(np.sum((observed - expected) ** 2 / expected))
        p_value = 1.0 - chi2.cdf(chi2_stat, dof)
        if p_value >= alpha:
            accepted += 1

    return accepted / len(sample_idx)


def _graph_connectivity(adata: AnnData, label_key: str) -> float:
    """Fraction of cells whose same-label subgraph is in the largest
    connected component (per label), averaged over all labels.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    connectivities = adata.obsp.get("connectivities")
    if connectivities is None:
        return 0.0
    if not isinstance(connectivities, csr_matrix):
        connectivities = csr_matrix(connectivities)

    labels = adata.obs[label_key].values
    unique_labels = pd.unique(labels)
    scores = []
    for lab in unique_labels:
        mask = labels == lab
        idx = np.where(mask)[0]
        if len(idx) < 2:
            scores.append(1.0)
            continue
        sub = connectivities[np.ix_(idx, idx)]
        n_comp, comp_labels = connected_components(sub, directed=False)
        if n_comp == 1:
            scores.append(1.0)
        else:
            largest = np.bincount(comp_labels).max()
            scores.append(largest / len(idx))

    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------


def harmonize_datasets(
    adatas: list[AnnData],
    config: dict[str, Any] | None = None,
    method: Literal["scvi", "harmony", "scanorama"] = "scvi",
) -> AnnData:
    """Concatenate and harmonize multiple scRNA-seq datasets.

    Parameters
    ----------
    adatas:
        List of preprocessed AnnData objects.  Each should have
        normalized, log-transformed expression in ``.X`` and batch /
        cell type annotations in ``.obs``.
    config:
        Integration configuration parameters.  Missing keys fall back
        to :data:`DEFAULT_CONFIG`.
    method:
        Integration method to use.

    Returns
    -------
    AnnData
        Concatenated and batch-corrected AnnData with an integrated
        embedding and neighbor graph.
    """
    cfg = _resolve_config(config)

    if method not in _METHODS:
        raise ValueError(
            f"Unknown integration method {method!r}. "
            f"Choose from {list(_METHODS)}"
        )

    batch_key = cfg["batch_key"]
    label_key = cfg["label_key"]

    logger.info(
        "Harmonizing %d datasets with method=%s (batch_key=%s)",
        len(adatas),
        method,
        batch_key,
    )

    # Tag each AnnData with its dataset label if not already present
    for i, ad in enumerate(adatas):
        if batch_key not in ad.obs.columns:
            ad.obs[batch_key] = f"dataset_{i}"
            logger.info("Assigned batch label 'dataset_%d' to AnnData %d", i, i)

    # Find shared genes
    common_genes = set(adatas[0].var_names)
    for ad in adatas[1:]:
        common_genes &= set(ad.var_names)
    common_genes_sorted = sorted(common_genes)
    logger.info("Using %d shared genes across all datasets", len(common_genes_sorted))

    adatas_sub = [ad[:, common_genes_sorted].copy() for ad in adatas]

    # Concatenate
    import anndata as ad_module

    adata = ad_module.concat(adatas_sub, join="inner", merge="same")
    logger.info("Concatenated: %d cells x %d genes", adata.n_obs, adata.n_vars)

    # Run integration
    adata = _METHODS[method](adata, cfg)

    return adata
