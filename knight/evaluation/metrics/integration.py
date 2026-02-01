"""Batch integration and harmonization metrics for KNIGHT v1 evaluation.

Measures how well the model embedding removes technical batch effects
while preserving biological variation across brain neuroimmune cell
datasets from different studies, protocols, and sequencing platforms.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import silhouette_samples
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------------
# Silhouette-based metrics
# ---------------------------------------------------------------------------

def compute_batch_asw(
    adata: sc.AnnData,
    batch_key: str,
    embed_key: str = "X_knight",
) -> float:
    """Compute average silhouette width for batch mixing.

    A value closer to **0** indicates better batch integration (batches
    are well-mixed in the embedding space).  The raw per-sample
    silhouette scores are rescaled from [-1, 1] to [0, 1] via
    ``1 - abs(score)``, so the returned value is in [0, 1] with 1 being
    optimal.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix with embedding stored in ``adata.obsm``.
    batch_key : str
        Column in ``adata.obs`` identifying the batch.
    embed_key : str, default ``"X_knight"``
        Key in ``adata.obsm`` for the embedding.

    Returns
    -------
    float
        Batch ASW (0-1, higher = better mixing).
    """
    embedding = adata.obsm[embed_key]
    labels = adata.obs[batch_key].values
    sil = silhouette_samples(embedding, labels)
    # Rescale: perfect mixing -> silhouette ~ 0 -> 1 - |0| = 1
    return float(np.mean(1 - np.abs(sil)))


def compute_bio_asw(
    adata: sc.AnnData,
    label_key: str,
    embed_key: str = "X_knight",
) -> float:
    """Compute average silhouette width for biological conservation.

    A **higher** value indicates that cell types remain well-separated
    after integration.  Raw silhouette scores (range [-1, 1]) are
    rescaled to [0, 1] via ``(score + 1) / 2``.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix.
    label_key : str
        Column in ``adata.obs`` with cell-type labels.
    embed_key : str, default ``"X_knight"``
        Key in ``adata.obsm``.

    Returns
    -------
    float
        Bio ASW (0-1, higher = better separation).
    """
    embedding = adata.obsm[embed_key]
    labels = adata.obs[label_key].values
    sil = silhouette_samples(embedding, labels)
    return float(np.mean((sil + 1) / 2))


# ---------------------------------------------------------------------------
# LISI-based metrics
# ---------------------------------------------------------------------------

def _compute_lisi(
    X: np.ndarray,
    labels: np.ndarray,
    perplexity: float = 30.0,
    n_neighbors: Optional[int] = None,
) -> np.ndarray:
    """Compute per-cell LISI (Local Inverse Simpson Index).

    This is a simplified implementation that uses a fixed k-nearest
    neighbour graph and estimates the effective number of label categories
    in each neighbourhood.

    Parameters
    ----------
    X : np.ndarray of shape (n_cells, n_dims)
        Embedding matrix.
    labels : np.ndarray of shape (n_cells,)
        Categorical labels (batch or cell type).
    perplexity : float
        Target perplexity (controls neighbourhood size).
    n_neighbors : int or None
        Number of neighbours.  Defaults to ``3 * perplexity``.

    Returns
    -------
    np.ndarray of shape (n_cells,)
        Per-cell LISI values.
    """
    if n_neighbors is None:
        n_neighbors = min(int(3 * perplexity), X.shape[0] - 1)
    n_neighbors = max(1, min(n_neighbors, X.shape[0] - 1))

    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto")
    nn.fit(X)
    indices = nn.kneighbors(X, return_distance=False)

    unique_labels, label_ints = np.unique(labels, return_inverse=True)
    n_labels = len(unique_labels)

    lisi = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        neighbour_labels = label_ints[indices[i]]
        # Proportion of each label in the neighbourhood
        counts = np.bincount(neighbour_labels, minlength=n_labels).astype(np.float64)
        props = counts / counts.sum()
        # Inverse Simpson index: 1 / sum(p^2)
        simpson = np.sum(props ** 2)
        lisi[i] = 1.0 / simpson if simpson > 0 else 1.0

    return lisi


def compute_ilisi(
    adata: sc.AnnData,
    batch_key: str,
    embed_key: str = "X_knight",
    perplexity: float = 30.0,
) -> float:
    """Compute integration LISI (iLISI).

    Higher values indicate better batch mixing (ideal = number of batches).
    The returned value is normalised to [0, 1] where 1 is optimal.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix.
    batch_key : str
        Batch column in ``adata.obs``.
    embed_key : str
        Embedding key in ``adata.obsm``.
    perplexity : float
        Perplexity for LISI computation.

    Returns
    -------
    float
        Normalised mean iLISI (0-1, higher = better).
    """
    embedding = adata.obsm[embed_key]
    labels = adata.obs[batch_key].values
    n_batches = len(np.unique(labels))

    lisi = _compute_lisi(embedding, labels, perplexity=perplexity)
    mean_lisi = float(np.mean(lisi))

    # Normalise to [0, 1]: (mean - 1) / (n_batches - 1)
    if n_batches <= 1:
        return 1.0
    return float((mean_lisi - 1) / (n_batches - 1))


def compute_clisi(
    adata: sc.AnnData,
    label_key: str,
    embed_key: str = "X_knight",
    perplexity: float = 30.0,
) -> float:
    """Compute cell-type LISI (cLISI).

    Lower cLISI means cell-type neighbourhoods are pure.  The returned
    value is normalised to [0, 1] where **1** indicates optimal purity
    (cLISI close to 1).

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix.
    label_key : str
        Cell-type column in ``adata.obs``.
    embed_key : str
        Embedding key in ``adata.obsm``.
    perplexity : float
        Perplexity for LISI computation.

    Returns
    -------
    float
        Normalised mean cLISI (0-1, higher = better purity).
    """
    embedding = adata.obsm[embed_key]
    labels = adata.obs[label_key].values
    n_types = len(np.unique(labels))

    lisi = _compute_lisi(embedding, labels, perplexity=perplexity)
    mean_lisi = float(np.mean(lisi))

    # Ideal cLISI = 1 (pure neighbourhoods); normalise so 1 is best.
    if n_types <= 1:
        return 1.0
    return float(1 - (mean_lisi - 1) / (n_types - 1))


# ---------------------------------------------------------------------------
# Graph connectivity
# ---------------------------------------------------------------------------

def compute_graph_connectivity(
    adata: sc.AnnData,
    label_key: str,
    embed_key: str = "X_knight",
    n_neighbors: int = 15,
) -> float:
    """Compute graph connectivity score.

    For each cell type, build a kNN graph on the embedding and check
    whether all cells of that type fall into a single connected component.
    The score is the mean fraction of cells in the largest connected
    component per cell type.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix.
    label_key : str
        Cell-type column in ``adata.obs``.
    embed_key : str
        Embedding key in ``adata.obsm``.
    n_neighbors : int
        Number of neighbours for the kNN graph.

    Returns
    -------
    float
        Graph connectivity score in [0, 1] (1 = fully connected per type).
    """
    embedding = adata.obsm[embed_key]
    labels = adata.obs[label_key].values

    nn = NearestNeighbors(n_neighbors=min(n_neighbors, embedding.shape[0] - 1))
    nn.fit(embedding)
    knn_graph = nn.kneighbors_graph(embedding, mode="connectivity")

    scores: list[float] = []
    for label in np.unique(labels):
        mask = labels == label
        idx = np.where(mask)[0]
        if len(idx) <= 1:
            scores.append(1.0)
            continue
        sub_graph = knn_graph[idx][:, idx]
        n_comp, comp_labels = connected_components(sub_graph, directed=False)
        largest_cc = np.bincount(comp_labels).max()
        scores.append(float(largest_cc / len(idx)))

    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Batch effect residual (variance explained)
# ---------------------------------------------------------------------------

def compute_batch_effect_residual(
    adata: sc.AnnData,
    batch_key: str,
    embed_key: str = "X_knight",
) -> float:
    """Compute fraction of embedding variance explained by batch.

    Uses a one-way ANOVA-style decomposition: for each embedding
    dimension, compute R-squared as ``SS_between / SS_total``, then
    average across dimensions.  Lower values mean less residual batch
    effect.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix.
    batch_key : str
        Batch column in ``adata.obs``.
    embed_key : str
        Embedding key in ``adata.obsm``.

    Returns
    -------
    float
        Mean fraction of variance explained by batch (0-1, lower = better).
    """
    embedding = adata.obsm[embed_key]
    batches = adata.obs[batch_key].values
    unique_batches = np.unique(batches)

    n_dims = embedding.shape[1]
    r_squared = np.zeros(n_dims)

    global_mean = embedding.mean(axis=0)

    for d in range(n_dims):
        col = embedding[:, d]
        ss_total = np.sum((col - global_mean[d]) ** 2)
        if ss_total == 0:
            r_squared[d] = 0.0
            continue

        ss_between = 0.0
        for batch in unique_batches:
            mask = batches == batch
            n_b = mask.sum()
            batch_mean = col[mask].mean()
            ss_between += n_b * (batch_mean - global_mean[d]) ** 2

        r_squared[d] = ss_between / ss_total

    return float(np.mean(r_squared))


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------

def integration_report(
    adata: sc.AnnData,
    batch_key: str,
    label_key: str,
    embed_key: str = "X_knight",
) -> dict[str, float]:
    """Run all integration metrics and return a summary dictionary.

    Parameters
    ----------
    adata : sc.AnnData
        Annotated data matrix.
    batch_key : str
        Batch column in ``adata.obs``.
    label_key : str
        Cell-type column in ``adata.obs``.
    embed_key : str
        Embedding key in ``adata.obsm``.

    Returns
    -------
    dict[str, float]
        Keys: ``batch_asw``, ``bio_asw``, ``ilisi``, ``clisi``,
        ``graph_connectivity``, ``batch_effect_residual``.
    """
    return {
        "batch_asw": compute_batch_asw(adata, batch_key, embed_key),
        "bio_asw": compute_bio_asw(adata, label_key, embed_key),
        "ilisi": compute_ilisi(adata, batch_key, embed_key),
        "clisi": compute_clisi(adata, label_key, embed_key),
        "graph_connectivity": compute_graph_connectivity(adata, label_key, embed_key),
        "batch_effect_residual": compute_batch_effect_residual(adata, batch_key, embed_key),
    }
