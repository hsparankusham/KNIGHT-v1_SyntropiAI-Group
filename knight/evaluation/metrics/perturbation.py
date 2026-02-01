"""Perturbation prediction metrics for KNIGHT v1 evaluation.

Evaluates how accurately the model predicts gene-expression changes
(deltas) in response to genetic or pharmacological perturbations of
brain neuroimmune cells.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.stats import pearsonr


def compute_gene_pearson(
    predicted_delta: np.ndarray,
    actual_delta: np.ndarray,
) -> float:
    """Mean Pearson correlation across genes.

    For each gene, compute the Pearson correlation between the predicted
    and actual delta vectors across cells, then return the mean.

    Parameters
    ----------
    predicted_delta : np.ndarray of shape (n_cells, n_genes)
        Predicted expression change per cell and gene.
    actual_delta : np.ndarray of shape (n_cells, n_genes)
        Observed expression change per cell and gene.

    Returns
    -------
    float
        Mean gene-wise Pearson *r* (higher = better).
    """
    predicted_delta = np.asarray(predicted_delta, dtype=np.float64)
    actual_delta = np.asarray(actual_delta, dtype=np.float64)

    n_genes = predicted_delta.shape[1]
    correlations: list[float] = []

    for g in range(n_genes):
        pred_g = predicted_delta[:, g]
        true_g = actual_delta[:, g]
        # Skip constant vectors (no variation).
        if np.std(pred_g) < 1e-12 or np.std(true_g) < 1e-12:
            correlations.append(0.0)
            continue
        r, _ = pearsonr(pred_g, true_g)
        correlations.append(float(r) if np.isfinite(r) else 0.0)

    return float(np.mean(correlations))


def compute_cell_pearson(
    predicted_delta: np.ndarray,
    actual_delta: np.ndarray,
) -> float:
    """Mean Pearson correlation across cells.

    For each cell, compute the Pearson correlation between the predicted
    and actual delta vectors across genes, then return the mean.

    Parameters
    ----------
    predicted_delta : np.ndarray of shape (n_cells, n_genes)
        Predicted expression change.
    actual_delta : np.ndarray of shape (n_cells, n_genes)
        Observed expression change.

    Returns
    -------
    float
        Mean cell-wise Pearson *r* (higher = better).
    """
    predicted_delta = np.asarray(predicted_delta, dtype=np.float64)
    actual_delta = np.asarray(actual_delta, dtype=np.float64)

    n_cells = predicted_delta.shape[0]
    correlations: list[float] = []

    for c in range(n_cells):
        pred_c = predicted_delta[c, :]
        true_c = actual_delta[c, :]
        if np.std(pred_c) < 1e-12 or np.std(true_c) < 1e-12:
            correlations.append(0.0)
            continue
        r, _ = pearsonr(pred_c, true_c)
        correlations.append(float(r) if np.isfinite(r) else 0.0)

    return float(np.mean(correlations))


def compute_top_k_accuracy(
    predicted_delta: np.ndarray,
    actual_delta: np.ndarray,
    k: int = 20,
) -> float:
    """Fraction of true top-k DE genes recovered in the prediction.

    Ranks genes by the absolute mean delta across cells for both the
    predicted and actual arrays, then computes the overlap among the
    top *k* genes.

    Parameters
    ----------
    predicted_delta : np.ndarray of shape (n_cells, n_genes)
        Predicted expression change.
    actual_delta : np.ndarray of shape (n_cells, n_genes)
        Observed expression change.
    k : int, default 20
        Number of top differentially expressed genes to consider.

    Returns
    -------
    float
        Fraction of top-k genes correctly identified (0-1).
    """
    predicted_delta = np.asarray(predicted_delta, dtype=np.float64)
    actual_delta = np.asarray(actual_delta, dtype=np.float64)

    # Mean absolute delta per gene (across cells).
    pred_importance = np.abs(predicted_delta).mean(axis=0)
    true_importance = np.abs(actual_delta).mean(axis=0)

    k = min(k, predicted_delta.shape[1])

    pred_top_k = set(np.argsort(pred_importance)[-k:])
    true_top_k = set(np.argsort(true_importance)[-k:])

    overlap = len(pred_top_k & true_top_k)
    return float(overlap / k)


def compute_direction_accuracy(
    predicted_delta: np.ndarray,
    actual_delta: np.ndarray,
) -> float:
    """Fraction of genes with correct up/down direction.

    Compares the sign of the mean delta (across cells) for each gene.
    Genes with near-zero actual change (absolute mean < 1e-8) are
    excluded from the denominator.

    Parameters
    ----------
    predicted_delta : np.ndarray of shape (n_cells, n_genes)
        Predicted expression change.
    actual_delta : np.ndarray of shape (n_cells, n_genes)
        Observed expression change.

    Returns
    -------
    float
        Fraction of genes with matching sign (0-1).
    """
    predicted_delta = np.asarray(predicted_delta, dtype=np.float64)
    actual_delta = np.asarray(actual_delta, dtype=np.float64)

    pred_mean = predicted_delta.mean(axis=0)
    true_mean = actual_delta.mean(axis=0)

    # Exclude genes with negligible true change.
    active_mask = np.abs(true_mean) > 1e-8
    if active_mask.sum() == 0:
        return 1.0  # No meaningful change to evaluate.

    pred_sign = np.sign(pred_mean[active_mask])
    true_sign = np.sign(true_mean[active_mask])

    return float(np.mean(pred_sign == true_sign))


def perturbation_report(
    predicted_delta: np.ndarray,
    actual_delta: np.ndarray,
    gene_names: Optional[Sequence[str]] = None,
    top_k: int = 20,
) -> dict:
    """Generate a comprehensive perturbation prediction report.

    Parameters
    ----------
    predicted_delta : np.ndarray of shape (n_cells, n_genes)
        Predicted expression change.
    actual_delta : np.ndarray of shape (n_cells, n_genes)
        Observed expression change.
    gene_names : sequence of str or None
        Optional gene names; if provided, the top-k gene lists are
        included in the report.
    top_k : int, default 20
        Number of top DE genes for the top-k accuracy metric.

    Returns
    -------
    dict
        ``gene_pearson`` : float
        ``cell_pearson`` : float
        ``top_k_accuracy`` : float
        ``top_k`` : int
        ``direction_accuracy`` : float
        ``n_cells`` : int
        ``n_genes`` : int
        ``top_predicted_genes`` : list[str] or None
        ``top_actual_genes`` : list[str] or None
    """
    predicted_delta = np.asarray(predicted_delta, dtype=np.float64)
    actual_delta = np.asarray(actual_delta, dtype=np.float64)

    report: dict = {
        "gene_pearson": compute_gene_pearson(predicted_delta, actual_delta),
        "cell_pearson": compute_cell_pearson(predicted_delta, actual_delta),
        "top_k_accuracy": compute_top_k_accuracy(predicted_delta, actual_delta, k=top_k),
        "top_k": top_k,
        "direction_accuracy": compute_direction_accuracy(predicted_delta, actual_delta),
        "n_cells": predicted_delta.shape[0],
        "n_genes": predicted_delta.shape[1],
    }

    # Optionally include named top-k gene lists.
    if gene_names is not None:
        gene_names = list(gene_names)
        k = min(top_k, len(gene_names))
        pred_importance = np.abs(predicted_delta).mean(axis=0)
        true_importance = np.abs(actual_delta).mean(axis=0)
        report["top_predicted_genes"] = [
            gene_names[i] for i in np.argsort(pred_importance)[-k:][::-1]
        ]
        report["top_actual_genes"] = [
            gene_names[i] for i in np.argsort(true_importance)[-k:][::-1]
        ]
    else:
        report["top_predicted_genes"] = None
        report["top_actual_genes"] = None

    return report
