"""Cell state classification metrics for KNIGHT v1 evaluation.

Provides balanced accuracy, per-class F1, confusion matrices, and
hierarchical accuracy for coarse/fine cell-type taxonomies used in
brain neuroimmune cell state annotation.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix as _sklearn_cm,
    f1_score,
)


def compute_balanced_accuracy(
    y_true: np.ndarray | Sequence,
    y_pred: np.ndarray | Sequence,
) -> float:
    """Compute balanced accuracy (macro-averaged recall across classes).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth cell-state labels.
    y_pred : array-like of shape (n_samples,)
        Predicted cell-state labels.

    Returns
    -------
    float
        Balanced accuracy in [0, 1].
    """
    return float(balanced_accuracy_score(y_true, y_pred))


def compute_per_class_f1(
    y_true: np.ndarray | Sequence,
    y_pred: np.ndarray | Sequence,
    class_names: Sequence[str],
) -> dict[str, float]:
    """Compute per-class F1 scores.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    class_names : sequence of str
        Ordered class names matching the label encoding.

    Returns
    -------
    dict[str, float]
        Mapping from class name to its F1 score.
    """
    scores = f1_score(y_true, y_pred, labels=class_names, average=None, zero_division=0)
    return {name: float(s) for name, s in zip(class_names, scores)}


def compute_confusion_matrix(
    y_true: np.ndarray | Sequence,
    y_pred: np.ndarray | Sequence,
    class_names: Sequence[str],
) -> pd.DataFrame:
    """Compute a labelled confusion matrix.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    class_names : sequence of str
        Ordered class names used as both index and columns.

    Returns
    -------
    pd.DataFrame
        Confusion matrix with rows = true, columns = predicted.
    """
    cm = _sklearn_cm(y_true, y_pred, labels=class_names)
    return pd.DataFrame(cm, index=class_names, columns=class_names)


def compute_hierarchical_accuracy(
    y_true_coarse: np.ndarray | Sequence,
    y_pred_coarse: np.ndarray | Sequence,
    y_true_fine: np.ndarray | Sequence,
    y_pred_fine: np.ndarray | Sequence,
) -> dict[str, float]:
    """Compute hierarchical classification accuracy.

    Evaluates a two-level taxonomy (e.g. coarse = microglia vs astrocyte,
    fine = homeostatic microglia vs DAM vs reactive astrocyte).

    Parameters
    ----------
    y_true_coarse : array-like of shape (n_samples,)
        Ground-truth coarse-level labels.
    y_pred_coarse : array-like of shape (n_samples,)
        Predicted coarse-level labels.
    y_true_fine : array-like of shape (n_samples,)
        Ground-truth fine-level labels.
    y_pred_fine : array-like of shape (n_samples,)
        Predicted fine-level labels.

    Returns
    -------
    dict[str, float]
        ``coarse_acc`` : balanced accuracy at the coarse level.
        ``fine_acc``   : balanced accuracy at the fine level.
        ``conditional_fine_acc`` : fine accuracy only among samples where
            the coarse prediction is correct.
    """
    y_true_coarse = np.asarray(y_true_coarse)
    y_pred_coarse = np.asarray(y_pred_coarse)
    y_true_fine = np.asarray(y_true_fine)
    y_pred_fine = np.asarray(y_pred_fine)

    coarse_acc = float(balanced_accuracy_score(y_true_coarse, y_pred_coarse))
    fine_acc = float(balanced_accuracy_score(y_true_fine, y_pred_fine))

    # Conditional fine accuracy: restrict to samples with correct coarse label.
    coarse_correct_mask = y_true_coarse == y_pred_coarse
    if coarse_correct_mask.sum() == 0:
        conditional_fine_acc = 0.0
    else:
        conditional_fine_acc = float(
            balanced_accuracy_score(
                y_true_fine[coarse_correct_mask],
                y_pred_fine[coarse_correct_mask],
            )
        )

    return {
        "coarse_acc": coarse_acc,
        "fine_acc": fine_acc,
        "conditional_fine_acc": conditional_fine_acc,
    }


def classification_report(
    y_true: np.ndarray | Sequence,
    y_pred: np.ndarray | Sequence,
    class_names: Sequence[str],
) -> dict:
    """Generate a comprehensive classification report.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    class_names : sequence of str
        Ordered class names.

    Returns
    -------
    dict
        ``balanced_accuracy`` : float
        ``macro_f1`` : float (macro-averaged F1)
        ``per_class_f1`` : dict[str, float]
        ``confusion_matrix`` : pd.DataFrame
        ``n_samples`` : int
        ``n_classes`` : int
    """
    per_class = compute_per_class_f1(y_true, y_pred, class_names)
    macro_f1 = float(np.mean(list(per_class.values()))) if per_class else 0.0

    return {
        "balanced_accuracy": compute_balanced_accuracy(y_true, y_pred),
        "macro_f1": macro_f1,
        "per_class_f1": per_class,
        "confusion_matrix": compute_confusion_matrix(y_true, y_pred, class_names),
        "n_samples": len(y_true),
        "n_classes": len(class_names),
    }
