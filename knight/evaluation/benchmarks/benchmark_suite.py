"""Full benchmark runner for KNIGHT v1 evaluation.

Orchestrates cell-state classification, batch harmonization, and
perturbation prediction benchmarks against quantitative targets for the
brain neuroimmune cell state foundation model.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np
import scanpy as sc

from knight.evaluation.metrics.classification import (
    classification_report,
    compute_balanced_accuracy,
)
from knight.evaluation.metrics.integration import (
    compute_batch_effect_residual,
    integration_report,
)
from knight.evaluation.metrics.perturbation import (
    compute_gene_pearson,
    perturbation_report,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Minimal protocol the model must satisfy for benchmarking
# ---------------------------------------------------------------------------

@runtime_checkable
class KnightModelProtocol(Protocol):
    """Protocol that a KNIGHT model must implement for benchmarking."""

    def predict_cell_states(
        self, adata: sc.AnnData
    ) -> np.ndarray:
        """Return predicted cell-state labels for each cell."""
        ...

    def embed(self, adata: sc.AnnData) -> np.ndarray:
        """Return cell embeddings of shape (n_cells, n_dims)."""
        ...

    def predict_perturbation(
        self,
        control_adata: sc.AnnData,
        perturbation_info: dict[str, Any],
    ) -> np.ndarray:
        """Predict expression delta given a perturbation specification."""
        ...


# ---------------------------------------------------------------------------
# Default performance targets
# ---------------------------------------------------------------------------

DEFAULT_TARGETS: dict[str, float] = {
    "balanced_accuracy": 0.90,
    "batch_effect_residual": 0.10,  # upper bound
    "perturbation_gene_pearson": 0.60,
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark suite.

    Attributes
    ----------
    embed_key : str
        Key in ``adata.obsm`` for storing / reading the KNIGHT embedding.
    batch_key : str
        Column in ``adata.obs`` identifying the batch.
    label_key : str
        Column in ``adata.obs`` with cell-type labels.
    coarse_label_key : str or None
        Optional coarse-level cell-type column for hierarchical eval.
    class_names : list[str] or None
        Explicit ordering of class names.  Inferred from data if *None*.
    perturbation_top_k : int
        *k* for the top-k DE gene accuracy metric.
    targets : dict[str, float]
        Performance targets.
    """

    embed_key: str = "X_knight"
    batch_key: str = "batch"
    label_key: str = "cell_type"
    coarse_label_key: Optional[str] = None
    class_names: Optional[list[str]] = None
    perturbation_top_k: int = 20
    targets: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TARGETS))


# ---------------------------------------------------------------------------
# Benchmark Suite
# ---------------------------------------------------------------------------

class BenchmarkSuite:
    """Runs the full KNIGHT v1 evaluation benchmark suite.

    Parameters
    ----------
    model : KnightModelProtocol
        A model that implements ``predict_cell_states``, ``embed``, and
        ``predict_perturbation``.
    config : BenchmarkConfig or None
        Benchmark configuration.  Uses defaults if *None*.
    """

    def __init__(
        self,
        model: KnightModelProtocol,
        config: Optional[BenchmarkConfig] = None,
    ) -> None:
        self.model = model
        self.config = config or BenchmarkConfig()

    # -----------------------------------------------------------------
    # Individual benchmarks
    # -----------------------------------------------------------------

    def run_cellstate_benchmark(
        self,
        test_data: sc.AnnData,
    ) -> dict[str, Any]:
        """Run cell-state classification on held-out data.

        Parameters
        ----------
        test_data : sc.AnnData
            Held-out annotated dataset with ground-truth labels in
            ``adata.obs[config.label_key]``.

        Returns
        -------
        dict
            Classification report (balanced accuracy, per-class F1, etc.).
        """
        logger.info("Running cell-state classification benchmark ...")
        t0 = time.time()

        y_true = test_data.obs[self.config.label_key].values
        y_pred = self.model.predict_cell_states(test_data)

        class_names = self.config.class_names
        if class_names is None:
            class_names = sorted(set(y_true) | set(y_pred))

        report = classification_report(y_true, y_pred, class_names)
        report["elapsed_seconds"] = time.time() - t0
        logger.info(
            "Cell-state benchmark done. Balanced accuracy: %.4f",
            report["balanced_accuracy"],
        )
        return report

    def run_harmonization_benchmark(
        self,
        multi_dataset_adata: sc.AnnData,
    ) -> dict[str, Any]:
        """Embed multi-dataset data and compute integration metrics.

        Parameters
        ----------
        multi_dataset_adata : sc.AnnData
            Combined dataset with a batch column in ``adata.obs`` and
            cell-type labels.

        Returns
        -------
        dict
            Integration report (batch ASW, bio ASW, LISI, etc.).
        """
        logger.info("Running harmonization benchmark ...")
        t0 = time.time()

        # Compute and store the embedding.
        embedding = self.model.embed(multi_dataset_adata)
        multi_dataset_adata.obsm[self.config.embed_key] = embedding

        report = integration_report(
            multi_dataset_adata,
            batch_key=self.config.batch_key,
            label_key=self.config.label_key,
            embed_key=self.config.embed_key,
        )
        report["elapsed_seconds"] = time.time() - t0
        logger.info(
            "Harmonization benchmark done. Batch residual: %.4f",
            report["batch_effect_residual"],
        )
        return report

    def run_perturbation_benchmark(
        self,
        test_perturbations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Predict perturbation responses and score them.

        Parameters
        ----------
        test_perturbations : list of dict
            Each dict must contain:
            - ``"control_adata"`` : sc.AnnData of control cells.
            - ``"perturbation_info"`` : dict describing the perturbation.
            - ``"actual_delta"`` : np.ndarray of shape (n_cells, n_genes)
              with the ground-truth expression change.
            - ``"gene_names"`` : list[str] (optional).

        Returns
        -------
        dict
            Aggregated perturbation metrics across all test perturbations.
        """
        logger.info("Running perturbation benchmark ...")
        t0 = time.time()

        all_gene_pearson: list[float] = []
        all_cell_pearson: list[float] = []
        all_top_k: list[float] = []
        all_direction: list[float] = []
        per_perturbation: list[dict] = []

        for idx, item in enumerate(test_perturbations):
            control = item["control_adata"]
            pert_info = item["perturbation_info"]
            actual_delta = item["actual_delta"]
            gene_names = item.get("gene_names")

            predicted_delta = self.model.predict_perturbation(control, pert_info)

            report_i = perturbation_report(
                predicted_delta,
                actual_delta,
                gene_names=gene_names,
                top_k=self.config.perturbation_top_k,
            )
            report_i["perturbation_index"] = idx
            per_perturbation.append(report_i)

            all_gene_pearson.append(report_i["gene_pearson"])
            all_cell_pearson.append(report_i["cell_pearson"])
            all_top_k.append(report_i["top_k_accuracy"])
            all_direction.append(report_i["direction_accuracy"])

        aggregate: dict[str, Any] = {
            "mean_gene_pearson": float(np.mean(all_gene_pearson)) if all_gene_pearson else 0.0,
            "mean_cell_pearson": float(np.mean(all_cell_pearson)) if all_cell_pearson else 0.0,
            "mean_top_k_accuracy": float(np.mean(all_top_k)) if all_top_k else 0.0,
            "mean_direction_accuracy": float(np.mean(all_direction)) if all_direction else 0.0,
            "n_perturbations": len(test_perturbations),
            "per_perturbation": per_perturbation,
            "elapsed_seconds": time.time() - t0,
        }
        logger.info(
            "Perturbation benchmark done. Mean gene Pearson: %.4f",
            aggregate["mean_gene_pearson"],
        )
        return aggregate

    # -----------------------------------------------------------------
    # Full suite
    # -----------------------------------------------------------------

    def run_all(
        self,
        test_data: sc.AnnData,
        multi_dataset_adata: sc.AnnData,
        test_perturbations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Run the complete benchmark suite.

        Parameters
        ----------
        test_data : sc.AnnData
            Held-out classification data.
        multi_dataset_adata : sc.AnnData
            Multi-batch integration data.
        test_perturbations : list of dict
            Perturbation test cases.

        Returns
        -------
        dict
            Comprehensive report with keys ``cellstate``,
            ``harmonization``, ``perturbation``, and ``passes``.
        """
        logger.info("=" * 60)
        logger.info("KNIGHT v1 Full Benchmark Suite")
        logger.info("=" * 60)
        t0 = time.time()

        results: dict[str, Any] = {
            "cellstate": self.run_cellstate_benchmark(test_data),
            "harmonization": self.run_harmonization_benchmark(multi_dataset_adata),
            "perturbation": self.run_perturbation_benchmark(test_perturbations),
        }

        results["passes"] = self.passes_targets(results)
        results["total_elapsed_seconds"] = time.time() - t0

        all_pass = all(results["passes"].values())
        logger.info(
            "All benchmarks complete. Overall PASS: %s  (%.1f s)",
            all_pass,
            results["total_elapsed_seconds"],
        )
        return results

    # -----------------------------------------------------------------
    # Target checking
    # -----------------------------------------------------------------

    def passes_targets(self, results: dict[str, Any]) -> dict[str, bool]:
        """Check each metric against the configured performance targets.

        Parameters
        ----------
        results : dict
            Output of :meth:`run_all` (or a manually assembled dict with
            the same structure).

        Returns
        -------
        dict[str, bool]
            ``balanced_accuracy_pass`` : True if >= target.
            ``batch_residual_pass`` : True if <= target.
            ``perturbation_pearson_pass`` : True if >= target.
        """
        targets = self.config.targets
        passes: dict[str, bool] = {}

        # Cell-state classification
        bal_acc = results.get("cellstate", {}).get("balanced_accuracy", 0.0)
        passes["balanced_accuracy_pass"] = bal_acc >= targets.get("balanced_accuracy", 0.90)

        # Batch harmonization
        batch_res = results.get("harmonization", {}).get("batch_effect_residual", 1.0)
        passes["batch_residual_pass"] = batch_res <= targets.get("batch_effect_residual", 0.10)

        # Perturbation prediction
        pert_r = results.get("perturbation", {}).get("mean_gene_pearson", 0.0)
        passes["perturbation_pearson_pass"] = pert_r >= targets.get(
            "perturbation_gene_pearson", 0.60
        )

        return passes


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_benchmark_report(results: dict[str, Any]) -> str:
    """Format a benchmark results dict into a human-readable string.

    Parameters
    ----------
    results : dict
        Output of :meth:`BenchmarkSuite.run_all`.

    Returns
    -------
    str
        Multi-line formatted report suitable for logging.
    """
    lines: list[str] = []
    sep = "=" * 60

    lines.append(sep)
    lines.append("KNIGHT v1 -- Benchmark Report")
    lines.append(sep)

    # --- Cell-state classification ---
    cs = results.get("cellstate", {})
    lines.append("")
    lines.append("## Cell-State Classification")
    lines.append(f"  Balanced Accuracy : {cs.get('balanced_accuracy', 'N/A'):.4f}")
    lines.append(f"  Macro F1          : {cs.get('macro_f1', 'N/A'):.4f}")
    lines.append(f"  Samples / Classes : {cs.get('n_samples', '?')} / {cs.get('n_classes', '?')}")
    per_f1 = cs.get("per_class_f1", {})
    if per_f1:
        lines.append("  Per-class F1:")
        for name, f1 in per_f1.items():
            lines.append(f"    {name:30s} : {f1:.4f}")

    # --- Harmonization ---
    hm = results.get("harmonization", {})
    lines.append("")
    lines.append("## Batch Harmonization")
    for key in ("batch_asw", "bio_asw", "ilisi", "clisi", "graph_connectivity", "batch_effect_residual"):
        val = hm.get(key, "N/A")
        label = key.replace("_", " ").title()
        if isinstance(val, float):
            lines.append(f"  {label:30s} : {val:.4f}")
        else:
            lines.append(f"  {label:30s} : {val}")

    # --- Perturbation ---
    pt = results.get("perturbation", {})
    lines.append("")
    lines.append("## Perturbation Prediction")
    lines.append(f"  Mean Gene Pearson      : {pt.get('mean_gene_pearson', 'N/A'):.4f}")
    lines.append(f"  Mean Cell Pearson      : {pt.get('mean_cell_pearson', 'N/A'):.4f}")
    lines.append(f"  Mean Top-k Accuracy    : {pt.get('mean_top_k_accuracy', 'N/A'):.4f}")
    lines.append(f"  Mean Direction Accuracy: {pt.get('mean_direction_accuracy', 'N/A'):.4f}")
    lines.append(f"  # Perturbations        : {pt.get('n_perturbations', '?')}")

    # --- Pass / Fail ---
    passes = results.get("passes", {})
    lines.append("")
    lines.append("## Target Pass / Fail")
    for check, passed in passes.items():
        status = "PASS" if passed else "FAIL"
        lines.append(f"  {check:35s} : {status}")

    all_pass = all(passes.values()) if passes else False
    lines.append("")
    lines.append(f"  Overall: {'ALL TARGETS MET' if all_pass else 'TARGETS NOT MET'}")

    # Elapsed time
    elapsed = results.get("total_elapsed_seconds")
    if elapsed is not None:
        lines.append(f"  Total time: {elapsed:.1f} s")

    lines.append(sep)
    return "\n".join(lines)
