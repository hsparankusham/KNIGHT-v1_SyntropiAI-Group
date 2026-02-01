#!/usr/bin/env python3
"""Step E1: Benchmark cell state classification.

Evaluates KNIGHT on held-out test set for cell state prediction.
Reports balanced accuracy, per-class F1, and confusion matrix.

Usage:
    python scripts/evaluation/01_benchmark_cellstate.py --checkpoint models/checkpoints/cellstate/cellstate_best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from knight.data.datasets.cell_dataset import CellStateDataset
from knight.evaluation.metrics.classification import classification_report
from knight.model.architectures.knight_model import build_knight_model, KnightConfig
from knight.utils.io import load_config, load_json, save_json, ensure_dir
from knight.utils.logging import setup_logging
from knight.utils.reproducibility import set_seed, get_device

logger = setup_logging("benchmark_cellstate")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/defaults.yaml"))
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Cell State Classification Benchmark")
    logger.info("=" * 60)

    config = load_config(args.config)
    set_seed(42)
    device = get_device()

    # Load test split
    test_adata = sc.read_h5ad(Path("data/splits/test.h5ad"))
    gene_vocab = load_json(Path("data/processed/gene_vocab.json"))

    label_key = "cell_type"
    all_labels = sorted(set(test_adata.obs[label_key].values))
    label_map = {l: i for i, l in enumerate(all_labels)}

    model_cfg = config.get("model", {}).get("knight_min", {})
    test_dataset = CellStateDataset(
        test_adata, gene_vocab, label_key, label_map,
        max_genes=model_cfg.get("n_genes", 4000),
    )
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

    # Build and load model
    model_config = KnightConfig(
        n_genes=len(gene_vocab),
        d_model=model_cfg.get("d_model", 256),
        n_layers=model_cfg.get("n_layers", 6),
        n_heads=model_cfg.get("n_heads", 4),
        n_cellstate_classes=len(label_map),
    )
    model = build_knight_model(model_config)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    model = model.to(device).eval()

    # Inference
    all_preds = []
    all_labels_arr = []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch, task="cellstate")
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.append(preds)
            all_labels_arr.append(batch["label"].cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels_arr)

    # Compute metrics
    class_names = list(label_map.keys())
    report = classification_report(y_true, y_pred, class_names)

    target = config.get("evaluation", {}).get("cellstate_balanced_accuracy_target", 0.90)
    passed = report["balanced_accuracy"] >= target

    logger.info("Balanced Accuracy: %.4f (target: %.2f) — %s",
                report["balanced_accuracy"], target, "PASS" if passed else "FAIL")

    # Save results
    results_dir = ensure_dir(Path("results/benchmarks"))
    save_json(report, results_dir / "cellstate_benchmark.json")

    logger.info("=" * 60)
    logger.info("BENCHMARK COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
