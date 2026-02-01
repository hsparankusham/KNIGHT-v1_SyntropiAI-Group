#!/usr/bin/env python3
"""Step 6: Harmonize datasets across batches.

Integrates all processed scRNA-seq datasets using scVI or Harmony,
removing batch effects while preserving biological variation.

Usage:
    python scripts/data_curation/06_harmonize_batches.py [--method scvi]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from knight.data.harmonizer.batch_correction import harmonize_datasets, compute_batch_metrics
from knight.utils.io import load_config, save_json, ensure_dir
from knight.utils.logging import setup_logging

logger = setup_logging("step06_harmonize")


def main() -> None:
    parser = argparse.ArgumentParser(description="Harmonize batches")
    parser.add_argument("--method", default="scvi", choices=["scvi", "harmony", "scanorama"])
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("STEP 6: Harmonize Batches (method=%s)", args.method)
    logger.info("=" * 60)

    config = load_config()
    processed_dir = Path("data/processed/expression")
    out_dir = ensure_dir(Path("data/processed/integrated"))

    # Load all processed datasets
    h5ad_files = sorted(processed_dir.glob("*_processed.h5ad"))
    logger.info("Loading %d processed datasets", len(h5ad_files))

    adatas = {}
    for f in h5ad_files:
        name = f.stem.replace("_processed", "")
        adata = sc.read_h5ad(f)
        adata.obs["dataset"] = name
        adatas[name] = adata
        logger.info("  %s: %d cells", name, adata.n_obs)

    # Harmonize
    integrated = harmonize_datasets(
        adatas=adatas,
        config=config.get("data", {}).get("preprocessing", {}),
        method=args.method,
    )

    # Compute batch metrics
    metrics = compute_batch_metrics(
        integrated,
        batch_key="dataset",
        label_key="cell_type",
    )
    logger.info("Batch metrics: %s", metrics)

    # Save
    integrated.write_h5ad(out_dir / f"integrated_{args.method}.h5ad")
    save_json(metrics, out_dir / f"batch_metrics_{args.method}.json")

    logger.info("=" * 60)
    logger.info("STEP 6 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
