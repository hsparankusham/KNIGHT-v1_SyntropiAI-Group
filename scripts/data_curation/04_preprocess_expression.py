#!/usr/bin/env python3
"""Step 4: Preprocess scRNA-seq expression data.

Runs QC filtering, normalization, HVG selection, and dimensionality
reduction on all downloaded datasets.

Usage:
    python scripts/data_curation/04_preprocess_expression.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from knight.data.preprocessors.expression import preprocess_expression, compute_qc_report
from knight.utils.io import load_config, save_json, ensure_dir
from knight.utils.logging import setup_logging

logger = setup_logging("step04_preprocess")


def main() -> None:
    logger.info("=" * 60)
    logger.info("STEP 4: Preprocess Expression Data")
    logger.info("=" * 60)

    config = load_config()
    raw_dir = Path(config.get("data", {}).get("paths", {}).get("raw_dir", "data/raw"))
    out_dir = ensure_dir(Path("data/processed/expression"))

    # Process each h5ad file in raw directories
    h5ad_files = list(raw_dir.rglob("*.h5ad"))
    logger.info("Found %d h5ad files to process", len(h5ad_files))

    for h5ad_path in h5ad_files:
        dataset_name = h5ad_path.stem
        output_path = out_dir / f"{dataset_name}_processed.h5ad"

        if output_path.exists():
            logger.info("Skipping %s (already processed)", dataset_name)
            continue

        logger.info("Processing: %s", dataset_name)
        adata = sc.read_h5ad(h5ad_path)
        logger.info("  Raw: %d cells x %d genes", adata.n_obs, adata.n_vars)

        # QC report before filtering
        qc_pre = compute_qc_report(adata)

        # Preprocess
        adata = preprocess_expression(adata, config.get("data", {}).get("preprocessing", {}))
        logger.info("  Processed: %d cells x %d genes", adata.n_obs, adata.n_vars)

        # QC report after filtering
        qc_post = compute_qc_report(adata)

        # Save
        adata.write_h5ad(output_path)
        save_json(
            {"pre_filter": qc_pre, "post_filter": qc_post},
            out_dir / f"{dataset_name}_qc_report.json",
        )
        logger.info("  Saved to %s", output_path)

    logger.info("=" * 60)
    logger.info("STEP 4 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
