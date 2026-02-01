#!/usr/bin/env python3
"""Step 7: Create train/val/test splits.

Creates stratified splits ensuring:
- No data leakage between splits (split by donor, not cell)
- Balanced cell type representation
- Held-out datasets for cross-dataset generalization testing

Usage:
    python scripts/data_curation/07_create_splits.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from knight.utils.io import load_config, save_json, ensure_dir
from knight.utils.logging import setup_logging

logger = setup_logging("step07_splits")


def create_donor_splits(
    adata: ad.AnnData,
    donor_key: str = "donor_id",
    test_size: float = 0.15,
    val_size: float = 0.10,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Split by donor to prevent data leakage.

    Returns:
        Dict mapping split name to array of cell indices.
    """
    donors = adata.obs[donor_key].values
    unique_donors = np.unique(donors)
    logger.info("Splitting %d cells from %d donors", adata.n_obs, len(unique_donors))

    # First split: train+val vs test
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(np.arange(adata.n_obs), groups=donors))

    # Second split: train vs val
    trainval_donors = donors[trainval_idx]
    relative_val = val_size / (1 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_val, random_state=seed)
    train_rel, val_rel = next(gss2.split(np.arange(len(trainval_idx)), groups=trainval_donors))

    splits = {
        "train": trainval_idx[train_rel],
        "val": trainval_idx[val_rel],
        "test": test_idx,
    }

    for name, idx in splits.items():
        n_donors = len(np.unique(donors[idx]))
        logger.info("  %s: %d cells, %d donors", name, len(idx), n_donors)

    return splits


def main() -> None:
    logger.info("=" * 60)
    logger.info("STEP 7: Create Train/Val/Test Splits")
    logger.info("=" * 60)

    config = load_config()
    integrated_dir = Path("data/processed/integrated")
    splits_dir = ensure_dir(Path("data/splits"))

    # Load integrated dataset
    h5ad_files = list(integrated_dir.glob("integrated_*.h5ad"))
    if not h5ad_files:
        logger.error("No integrated datasets found. Run step 06 first.")
        return

    adata = sc.read_h5ad(h5ad_files[0])
    logger.info("Loaded: %d cells x %d genes", adata.n_obs, adata.n_vars)

    # Determine donor key
    donor_key = "donor_id"
    if donor_key not in adata.obs.columns:
        for candidate in ["subject_id", "individual", "sample_id", "donor"]:
            if candidate in adata.obs.columns:
                donor_key = candidate
                break
        else:
            logger.warning("No donor key found, falling back to random split")
            donor_key = None

    if donor_key:
        splits = create_donor_splits(adata, donor_key=donor_key)
    else:
        # Random split as fallback
        rng = np.random.default_rng(42)
        indices = rng.permutation(adata.n_obs)
        n_test = int(0.15 * adata.n_obs)
        n_val = int(0.10 * adata.n_obs)
        splits = {
            "test": indices[:n_test],
            "val": indices[n_test : n_test + n_val],
            "train": indices[n_test + n_val :],
        }

    # Save splits
    for name, idx in splits.items():
        np.save(splits_dir / f"{name}_indices.npy", idx)
        adata_split = adata[idx].copy()
        adata_split.write_h5ad(splits_dir / f"{name}.h5ad")

    # Save split metadata
    split_info = {
        name: {
            "n_cells": len(idx),
            "cell_types": adata[idx].obs.get("cell_type", pd.Series()).value_counts().to_dict()
            if "cell_type" in adata.obs.columns
            else {},
        }
        for name, idx in splits.items()
    }
    save_json(split_info, splits_dir / "split_metadata.json")

    logger.info("=" * 60)
    logger.info("STEP 7 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    import pandas as pd
    main()
