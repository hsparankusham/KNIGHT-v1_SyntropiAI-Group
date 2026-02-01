#!/usr/bin/env python3
"""Step 2: Download supplementary GEO single-cell datasets.

Downloads key AD/neuroimmunology scRNA-seq datasets from GEO
for cross-dataset validation and training diversity.

Usage:
    python scripts/data_curation/02_download_geo_datasets.py [--datasets GSE188236,GSE174367]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from knight.data.downloaders.geo import download_all_geo, GEO_DATASETS
from knight.utils.logging import setup_logging

logger = setup_logging("step02_download_geo")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GEO datasets")
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated list of GEO accessions (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/geo"),
        help="Output directory",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("STEP 2: Download GEO Datasets")
    logger.info("=" * 60)

    datasets = args.datasets.split(",") if args.datasets else None
    if datasets:
        logger.info("Downloading %d specified datasets: %s", len(datasets), datasets)
    else:
        logger.info("Downloading all %d configured datasets", len(GEO_DATASETS))

    download_all_geo(output_dir=args.output_dir, datasets=datasets)

    logger.info("=" * 60)
    logger.info("STEP 2 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
