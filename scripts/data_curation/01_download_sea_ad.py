#!/usr/bin/env python3
"""Step 1: Download SEA-AD atlas data from the Allen Institute.

This script downloads the Seattle Alzheimer's Disease Brain Cell Atlas
scRNA-seq and scATAC-seq data for KNIGHT pretraining.

Usage:
    python scripts/data_curation/01_download_sea_ad.py [--force]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from knight.data.downloaders.sea_ad import download_sea_ad
from knight.utils.logging import setup_logging

logger = setup_logging("step01_download_sea_ad")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SEA-AD atlas data")
    parser.add_argument("--force", action="store_true", help="Re-download existing files")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/sea_ad"),
        help="Output directory",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("STEP 1: Download SEA-AD Atlas")
    logger.info("=" * 60)

    download_sea_ad(output_dir=args.output_dir, force=args.force)

    logger.info("=" * 60)
    logger.info("STEP 1 COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
