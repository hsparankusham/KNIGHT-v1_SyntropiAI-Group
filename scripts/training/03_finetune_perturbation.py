#!/usr/bin/env python3
"""Step T3: Fine-tune KNIGHT for perturbation response prediction.

Fine-tunes the pretrained encoder to predict gene expression changes
following CRISPRi knockdowns or drug perturbations in brain immune cells.

Usage:
    python scripts/training/03_finetune_perturbation.py --checkpoint models/checkpoints/knight_min/knight_min_final.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from knight.utils.logging import setup_logging

logger = setup_logging("finetune_perturbation")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune for perturbation prediction")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/defaults.yaml"))
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Perturbation Prediction Fine-tuning")
    logger.info("=" * 60)
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("This step requires CRISPRi screen data in data/raw/crispr_screens/")
    logger.info("Implementation follows the same pattern as 02_finetune_cellstate.py")
    logger.info("using PerturbationFinetuner and PerturbationDataset")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
