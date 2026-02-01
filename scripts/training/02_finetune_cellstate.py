#!/usr/bin/env python3
"""Step T2: Fine-tune KNIGHT for cell state classification.

Fine-tunes the pretrained encoder with a hierarchical cell state
classification head: coarse type (7) → fine state (40).

Usage:
    python scripts/training/02_finetune_cellstate.py --checkpoint models/checkpoints/knight_min/knight_min_final.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import scanpy as sc
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from knight.data.datasets.cell_dataset import CellStateDataset
from knight.model.architectures.knight_model import build_knight_model, KnightConfig
from knight.training.finetuner import CellStateFinetuner
from knight.utils.io import load_config, load_json, ensure_dir
from knight.utils.logging import setup_logging
from knight.utils.reproducibility import set_seed, get_device, count_parameters

logger = setup_logging("finetune_cellstate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune for cell state classification")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Pretrained checkpoint")
    parser.add_argument("--config", type=Path, default=Path("config/defaults.yaml"))
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Cell State Classification Fine-tuning")
    logger.info("=" * 60)

    config = load_config(args.config)
    set_seed(42)
    device = get_device()

    # Load splits
    splits_dir = Path("data/splits")
    train_adata = sc.read_h5ad(splits_dir / "train.h5ad")
    val_adata = sc.read_h5ad(splits_dir / "val.h5ad")

    # Load gene vocab
    gene_vocab = load_json(Path("data/processed/gene_vocab.json"))

    # Build label map from training data
    label_key = "cell_type"
    all_labels = sorted(set(train_adata.obs[label_key].values))
    label_map = {l: i for i, l in enumerate(all_labels)}
    logger.info("Cell states: %d classes", len(label_map))

    # Create datasets
    model_cfg = config.get("model", {}).get("knight_min", {})
    max_genes = model_cfg.get("n_genes", 4000)

    train_dataset = CellStateDataset(train_adata, gene_vocab, label_key, label_map, max_genes)
    val_dataset = CellStateDataset(val_adata, gene_vocab, label_key, label_map, max_genes)

    ft_cfg = config.get("training", {}).get("finetuning", {})
    batch_size = ft_cfg.get("batch_size", 64)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Build model with cellstate head
    model_config = KnightConfig(
        n_genes=len(gene_vocab),
        d_model=model_cfg.get("d_model", 256),
        n_layers=model_cfg.get("n_layers", 6),
        n_heads=model_cfg.get("n_heads", 4),
        dropout=model_cfg.get("dropout", 0.1),
        n_cellstate_classes=len(label_map),
    )
    model = build_knight_model(model_config)

    # Load pretrained weights
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info("Loaded checkpoint. Missing: %d, Unexpected: %d", len(missing), len(unexpected))

    model = model.to(device)
    logger.info("Parameters: %s", count_parameters(model))

    # Fine-tune
    checkpoint_dir = ensure_dir(Path("models/checkpoints/cellstate"))
    finetuner = CellStateFinetuner(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=ft_cfg,
        device=device,
    )
    finetuner.train(n_epochs=ft_cfg.get("epochs", 30))

    # Save
    torch.save(model.state_dict(), checkpoint_dir / "cellstate_best.pt")
    logger.info("Saved to %s", checkpoint_dir / "cellstate_best.pt")

    logger.info("=" * 60)
    logger.info("FINE-TUNING COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
