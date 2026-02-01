#!/usr/bin/env python3
"""Step T1: Pretrain KNIGHT-min (100M param prototype).

Masked gene expression modeling on the integrated brain cell atlas.
This is the foundational pretraining step — learn general cell representations
before task-specific fine-tuning.

Usage:
    python scripts/training/01_pretrain_knight_min.py [--config config/defaults.yaml]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import scanpy as sc
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from knight.data.datasets.cell_dataset import CellExpressionDataset
from knight.model.architectures.knight_encoder import KnightMinEncoder
from knight.model.architectures.knight_model import build_knight_model, KnightConfig
from knight.model.layers.gene_expression_encoder import GeneTokenizer
from knight.training.pretrainer import Pretrainer
from knight.utils.io import load_config, ensure_dir
from knight.utils.logging import setup_logging
from knight.utils.reproducibility import set_seed, get_device, count_parameters

logger = setup_logging("pretrain_knight_min")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain KNIGHT-min")
    parser.add_argument("--config", type=Path, default=Path("config/defaults.yaml"))
    parser.add_argument("--resume", type=Path, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("KNIGHT-min Pretraining")
    logger.info("=" * 60)

    config = load_config(args.config)
    set_seed(42)
    device = get_device()
    logger.info("Device: %s", device)

    # Load data
    splits_dir = Path("data/splits")
    train_adata = sc.read_h5ad(splits_dir / "train.h5ad")
    val_adata = sc.read_h5ad(splits_dir / "val.h5ad")
    logger.info("Train: %d cells, Val: %d cells", train_adata.n_obs, val_adata.n_obs)

    # Build gene vocabulary from training data
    gene_vocab_path = Path("data/processed/gene_vocab.json")
    if gene_vocab_path.exists():
        import json
        with open(gene_vocab_path) as f:
            gene_vocab = json.load(f)
    else:
        tokenizer = GeneTokenizer.build_vocab(list(train_adata.var_names))
        gene_vocab = tokenizer.vocab
        ensure_dir(gene_vocab_path.parent)
        import json
        with open(gene_vocab_path, "w") as f:
            json.dump(gene_vocab, f)

    # Create datasets
    model_cfg = config.get("model", {}).get("knight_min", {})
    max_genes = model_cfg.get("n_genes", 4000)

    train_dataset = CellExpressionDataset(train_adata, gene_vocab, max_genes=max_genes)
    val_dataset = CellExpressionDataset(val_adata, gene_vocab, max_genes=max_genes)

    train_cfg = config.get("training", {}).get("pretraining", {})
    batch_size = train_cfg.get("batch_size", 128)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.get("compute", {}).get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.get("compute", {}).get("num_workers", 4),
        pin_memory=True,
    )

    # Build model
    model_config = KnightConfig(
        n_genes=len(gene_vocab),
        d_model=model_cfg.get("d_model", 256),
        n_layers=model_cfg.get("n_layers", 6),
        n_heads=model_cfg.get("n_heads", 4),
        dropout=model_cfg.get("dropout", 0.1),
    )
    model = build_knight_model(model_config)
    model = model.to(device)

    params = count_parameters(model)
    logger.info("Model parameters: %s", params)

    # Train
    checkpoint_dir = ensure_dir(Path("models/checkpoints/knight_min"))
    pretrainer = Pretrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_cfg,
        device=device,
    )

    if args.resume:
        pretrainer.load_checkpoint(args.resume)

    pretrainer.train(n_epochs=train_cfg.get("epochs", 50))

    # Save final model
    torch.save(model.state_dict(), checkpoint_dir / "knight_min_final.pt")
    logger.info("Saved final model to %s", checkpoint_dir / "knight_min_final.pt")

    logger.info("=" * 60)
    logger.info("PRETRAINING COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
