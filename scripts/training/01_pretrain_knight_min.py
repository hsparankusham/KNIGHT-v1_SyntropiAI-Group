#!/usr/bin/env python3
"""Step T1: Pretrain KNIGHT-min (100M param prototype).

Masked gene expression modeling on brain cell atlas data.
This is the foundational pretraining step — learn general cell representations
before task-specific fine-tuning.

Usage:
    python scripts/training/01_pretrain_knight_min.py [--config config/defaults.yaml]
    python scripts/training/01_pretrain_knight_min.py --smoke-test  # verify pipeline
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from knight.data.datasets.cell_dataset import CellExpressionDataset
from knight.model.architectures.knight_encoder import KnightEncoder, KnightEncoderConfig
from knight.training.pretrainer import Pretrainer, ReconstructionHead
from knight.utils.io import load_config, ensure_dir
from knight.utils.logging import setup_logging
from knight.utils.reproducibility import set_seed, get_device, count_parameters

logger = setup_logging("pretrain_knight_min")


class KnightPretrainModel(nn.Module):
    """Encoder + reconstruction head for masked expression pretraining.

    Takes dict-based batches from CellExpressionDataset and outputs
    predicted expression values for all gene positions.
    """

    def __init__(self, encoder: KnightEncoder, max_genes: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.reconstruction_head = ReconstructionHead(encoder.d_model, max_genes)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        cell_embedding = self.encoder(
            gene_ids=batch["gene_ids"],
            expression_values=batch["expression_values"],
            padding_mask=batch.get("padding_mask"),
        )
        return self.reconstruction_head(cell_embedding)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain KNIGHT-min")
    parser.add_argument("--config", type=Path, default=Path("config/defaults.yaml"))
    parser.add_argument("--resume", type=Path, default=None, help="Resume from checkpoint")
    parser.add_argument("--smoke-test", action="store_true", help="Quick test with synthetic data")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("KNIGHT-min Pretraining")
    logger.info("=" * 60)

    config = load_config(args.config)
    set_seed(42)
    device = get_device()
    logger.info("Device: %s", device)

    model_cfg = config.get("model", {}).get("knight_min", {})
    train_cfg = config.get("training", {}).get("pretraining", {})
    compute_cfg = config.get("compute", {})

    # Override fp16 based on device
    if device.type != "cuda":
        train_cfg["fp16"] = False

    max_genes = model_cfg.get("max_seq_len", 1200)

    if args.smoke_test:
        # Generate synthetic data for pipeline verification
        logger.info("=== SMOKE TEST MODE ===")
        import anndata as ad
        import numpy as np
        from scipy.sparse import random as sp_random

        n_train, n_val, n_genes_data = 500, 100, 2000
        gene_names = [f"GENE_{i}" for i in range(n_genes_data)]

        train_X = sp_random(n_train, n_genes_data, density=0.3, format="csr").astype(np.float32)
        val_X = sp_random(n_val, n_genes_data, density=0.3, format="csr").astype(np.float32)

        train_adata = ad.AnnData(X=train_X)
        train_adata.var_names = gene_names
        val_adata = ad.AnnData(X=val_X)
        val_adata.var_names = gene_names

        gene_vocab = {g: i + 4 for i, g in enumerate(gene_names)}  # +4 for special tokens

        # Reduce for smoke test
        train_cfg["epochs"] = 3
        train_cfg["warmup_steps"] = 10
        batch_size = 32
    else:
        # Load real data
        splits_dir = Path("data/splits")
        train_adata = sc.read_h5ad(splits_dir / "train.h5ad")
        val_adata = sc.read_h5ad(splits_dir / "val.h5ad")

        # Build or load gene vocabulary
        gene_vocab_path = Path("data/processed/gene_vocab.json")
        if gene_vocab_path.exists():
            with open(gene_vocab_path) as f:
                gene_vocab = json.load(f)
        else:
            all_genes = sorted(set(train_adata.var_names))
            gene_vocab = {g: i + 4 for i, g in enumerate(all_genes)}
            ensure_dir(gene_vocab_path.parent)
            with open(gene_vocab_path, "w") as f:
                json.dump(gene_vocab, f)

        batch_size = train_cfg.get("batch_size", 32)

    logger.info("Train: %d cells, Val: %d cells", train_adata.n_obs, val_adata.n_obs)

    # Create datasets
    train_dataset = CellExpressionDataset(train_adata, gene_vocab, max_genes=max_genes)
    val_dataset = CellExpressionDataset(val_adata, gene_vocab, max_genes=max_genes)

    num_workers = compute_cfg.get("num_workers", 2)
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Build model: encoder + reconstruction head
    encoder_config = KnightEncoderConfig(
        n_genes=len(gene_vocab) + 4,  # +4 for CLS, PAD, MASK, UNK
        d_model=model_cfg.get("d_model", 256),
        n_layers=model_cfg.get("n_layers", 6),
        n_heads=model_cfg.get("n_heads", 4),
        dropout=model_cfg.get("dropout", 0.1),
        max_seq_len=max_genes + 1,
    )
    encoder = KnightEncoder(config=encoder_config)
    model = KnightPretrainModel(encoder, max_genes=max_genes)
    model = model.to(device)

    params = count_parameters(model)
    logger.info("Model parameters: %s", params)

    # Train
    checkpoint_dir = ensure_dir(Path("models/checkpoints/knight_min"))
    train_cfg["checkpoint_dir"] = str(checkpoint_dir)

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
    final_path = checkpoint_dir / "knight_min_final.pt"
    torch.save(model.state_dict(), final_path)
    logger.info("Saved final model to %s", final_path)

    # Save just the encoder (for downstream fine-tuning)
    encoder_path = checkpoint_dir / "knight_min_encoder.pt"
    torch.save(encoder.state_dict(), encoder_path)
    logger.info("Saved encoder weights to %s", encoder_path)

    logger.info("=" * 60)
    logger.info("PRETRAINING COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
