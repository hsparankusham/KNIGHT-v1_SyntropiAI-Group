#!/usr/bin/env python3
"""Step T2: Fine-tune KNIGHT for cell state classification.

Fine-tunes the pretrained encoder with a cell state classification head.
Loads encoder weights from pretraining checkpoint and adds a fresh
classification head for the cell types found in the training data.

Usage:
    python scripts/training/02_finetune_cellstate.py \
        --checkpoint models/checkpoints/knight_min/knight_min_encoder.pt \
        --config config/compute_kaggle.yaml
    python scripts/training/02_finetune_cellstate.py \
        --checkpoint models/checkpoints/knight_min/knight_min_encoder.pt \
        --smoke-test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import scanpy as sc
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from knight.data.datasets.cell_dataset import CellStateDataset
from knight.model.architectures.knight_model import (
    KnightConfig,
    KnightModel,
    HeadConfig,
)
from knight.training.finetuner import CellStateFinetuner
from knight.utils.io import load_config, ensure_dir
from knight.utils.logging import setup_logging
from knight.utils.reproducibility import set_seed, get_device, count_parameters

logger = setup_logging("finetune_cellstate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune for cell state classification")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Pretrained encoder checkpoint")
    parser.add_argument("--config", type=Path, default=Path("config/defaults.yaml"))
    parser.add_argument("--smoke-test", action="store_true", help="Quick test with synthetic data")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Cell State Classification Fine-tuning")
    logger.info("=" * 60)

    config = load_config(args.config)
    set_seed(42)
    device = get_device()
    logger.info("Device: %s", device)

    model_cfg = config.get("model", {}).get("knight_min", {})
    ft_cfg = config.get("training", {}).get("finetuning", {})
    compute_cfg = config.get("compute", {})

    # Override fp16 based on device
    if device.type != "cuda":
        ft_cfg["fp16"] = False

    max_genes = model_cfg.get("max_seq_len", 1200)

    if args.smoke_test:
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
        # Assign random cell types
        cell_types = ["astrocyte", "microglial cell"]
        train_adata.obs["cell_type"] = np.random.choice(cell_types, n_train)

        val_adata = ad.AnnData(X=val_X)
        val_adata.var_names = gene_names
        val_adata.obs["cell_type"] = np.random.choice(cell_types, n_val)

        gene_vocab = {g: i + 4 for i, g in enumerate(gene_names)}

        ft_cfg["epochs"] = 3
        ft_cfg["warmup_steps"] = 10
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

        batch_size = ft_cfg.get("batch_size", 64)

    # Build label map from training data
    label_key = "cell_type"
    all_labels = sorted(set(train_adata.obs[label_key].values))
    label_map = {l: i for i, l in enumerate(all_labels)}
    num_classes = len(label_map)
    logger.info("Cell states: %d classes — %s", num_classes, all_labels)

    # Create datasets
    train_dataset = CellStateDataset(train_adata, gene_vocab, label_key, label_map, max_genes)
    val_dataset = CellStateDataset(val_adata, gene_vocab, label_key, label_map, max_genes)

    logger.info("Train: %d cells, Val: %d cells", len(train_dataset), len(val_dataset))

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

    # Build model with cellstate head
    d_model = model_cfg.get("d_model", 256)
    model_config = KnightConfig(
        n_genes=len(gene_vocab) + 4,  # +4 for CLS, PAD, MASK, UNK
        d_model=d_model,
        n_layers=model_cfg.get("n_layers", 6),
        n_heads=model_cfg.get("n_heads", 4),
        dropout=model_cfg.get("dropout", 0.1),
        max_seq_len=max_genes + 1,
        head_configs={
            "cellstate": HeadConfig(
                head_type="cellstate",
                head_kwargs={"d_model": d_model, "n_classes": num_classes, "dropout": 0.1},
            ),
        },
    )
    model = KnightModel(model_config)

    # Load pretrained encoder weights
    checkpoint_path = args.checkpoint
    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        # Handle different checkpoint formats
        if "model_state_dict" in ckpt:
            # Full pretrainer checkpoint — extract encoder keys
            encoder_state = {}
            for k, v in ckpt["model_state_dict"].items():
                if k.startswith("encoder."):
                    encoder_state[k.replace("encoder.", "", 1)] = v
        elif any(k.startswith("encoder.") for k in ckpt.keys()):
            # Already has encoder. prefix
            encoder_state = {k.replace("encoder.", "", 1): v for k, v in ckpt.items() if k.startswith("encoder.")}
        else:
            # Raw encoder state dict
            encoder_state = ckpt

        missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)
        logger.info(
            "Loaded encoder from %s. Missing: %d, Unexpected: %d",
            checkpoint_path, len(missing), len(unexpected),
        )
        if missing:
            logger.warning("Missing encoder keys: %s", missing[:5])
        if unexpected:
            logger.warning("Unexpected encoder keys: %s", unexpected[:5])
    else:
        logger.warning("Checkpoint %s not found — training from scratch.", checkpoint_path)

    model = model.to(device)
    logger.info("Parameters: %s", count_parameters(model))

    # Fine-tune config
    checkpoint_dir = ensure_dir(Path("models/checkpoints/cellstate"))
    ft_cfg["checkpoint_dir"] = str(checkpoint_dir)
    ft_cfg["num_classes"] = num_classes
    ft_cfg.setdefault("freeze_encoder_epochs", 2)
    ft_cfg.setdefault("lr", 5e-5)
    ft_cfg.setdefault("focal_gamma", 2.0)

    finetuner = CellStateFinetuner(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=ft_cfg,
        device=device,
    )

    finetuner.train(n_epochs=ft_cfg.get("epochs", 30))

    # Save final model
    final_path = checkpoint_dir / "cellstate_final.pt"
    torch.save(model.state_dict(), final_path)
    logger.info("Saved final model to %s", final_path)

    # Save label map
    label_map_path = checkpoint_dir / "label_map.json"
    with open(label_map_path, "w") as f:
        json.dump(label_map, f, indent=2)
    logger.info("Saved label map to %s", label_map_path)

    logger.info("=" * 60)
    logger.info("FINE-TUNING COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
