"""Masked gene-expression pretraining for KNIGHT v1.

Implements self-supervised pretraining in which a random subset of gene
expression values is masked and the model learns to reconstruct them.  This is
the genomics analogue of masked-language modelling and forms the foundation
training stage before any task-specific fine-tuning.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from knight.training.losses import MaskedMSELoss
from knight.training.schedulers import get_scheduler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reconstruction head for pretraining
# ---------------------------------------------------------------------------

class ReconstructionHead(nn.Module):
    """Maps cell embeddings back to per-gene expression predictions.

    Used during masked expression pretraining: the encoder produces a cell
    embedding, and this head projects it back to predict the original
    expression values for all input genes.
    """

    def __init__(self, d_model: int, max_genes: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, max_genes),
        )

    def forward(self, cell_embedding: torch.Tensor) -> torch.Tensor:
        """Predict expression values from cell embedding.

        Parameters
        ----------
        cell_embedding : Tensor [B, D]
            Cell embeddings from the encoder.

        Returns
        -------
        Tensor [B, max_genes]
            Predicted expression values.
        """
        return self.head(cell_embedding)


# ---------------------------------------------------------------------------
# Pretrainer
# ---------------------------------------------------------------------------

class Pretrainer:
    """Self-supervised masked gene-expression pretrainer.

    Works with dict-based batches from CellExpressionDataset, which yield:
    - gene_ids: (B, max_genes) int64
    - expression_values: (B, max_genes) float32
    - padding_mask: (B, max_genes) bool
    - n_genes: (B,) int64
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict[str, Any],
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(device)

        # Hyper-parameters
        self.lr: float = config.get("lr", 1e-4)
        self.weight_decay: float = config.get("weight_decay", 0.01)
        self.mask_ratio: float = config.get("mask_ratio", 0.15)
        self.max_grad_norm: float = config.get("max_grad_norm", 1.0)
        self.fp16: bool = config.get("fp16", True) and self.device.type == "cuda"
        self.patience: int = config.get("patience", 5)
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Loss
        self.criterion = MaskedMSELoss(reduction="mean")

        # Optimiser
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Compute total_steps for scheduler
        n_epochs = config.get("epochs", 50)
        steps_per_epoch = len(train_loader)
        total_steps = n_epochs * steps_per_epoch
        scheduler_config = {
            **config,
            "total_steps": total_steps,
            "warmup_steps": config.get("warmup_steps", min(2000, total_steps // 10)),
        }

        # Scheduler
        scheduler_name: str = config.get("scheduler", "cosine_warmup")
        self.scheduler = get_scheduler(scheduler_name, self.optimizer, scheduler_config)

        # Mixed-precision scaler
        self.scaler = GradScaler("cuda", enabled=self.fp16)

        # Tracking
        self.global_step: int = 0
        self.best_val_loss: float = float("inf")
        self.epochs_without_improvement: int = 0

        # Optional W&B
        self.use_wandb: bool = config.get("use_wandb", False)
        if self.use_wandb:
            try:
                import wandb

                if wandb.run is None:
                    wandb.init(
                        project=config.get("wandb_project", "knight-pretrain"),
                        name=config.get("wandb_run_name", None),
                        config=config,
                    )
                logger.info("Weights & Biases logging enabled.")
            except ImportError:
                logger.warning("wandb not installed — disabling W&B logging.")
                self.use_wandb = False

        logger.info(
            "Pretrainer initialised  lr=%.2e  mask_ratio=%.2f  fp16=%s  device=%s  "
            "total_steps=%d  warmup_steps=%d",
            self.lr,
            self.mask_ratio,
            self.fp16,
            self.device,
            total_steps,
            scheduler_config["warmup_steps"],
        )

    # ------------------------------------------------------------------
    # Batch handling
    # ------------------------------------------------------------------

    def _move_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Move all tensors in a batch dict to the training device."""
        return {k: v.to(self.device) for k, v in batch.items()}

    def _mask_expression(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Mask expression values in a batch for reconstruction.

        Returns the modified batch (with masked expression values),
        the original target values, and the binary mask.
        """
        expr = batch["expression_values"]  # (B, max_genes)
        padding = batch["padding_mask"]  # (B, max_genes) True=padding

        # Only mask non-padding positions
        rand = torch.rand_like(expr)
        mask = (rand < self.mask_ratio) & (~padding)  # (B, max_genes)

        # Zero out masked expression values
        masked_expr = expr.clone()
        masked_expr[mask] = 0.0

        masked_batch = {**batch, "expression_values": masked_expr}
        return masked_batch, expr, mask.float()

    # ------------------------------------------------------------------
    # Single training epoch
    # ------------------------------------------------------------------

    def train_epoch(self) -> dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            batch = self._move_batch(batch)
            masked_batch, target_expr, mask = self._mask_expression(batch)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=self.fp16):
                predicted = self.model(masked_batch)
                loss = self.criterion(predicted, target_expr, mask)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            n_batches += 1
            self.global_step += 1

            if self.use_wandb and self.global_step % 50 == 0:
                import wandb

                wandb.log(
                    {
                        "train/step_loss": loss.item(),
                        "train/lr": self.optimizer.param_groups[0]["lr"],
                        "global_step": self.global_step,
                    },
                    step=self.global_step,
                )

        avg_loss = total_loss / max(n_batches, 1)
        return {"train_loss": avg_loss}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Evaluate on the validation set."""
        self.model.eval()
        total_loss = 0.0
        ss_res = 0.0
        ss_tot = 0.0
        n_batches = 0

        for batch in self.val_loader:
            batch = self._move_batch(batch)
            masked_batch, target_expr, mask = self._mask_expression(batch)

            with autocast("cuda", enabled=self.fp16):
                predicted = self.model(masked_batch)
                loss = self.criterion(predicted, target_expr, mask)

            total_loss += loss.item()

            # R² on masked positions
            mask_bool = mask.bool()
            pred_masked = predicted[mask_bool]
            true_masked = target_expr[mask_bool]
            ss_res += ((pred_masked - true_masked) ** 2).sum().item()
            ss_tot += ((true_masked - true_masked.mean()) ** 2).sum().item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

        return {"val_loss": avg_loss, "val_r2": r2}

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(self, n_epochs: int) -> dict[str, list[float]]:
        """Run the full pretraining loop."""
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_r2": [],
        }

        logger.info("Starting pretraining for up to %d epochs.", n_epochs)

        for epoch in range(1, n_epochs + 1):
            t0 = time.time()

            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            elapsed = time.time() - t0

            history["train_loss"].append(train_metrics["train_loss"])
            history["val_loss"].append(val_metrics["val_loss"])
            history["val_r2"].append(val_metrics["val_r2"])

            logger.info(
                "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_r2=%.4f  "
                "lr=%.2e  time=%.1fs",
                epoch,
                n_epochs,
                train_metrics["train_loss"],
                val_metrics["val_loss"],
                val_metrics["val_r2"],
                self.optimizer.param_groups[0]["lr"],
                elapsed,
            )

            if self.use_wandb:
                import wandb

                wandb.log(
                    {
                        "epoch": epoch,
                        "train/epoch_loss": train_metrics["train_loss"],
                        "val/loss": val_metrics["val_loss"],
                        "val/r2": val_metrics["val_r2"],
                        "lr": self.optimizer.param_groups[0]["lr"],
                    },
                    step=self.global_step,
                )

            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.epochs_without_improvement = 0
                ckpt_path = self.checkpoint_dir / "best_pretrain.pt"
                self.save_checkpoint(ckpt_path)
                logger.info("New best val_loss=%.4f — checkpoint saved.", self.best_val_loss)
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    logger.info(
                        "Early stopping triggered after %d epochs without improvement.",
                        self.patience,
                    )
                    break

        logger.info("Pretraining complete.  Best val_loss=%.4f", self.best_val_loss)
        return history

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str | Path) -> None:
        """Save model, optimiser, scheduler, and scaler state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "global_step": self.global_step,
                "best_val_loss": self.best_val_loss,
                "config": self.config,
            },
            path,
        )
        logger.debug("Checkpoint saved to %s", path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore state from a checkpoint file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.scaler.load_state_dict(ckpt["scaler_state_dict"])
        self.global_step = ckpt["global_step"]
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        logger.info(
            "Checkpoint loaded from %s  (global_step=%d, best_val_loss=%.4f)",
            path,
            self.global_step,
            self.best_val_loss,
        )
