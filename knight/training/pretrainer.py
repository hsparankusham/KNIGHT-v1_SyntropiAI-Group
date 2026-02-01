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
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from knight.training.losses import MaskedMSELoss
from knight.training.schedulers import get_scheduler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standalone helper — also usable outside the class
# ---------------------------------------------------------------------------

def masked_mse_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute MSE only on masked positions.

    Parameters
    ----------
    predicted : Tensor [B, G]
        Model output for all genes.
    target : Tensor [B, G]
        Ground-truth expression values.
    mask : Tensor [B, G]
        Boolean / binary mask — ``1`` where the gene was masked.

    Returns
    -------
    Tensor
        Scalar mean-squared error over masked entries.
    """
    mask_f = mask.float()
    sq_err = (predicted - target) ** 2 * mask_f
    return sq_err.sum() / mask_f.sum().clamp(min=1.0)


# ---------------------------------------------------------------------------
# Pretrainer
# ---------------------------------------------------------------------------

class Pretrainer:
    """Self-supervised masked gene-expression pretrainer.

    Parameters
    ----------
    model : nn.Module
        The KNIGHT encoder (or encoder + reconstruction head).
    train_loader : DataLoader
        Training dataloader yielding batches of expression tensors.
    val_loader : DataLoader
        Validation dataloader.
    config : dict
        Training hyper-parameters.  Expected keys:

        * ``lr`` — peak learning rate (default ``1e-4``).
        * ``weight_decay`` — AdamW weight decay (default ``0.01``).
        * ``mask_ratio`` — fraction of genes to mask (default ``0.15``).
        * ``max_grad_norm`` — gradient-clipping threshold (default ``1.0``).
        * ``fp16`` — whether to use mixed precision (default ``True``).
        * ``scheduler`` — scheduler name (default ``"cosine_warmup"``).
        * ``warmup_steps`` — warmup steps for the scheduler.
        * ``total_steps`` — total training steps.
        * ``min_lr`` — minimum LR for cosine decay (default ``1e-7``).
        * ``patience`` — early-stopping patience in epochs (default ``5``).
        * ``checkpoint_dir`` — directory for saving checkpoints.
        * ``use_wandb`` — enable Weights & Biases logging (default ``False``).
        * ``wandb_project`` — W&B project name.
        * ``wandb_run_name`` — W&B run name.
    device : torch.device | str
        Device to train on.
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

        # Hyper-parameters with sensible defaults
        self.lr: float = config.get("lr", 1e-4)
        self.weight_decay: float = config.get("weight_decay", 0.01)
        self.mask_ratio: float = config.get("mask_ratio", 0.15)
        self.max_grad_norm: float = config.get("max_grad_norm", 1.0)
        self.fp16: bool = config.get("fp16", True)
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

        # Scheduler
        scheduler_name: str = config.get("scheduler", "cosine_warmup")
        self.scheduler = get_scheduler(scheduler_name, self.optimizer, config)

        # Mixed-precision scaler
        self.scaler = GradScaler(enabled=self.fp16)

        # Tracking
        self.global_step: int = 0
        self.best_val_loss: float = float("inf")
        self.epochs_without_improvement: int = 0

        # Optional W&B
        self.use_wandb: bool = config.get("use_wandb", False)
        if self.use_wandb:
            try:
                import wandb  # noqa: F811

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
            "Pretrainer initialised  lr=%.2e  mask_ratio=%.2f  fp16=%s  device=%s",
            self.lr,
            self.mask_ratio,
            self.fp16,
            self.device,
        )

    # ------------------------------------------------------------------
    # Masking utility
    # ------------------------------------------------------------------

    def _generate_mask(self, batch: torch.Tensor) -> torch.Tensor:
        """Create a random binary mask with ``self.mask_ratio`` fraction of genes set to 1."""
        return (torch.rand_like(batch) < self.mask_ratio).float()

    # ------------------------------------------------------------------
    # Single training epoch
    # ------------------------------------------------------------------

    def train_epoch(self) -> dict[str, float]:
        """Run one training epoch.

        Returns
        -------
        dict
            ``"train_loss"`` averaged over the epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            if isinstance(batch, (list, tuple)):
                expr = batch[0].to(self.device)
            else:
                expr = batch.to(self.device)

            # Mask random genes
            mask = self._generate_mask(expr)
            masked_input = expr * (1.0 - mask)  # zero out masked positions

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.fp16):
                predicted = self.model(masked_input)
                loss = self.criterion(predicted, expr, mask)

            self.scaler.scale(loss).backward()

            # Gradient clipping (unscale first for correct norm)
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            n_batches += 1
            self.global_step += 1

            # Per-step W&B logging
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
        """Evaluate on the validation set.

        Returns
        -------
        dict
            ``"val_loss"`` and ``"val_r2"`` (R-squared on masked positions).
        """
        self.model.eval()
        total_loss = 0.0
        ss_res = 0.0
        ss_tot = 0.0
        n_batches = 0

        for batch in self.val_loader:
            if isinstance(batch, (list, tuple)):
                expr = batch[0].to(self.device)
            else:
                expr = batch.to(self.device)

            mask = self._generate_mask(expr)
            masked_input = expr * (1.0 - mask)

            with autocast(enabled=self.fp16):
                predicted = self.model(masked_input)
                loss = self.criterion(predicted, expr, mask)

            total_loss += loss.item()

            # Reconstruction R² on masked positions only
            mask_bool = mask.bool()
            pred_masked = predicted[mask_bool]
            true_masked = expr[mask_bool]
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
        """Run the full pretraining loop.

        Parameters
        ----------
        n_epochs : int
            Maximum number of epochs.

        Returns
        -------
        dict
            History with keys ``"train_loss"``, ``"val_loss"``, ``"val_r2"``.
        """
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

            # W&B epoch-level logging
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

            # Checkpoint & early stopping
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
