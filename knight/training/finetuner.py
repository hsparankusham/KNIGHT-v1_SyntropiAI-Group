"""Task-specific fine-tuning for KNIGHT v1.

Provides two fine-tuning trainers:

* :class:`CellStateFinetuner` — cell state classification with focal loss,
  class weighting, optional encoder freezing during warmup, and balanced-
  accuracy evaluation.
* :class:`PerturbationFinetuner` — perturbation-response prediction using a
  combined MSE + Pearson-correlation objective with gene-wise and cell-wise
  correlation metrics.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from knight.training.losses import CombinedPerturbationLoss, FocalLoss
from knight.training.schedulers import get_scheduler

logger = logging.getLogger(__name__)


# ===================================================================
# Cell-state classification fine-tuner
# ===================================================================

class CellStateFinetuner:
    """Fine-tuner for cell-state classification.

    Optionally freezes the encoder for the first *N* warmup epochs and then
    unfreezes it for end-to-end training.  Uses :class:`FocalLoss` to handle
    the heavy class imbalance typical of neuroimmune cell-state datasets.

    Parameters
    ----------
    model : nn.Module
        Full model (encoder + classification head).  The encoder is expected
        to be accessible as ``model.encoder`` for selective freezing.
    train_loader : DataLoader
        Yields ``(expression, label)`` tuples.
    val_loader : DataLoader
        Validation dataloader.
    config : dict
        Training hyper-parameters.  Expected keys:

        * ``lr`` — learning rate (default ``5e-5``).
        * ``weight_decay`` — AdamW weight decay (default ``0.01``).
        * ``max_grad_norm`` — gradient-clipping norm (default ``1.0``).
        * ``fp16`` — mixed precision (default ``True``).
        * ``num_classes`` — number of cell-state classes.
        * ``class_weights`` — optional list/tensor of per-class weights.
        * ``focal_gamma`` — focusing parameter for focal loss (default ``2.0``).
        * ``freeze_encoder_epochs`` — epochs to keep encoder frozen (default ``0``).
        * ``patience`` — early-stopping patience in epochs (default ``5``).
        * ``scheduler``, ``warmup_steps``, ``total_steps``, ``min_lr`` — scheduler params.
        * ``checkpoint_dir`` — directory for checkpoints.
        * ``use_wandb`` — enable W&B logging (default ``False``).
    device : torch.device | str
        Target device.
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
        self.lr: float = config.get("lr", 5e-5)
        self.weight_decay: float = config.get("weight_decay", 0.01)
        self.max_grad_norm: float = config.get("max_grad_norm", 1.0)
        self.fp16: bool = config.get("fp16", True)
        self.num_classes: int = config["num_classes"]
        self.freeze_encoder_epochs: int = config.get("freeze_encoder_epochs", 0)
        self.patience: int = config.get("patience", 5)
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Loss — Focal loss with optional per-class weighting
        class_weights = config.get("class_weights", None)
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
        self.criterion = FocalLoss(
            gamma=config.get("focal_gamma", 2.0),
            alpha=class_weights,
        )

        # Optimiser
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Scheduler
        scheduler_name: str = config.get("scheduler", "cosine_warmup")
        self.scheduler = get_scheduler(scheduler_name, self.optimizer, config)

        # Mixed precision
        self.scaler = GradScaler(enabled=self.fp16)

        # Tracking
        self.global_step: int = 0
        self.best_val_metric: float = 0.0  # balanced accuracy (higher is better)
        self.epochs_without_improvement: int = 0
        self._encoder_frozen: bool = False

        # W&B
        self.use_wandb: bool = config.get("use_wandb", False)
        if self.use_wandb:
            self._init_wandb(project_default="knight-cellstate")

        logger.info(
            "CellStateFinetuner initialised  classes=%d  freeze_epochs=%d  focal_gamma=%.1f",
            self.num_classes,
            self.freeze_encoder_epochs,
            config.get("focal_gamma", 2.0),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_wandb(self, project_default: str) -> None:
        try:
            import wandb

            if wandb.run is None:
                wandb.init(
                    project=self.config.get("wandb_project", project_default),
                    name=self.config.get("wandb_run_name", None),
                    config=self.config,
                )
        except ImportError:
            logger.warning("wandb not installed — disabling W&B logging.")
            self.use_wandb = False

    def _freeze_encoder(self) -> None:
        """Freeze encoder parameters."""
        encoder = getattr(self.model, "encoder", None)
        if encoder is None:
            logger.warning("model.encoder not found — skipping freeze.")
            return
        for param in encoder.parameters():
            param.requires_grad = False
        self._encoder_frozen = True
        logger.info("Encoder frozen.")

    def _unfreeze_encoder(self) -> None:
        """Unfreeze encoder parameters."""
        encoder = getattr(self.model, "encoder", None)
        if encoder is None:
            return
        for param in encoder.parameters():
            param.requires_grad = True
        self._encoder_frozen = False
        logger.info("Encoder unfrozen for end-to-end training.")

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def train_epoch(self) -> dict[str, float]:
        """Run one training epoch with cross-entropy / focal loss.

        Returns
        -------
        dict
            ``"train_loss"`` averaged over the epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for expr, labels in self.train_loader:
            expr = expr.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.fp16):
                logits = self.model(expr)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            n_batches += 1
            self.global_step += 1

        return {"train_loss": total_loss / max(n_batches, 1)}

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self) -> dict[str, Any]:
        """Evaluate on the validation set.

        Returns
        -------
        dict
            ``"val_loss"``, ``"balanced_accuracy"``, ``"per_class_f1"`` (list),
            ``"confusion_matrix"`` (ndarray).
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        for expr, labels in self.val_loader:
            expr = expr.to(self.device)
            labels = labels.to(self.device)

            with autocast(enabled=self.fp16):
                logits = self.model(expr)
                loss = self.criterion(logits, labels)

            total_loss += loss.item()
            n_batches += 1
            all_preds.append(logits.argmax(dim=-1).cpu())
            all_labels.append(labels.cpu())

        avg_loss = total_loss / max(n_batches, 1)

        preds_np = torch.cat(all_preds).numpy()
        labels_np = torch.cat(all_labels).numpy()

        # Balanced accuracy
        bal_acc = self._balanced_accuracy(labels_np, preds_np)

        # Per-class F1
        per_class_f1 = self._per_class_f1(labels_np, preds_np)

        # Confusion matrix
        cm = self._confusion_matrix(labels_np, preds_np)

        return {
            "val_loss": avg_loss,
            "balanced_accuracy": bal_acc,
            "per_class_f1": per_class_f1,
            "confusion_matrix": cm,
        }

    # ------------------------------------------------------------------
    # Metric helpers (avoid hard sklearn dependency)
    # ------------------------------------------------------------------

    @staticmethod
    def _balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute balanced accuracy (mean per-class recall)."""
        classes = np.unique(y_true)
        recalls: list[float] = []
        for c in classes:
            mask = y_true == c
            if mask.sum() == 0:
                continue
            recalls.append((y_pred[mask] == c).sum() / mask.sum())
        return float(np.mean(recalls)) if recalls else 0.0

    def _per_class_f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> list[float]:
        """Compute per-class F1 scores."""
        f1s: list[float] = []
        for c in range(self.num_classes):
            tp = ((y_pred == c) & (y_true == c)).sum()
            fp = ((y_pred == c) & (y_true != c)).sum()
            fn = ((y_pred != c) & (y_true == c)).sum()
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            if precision + recall == 0:
                f1s.append(0.0)
            else:
                f1s.append(float(2 * precision * recall / (precision + recall)))
        return f1s

    def _confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute confusion matrix [num_classes, num_classes]."""
        cm = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        for t, p in zip(y_true, y_pred):
            cm[int(t), int(p)] += 1
        return cm

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(self, n_epochs: int) -> dict[str, list[float]]:
        """Run the full fine-tuning loop with early stopping.

        Parameters
        ----------
        n_epochs : int
            Maximum number of epochs.

        Returns
        -------
        dict
            History with ``"train_loss"``, ``"val_loss"``, ``"balanced_accuracy"``.
        """
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "balanced_accuracy": [],
        }

        # Optionally freeze encoder for warmup
        if self.freeze_encoder_epochs > 0:
            self._freeze_encoder()

        logger.info("Starting cell-state fine-tuning for up to %d epochs.", n_epochs)

        for epoch in range(1, n_epochs + 1):
            # Unfreeze encoder after warmup phase
            if self._encoder_frozen and epoch > self.freeze_encoder_epochs:
                self._unfreeze_encoder()

            t0 = time.time()
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            elapsed = time.time() - t0

            history["train_loss"].append(train_metrics["train_loss"])
            history["val_loss"].append(val_metrics["val_loss"])
            history["balanced_accuracy"].append(val_metrics["balanced_accuracy"])

            mean_f1 = float(np.mean(val_metrics["per_class_f1"]))

            logger.info(
                "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  bal_acc=%.4f  "
                "mean_f1=%.4f  time=%.1fs",
                epoch,
                n_epochs,
                train_metrics["train_loss"],
                val_metrics["val_loss"],
                val_metrics["balanced_accuracy"],
                mean_f1,
                elapsed,
            )

            if self.use_wandb:
                import wandb

                wandb.log(
                    {
                        "epoch": epoch,
                        "train/loss": train_metrics["train_loss"],
                        "val/loss": val_metrics["val_loss"],
                        "val/balanced_accuracy": val_metrics["balanced_accuracy"],
                        "val/mean_f1": mean_f1,
                    },
                    step=self.global_step,
                )

            # Early stopping on balanced accuracy
            if val_metrics["balanced_accuracy"] > self.best_val_metric:
                self.best_val_metric = val_metrics["balanced_accuracy"]
                self.epochs_without_improvement = 0
                ckpt_path = self.checkpoint_dir / "best_cellstate.pt"
                self._save_checkpoint(ckpt_path)
                logger.info(
                    "New best balanced_accuracy=%.4f — checkpoint saved.",
                    self.best_val_metric,
                )
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    logger.info(
                        "Early stopping after %d epochs without improvement.",
                        self.patience,
                    )
                    break

        logger.info(
            "Cell-state fine-tuning complete.  Best balanced_accuracy=%.4f",
            self.best_val_metric,
        )
        return history

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _save_checkpoint(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "global_step": self.global_step,
                "best_val_metric": self.best_val_metric,
                "config": self.config,
            },
            path,
        )


# ===================================================================
# Perturbation-response fine-tuner
# ===================================================================

class PerturbationFinetuner:
    """Fine-tuner for perturbation-response (delta-expression) prediction.

    Uses a combined MSE + Pearson-correlation loss so that the model
    simultaneously minimises pointwise error and preserves the correlation
    structure across genes.

    Parameters
    ----------
    model : nn.Module
        Model that takes expression input and returns predicted delta-expression.
    train_loader : DataLoader
        Yields ``(expression, delta_expression)`` tuples.
    val_loader : DataLoader
        Validation dataloader.
    config : dict
        Expected keys:

        * ``lr`` — learning rate (default ``5e-5``).
        * ``weight_decay`` — AdamW weight decay (default ``0.01``).
        * ``max_grad_norm`` — gradient clipping (default ``1.0``).
        * ``fp16`` — mixed precision (default ``True``).
        * ``mse_weight`` — weight for MSE term (default ``1.0``).
        * ``pearson_weight`` — weight for Pearson term (default ``1.0``).
        * ``freeze_encoder_epochs`` — epochs with frozen encoder (default ``0``).
        * ``patience`` — early-stopping patience (default ``5``).
        * ``scheduler``, ``warmup_steps``, ``total_steps``, ``min_lr`` — scheduler.
        * ``checkpoint_dir``, ``use_wandb``, ``wandb_project``, ``wandb_run_name``.
    device : torch.device | str
        Target device.
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

        self.lr: float = config.get("lr", 5e-5)
        self.weight_decay: float = config.get("weight_decay", 0.01)
        self.max_grad_norm: float = config.get("max_grad_norm", 1.0)
        self.fp16: bool = config.get("fp16", True)
        self.freeze_encoder_epochs: int = config.get("freeze_encoder_epochs", 0)
        self.patience: int = config.get("patience", 5)
        self.checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Loss
        self.criterion = CombinedPerturbationLoss(
            mse_weight=config.get("mse_weight", 1.0),
            pearson_weight=config.get("pearson_weight", 1.0),
        )

        # Optimiser & scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        scheduler_name: str = config.get("scheduler", "cosine_warmup")
        self.scheduler = get_scheduler(scheduler_name, self.optimizer, config)

        self.scaler = GradScaler(enabled=self.fp16)

        self.global_step: int = 0
        self.best_val_loss: float = float("inf")
        self.epochs_without_improvement: int = 0
        self._encoder_frozen: bool = False

        self.use_wandb: bool = config.get("use_wandb", False)
        if self.use_wandb:
            self._init_wandb()

        logger.info(
            "PerturbationFinetuner initialised  mse_w=%.2f  pearson_w=%.2f",
            config.get("mse_weight", 1.0),
            config.get("pearson_weight", 1.0),
        )

    def _init_wandb(self) -> None:
        try:
            import wandb

            if wandb.run is None:
                wandb.init(
                    project=self.config.get("wandb_project", "knight-perturbation"),
                    name=self.config.get("wandb_run_name", None),
                    config=self.config,
                )
        except ImportError:
            logger.warning("wandb not installed — disabling.")
            self.use_wandb = False

    def _freeze_encoder(self) -> None:
        encoder = getattr(self.model, "encoder", None)
        if encoder is None:
            logger.warning("model.encoder not found — skipping freeze.")
            return
        for p in encoder.parameters():
            p.requires_grad = False
        self._encoder_frozen = True
        logger.info("Encoder frozen.")

    def _unfreeze_encoder(self) -> None:
        encoder = getattr(self.model, "encoder", None)
        if encoder is None:
            return
        for p in encoder.parameters():
            p.requires_grad = True
        self._encoder_frozen = False
        logger.info("Encoder unfrozen.")

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def train_epoch(self) -> dict[str, float]:
        """Run one epoch — predict delta expression, loss = MSE + (1 - Pearson r).

        Returns
        -------
        dict
            ``"train_loss"``, ``"train_mse"``, ``"train_pearson"``.
        """
        self.model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_pearson = 0.0
        n_batches = 0

        for expr, delta_expr in self.train_loader:
            expr = expr.to(self.device)
            delta_expr = delta_expr.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.fp16):
                pred_delta = self.model(expr)
                loss, components = self.criterion(pred_delta, delta_expr)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            total_mse += components["mse"]
            total_pearson += components["pearson"]
            n_batches += 1
            self.global_step += 1

        denom = max(n_batches, 1)
        return {
            "train_loss": total_loss / denom,
            "train_mse": total_mse / denom,
            "train_pearson": total_pearson / denom,
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Evaluate perturbation prediction on the validation set.

        Returns
        -------
        dict
            ``"val_loss"``, ``"val_mse"``, ``"val_pearson_loss"``,
            ``"genewise_pearson"`` (mean gene-wise *r*),
            ``"cellwise_pearson"`` (mean cell-wise *r*).
        """
        self.model.eval()
        total_loss = 0.0
        total_mse = 0.0
        total_pearson = 0.0
        n_batches = 0

        all_preds: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []

        for expr, delta_expr in self.val_loader:
            expr = expr.to(self.device)
            delta_expr = delta_expr.to(self.device)

            with autocast(enabled=self.fp16):
                pred_delta = self.model(expr)
                loss, components = self.criterion(pred_delta, delta_expr)

            total_loss += loss.item()
            total_mse += components["mse"]
            total_pearson += components["pearson"]
            n_batches += 1

            all_preds.append(pred_delta.cpu().float())
            all_targets.append(delta_expr.cpu().float())

        denom = max(n_batches, 1)

        # Aggregate predictions for correlation computation
        preds_cat = torch.cat(all_preds, dim=0)   # [N, G]
        tgts_cat = torch.cat(all_targets, dim=0)   # [N, G]

        genewise_r = self._pearson_along_dim(preds_cat, tgts_cat, dim=0)  # per-gene
        cellwise_r = self._pearson_along_dim(preds_cat, tgts_cat, dim=1)  # per-cell

        return {
            "val_loss": total_loss / denom,
            "val_mse": total_mse / denom,
            "val_pearson_loss": total_pearson / denom,
            "genewise_pearson": float(genewise_r.mean()),
            "cellwise_pearson": float(cellwise_r.mean()),
        }

    @staticmethod
    def _pearson_along_dim(
        x: torch.Tensor,
        y: torch.Tensor,
        dim: int,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Compute Pearson *r* along *dim* (0 = gene-wise, 1 = cell-wise).

        Returns a 1-D tensor of correlations for each slice along the
        *other* dimension.
        """
        x_c = x - x.mean(dim=dim, keepdim=True)
        y_c = y - y.mean(dim=dim, keepdim=True)
        cov = (x_c * y_c).sum(dim=dim)
        std_x = x_c.pow(2).sum(dim=dim).sqrt().clamp(min=eps)
        std_y = y_c.pow(2).sum(dim=dim).sqrt().clamp(min=eps)
        return cov / (std_x * std_y)

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(self, n_epochs: int) -> dict[str, list[float]]:
        """Run perturbation fine-tuning with early stopping.

        Parameters
        ----------
        n_epochs : int
            Maximum number of epochs.

        Returns
        -------
        dict
            Training history.
        """
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "genewise_pearson": [],
            "cellwise_pearson": [],
        }

        if self.freeze_encoder_epochs > 0:
            self._freeze_encoder()

        logger.info("Starting perturbation fine-tuning for up to %d epochs.", n_epochs)

        for epoch in range(1, n_epochs + 1):
            if self._encoder_frozen and epoch > self.freeze_encoder_epochs:
                self._unfreeze_encoder()

            t0 = time.time()
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            elapsed = time.time() - t0

            history["train_loss"].append(train_metrics["train_loss"])
            history["val_loss"].append(val_metrics["val_loss"])
            history["genewise_pearson"].append(val_metrics["genewise_pearson"])
            history["cellwise_pearson"].append(val_metrics["cellwise_pearson"])

            logger.info(
                "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  "
                "gene_r=%.4f  cell_r=%.4f  time=%.1fs",
                epoch,
                n_epochs,
                train_metrics["train_loss"],
                val_metrics["val_loss"],
                val_metrics["genewise_pearson"],
                val_metrics["cellwise_pearson"],
                elapsed,
            )

            if self.use_wandb:
                import wandb

                wandb.log(
                    {
                        "epoch": epoch,
                        "train/loss": train_metrics["train_loss"],
                        "train/mse": train_metrics["train_mse"],
                        "train/pearson_loss": train_metrics["train_pearson"],
                        "val/loss": val_metrics["val_loss"],
                        "val/mse": val_metrics["val_mse"],
                        "val/genewise_pearson": val_metrics["genewise_pearson"],
                        "val/cellwise_pearson": val_metrics["cellwise_pearson"],
                    },
                    step=self.global_step,
                )

            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self.epochs_without_improvement = 0
                ckpt_path = self.checkpoint_dir / "best_perturbation.pt"
                self._save_checkpoint(ckpt_path)
                logger.info("New best val_loss=%.4f — checkpoint saved.", self.best_val_loss)
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    logger.info(
                        "Early stopping after %d epochs without improvement.",
                        self.patience,
                    )
                    break

        logger.info(
            "Perturbation fine-tuning complete.  Best val_loss=%.4f",
            self.best_val_loss,
        )
        return history

    def _save_checkpoint(self, path: Path) -> None:
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
