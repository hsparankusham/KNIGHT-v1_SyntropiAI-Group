"""Learning-rate schedulers for KNIGHT v1 training.

Provides warmup-aware schedulers commonly used with transformer-based genomics
models:  cosine-with-warmup, linear-warmup-then-constant, and a factory
function for config-driven instantiation.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cosine warmup scheduler
# ---------------------------------------------------------------------------

class CosineWarmupScheduler(_LRScheduler):
    """Cosine annealing with linear warmup.

    During the first ``warmup_steps`` optimiser steps the learning rate
    increases linearly from 0 to the base LR.  After warmup the LR follows a
    cosine curve that decays to ``min_lr`` over the remaining
    ``total_steps - warmup_steps`` steps.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Wrapped optimiser.
    warmup_steps : int
        Number of linear-warmup steps.
    total_steps : int
        Total number of training steps (warmup + cosine decay).
    min_lr : float
        Minimum learning rate at the end of the cosine schedule.
    last_epoch : int
        Index of the last epoch (for resuming). Default ``-1``.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-7,
        last_epoch: int = -1,
    ) -> None:
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if total_steps < warmup_steps:
            raise ValueError(
                f"total_steps ({total_steps}) must be >= warmup_steps ({warmup_steps})"
            )
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> list[float]:  # type: ignore[override]
        """Compute the LR for the current step."""
        step = self.last_epoch  # _LRScheduler increments before calling get_lr

        if step < self.warmup_steps:
            # Linear warmup: 0 → base_lr
            scale = step / max(1, self.warmup_steps)
            return [base_lr * scale for base_lr in self.base_lrs]

        # Cosine decay phase
        decay_steps = self.total_steps - self.warmup_steps
        progress = (step - self.warmup_steps) / max(1, decay_steps)
        progress = min(progress, 1.0)
        cosine_scale = 0.5 * (1.0 + math.cos(math.pi * progress))

        return [
            self.min_lr + (base_lr - self.min_lr) * cosine_scale
            for base_lr in self.base_lrs
        ]


# ---------------------------------------------------------------------------
# Linear warmup then constant
# ---------------------------------------------------------------------------

class LinearWarmupScheduler(_LRScheduler):
    """Linear warmup followed by a constant learning rate.

    Useful for short fine-tuning runs where full cosine decay is unnecessary.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Wrapped optimiser.
    warmup_steps : int
        Number of linear-warmup steps.
    last_epoch : int
        Index of the last epoch (for resuming). Default ``-1``.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        last_epoch: int = -1,
    ) -> None:
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> list[float]:  # type: ignore[override]
        """Compute the LR for the current step."""
        step = self.last_epoch
        if step < self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
            return [base_lr * scale for base_lr in self.base_lrs]
        return list(self.base_lrs)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
) -> _LRScheduler:
    """Instantiate a scheduler by name from a configuration dictionary.

    Parameters
    ----------
    name : str
        One of ``"cosine_warmup"``, ``"linear_warmup"``, or ``"constant"``.
    optimizer : torch.optim.Optimizer
        The optimiser whose LR will be scheduled.
    config : dict
        Must contain the keys required by the chosen scheduler:

        * ``"cosine_warmup"`` — ``warmup_steps``, ``total_steps``,
          optionally ``min_lr``.
        * ``"linear_warmup"`` — ``warmup_steps``.
        * ``"constant"`` — no extra keys.

    Returns
    -------
    _LRScheduler
        Configured scheduler instance.

    Raises
    ------
    ValueError
        If *name* is not recognised.
    """
    name_lower = name.lower().replace("-", "_")

    if name_lower == "cosine_warmup":
        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_steps=config["warmup_steps"],
            total_steps=config["total_steps"],
            min_lr=config.get("min_lr", 1e-7),
        )
        logger.info(
            "Created CosineWarmupScheduler  warmup=%d  total=%d  min_lr=%.2e",
            config["warmup_steps"],
            config["total_steps"],
            config.get("min_lr", 1e-7),
        )
        return scheduler

    if name_lower == "linear_warmup":
        scheduler = LinearWarmupScheduler(
            optimizer,
            warmup_steps=config["warmup_steps"],
        )
        logger.info(
            "Created LinearWarmupScheduler  warmup=%d",
            config["warmup_steps"],
        )
        return scheduler

    if name_lower == "constant":
        # No-op scheduler: LR stays at the optimiser's initial value.
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        logger.info("Created constant LR scheduler")
        return scheduler

    raise ValueError(
        f"Unknown scheduler '{name}'. Choose from: cosine_warmup, linear_warmup, constant."
    )
