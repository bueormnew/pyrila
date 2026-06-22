"""RILA Trainer: full training loop with composite loss and gradient safety.

Supports:
- train_step(): single training step
- train(): full training loop over dataset
- evaluate(): evaluation mode
- save_checkpoint() / load_checkpoint(): model persistence
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from pyrila.training.losses import (
    UncertaintyWeightedLoss,
    compute_answer_loss,
    compute_budget_loss,
    compute_composite_loss,
    compute_confidence_loss,
    compute_retrieval_loss,
)
from pyrila.exceptions import CheckpointError, DivergenceError, GradientError
from pyrila.training.utils import compute_answer_correctness, find_bad_gradient_param, has_bad_gradients


class RILATrainer:
    """Training loop with composite loss, gradient safety, and divergence detection.

    Args:
        model: RILA model to train.
        optimizer: PyTorch optimizer.
        alpha: Retrieval cost coefficient.
        beta: Budget cost coefficient.
        lambda1: Retrieval loss weight.
        lambda2: Confidence loss weight.
        lambda3: Budget loss weight.
        max_grad_norm: Maximum gradient norm for clipping.
        divergence_threshold: Loss threshold for divergence detection.
        divergence_patience: Consecutive high-loss batches before halt.
        use_uncertainty_weighting: Use learned loss weights (Kendall et al., 2018).

    Example:
        >>> from pyrila import RILA, RILATrainer, rila_small
        >>> model = RILA(rila_small())
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        >>> trainer = RILATrainer(model, optimizer)
        >>> result = trainer.train_step({"input_ids": ..., "target_ids": ...})
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        alpha: float = 0.01,
        beta: float = 0.005,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        lambda3: float = 1.0,
        max_grad_norm: float = 1.0,
        divergence_threshold: float = 1e6,
        divergence_patience: int = 3,
        use_uncertainty_weighting: bool = False,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.alpha = alpha
        self.beta = beta
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.max_grad_norm = max_grad_norm
        self.divergence_threshold = divergence_threshold
        self.divergence_patience = divergence_patience
        self.use_uncertainty_weighting = use_uncertainty_weighting

        self.uncertainty_loss: Optional[UncertaintyWeightedLoss] = None
        if use_uncertainty_weighting:
            self.uncertainty_loss = UncertaintyWeightedLoss()
            # Move to same device as model
            device = next(model.parameters()).device
            self.uncertainty_loss = self.uncertainty_loss.to(device)
            self.optimizer.add_param_group({"params": self.uncertainty_loss.parameters()})

        self._consecutive_high_loss: int = 0
        self._high_loss_batches: List[int] = []
        self._high_loss_values: List[float] = []
        self._batch_count: int = 0
        self._clipping_active: bool = False

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Execute one training step.

        Args:
            batch: Dict with input_ids, target_ids, and optional attention_mask, reasoning_budget.

        Returns:
            Dict with loss, l_answer, l_retrieval, l_confidence, l_budget,
            batch_discarded, divergence_count.

        Raises:
            DivergenceError: If divergence detected.
        """
        self._batch_count += 1
        self.model.train()
        self.optimizer.zero_grad()

        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            target_ids=batch["target_ids"],
            reasoning_budget=batch.get("reasoning_budget"),
        )

        logits = outputs["logits"]
        confidence = outputs["confidence"]
        retrieval_stats = outputs["retrieval_stats"]
        budget_used = outputs["budget_used"]

        # Individual losses
        l_answer = compute_answer_loss(logits, batch["target_ids"])
        l_retrieval = compute_retrieval_loss(retrieval_stats["num_cells_retrieved"], self.alpha)
        l_retrieval = l_retrieval.to(l_answer.device)

        correctness = compute_answer_correctness(logits, batch["target_ids"])
        l_confidence = compute_confidence_loss(confidence.squeeze(-1), correctness.detach())
        l_budget = compute_budget_loss(budget_used, self.beta).to(l_answer.device)

        # Composite loss
        if self.use_uncertainty_weighting and self.uncertainty_loss is not None:
            loss = self.uncertainty_loss(l_answer, l_retrieval, l_confidence, l_budget)
        else:
            loss = compute_composite_loss(
                l_answer, l_retrieval, l_confidence, l_budget,
                self.lambda1, self.lambda2, self.lambda3,
            )

        loss_value = loss.item()
        self._check_divergence(loss_value)

        # Backward with gradient safety
        batch_discarded = self._safe_backward(loss)

        return {
            "loss": loss_value,
            "l_answer": l_answer.item(),
            "l_retrieval": l_retrieval.item(),
            "l_confidence": l_confidence.item(),
            "l_budget": l_budget.item(),
            "batch_discarded": batch_discarded,
            "divergence_count": self._consecutive_high_loss,
        }

    def train(
        self,
        dataset: List[Dict[str, torch.Tensor]],
        epochs: int = 1,
        log_every: int = 10,
    ) -> List[Dict[str, Any]]:
        """Full training loop over a dataset.

        Args:
            dataset: List of batch dicts.
            epochs: Number of epochs.
            log_every: Log interval in steps.

        Returns:
            List of step results.
        """
        all_results: List[Dict[str, Any]] = []
        for epoch in range(epochs):
            for i, batch in enumerate(dataset):
                result = self.train_step(batch)
                result["epoch"] = epoch
                result["step"] = i
                all_results.append(result)
        return all_results

    def evaluate(self, dataset: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """Evaluate model on dataset.

        Args:
            dataset: List of batch dicts.

        Returns:
            Dict with avg_loss and avg_confidence.
        """
        self.model.eval()
        total_loss = 0.0
        total_conf = 0.0
        count = 0

        with torch.no_grad():
            for batch in dataset:
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    target_ids=batch["target_ids"],
                )
                logits = outputs["logits"]
                l_answer = compute_answer_loss(logits, batch["target_ids"])
                total_loss += l_answer.item()
                total_conf += outputs["confidence"].mean().item()
                count += 1

        return {
            "avg_loss": total_loss / max(count, 1),
            "avg_confidence": total_conf / max(count, 1),
        }

    def save_checkpoint(self, path: str) -> None:
        """Save model and optimizer state.

        Args:
            path: File path for the checkpoint.

        Raises:
            CheckpointError: If saving fails.
        """
        try:
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "batch_count": self._batch_count,
            }
            if self.uncertainty_loss is not None:
                checkpoint["uncertainty_loss_state_dict"] = self.uncertainty_loss.state_dict()
            torch.save(checkpoint, path)
        except Exception as e:
            raise CheckpointError(path, "save", str(e)) from e

    def load_checkpoint(self, path: str) -> None:
        """Load model and optimizer state from checkpoint.

        Args:
            path: File path of the checkpoint.

        Raises:
            CheckpointError: If loading fails.
        """
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self._batch_count = checkpoint.get("batch_count", 0)
            if self.uncertainty_loss is not None and "uncertainty_loss_state_dict" in checkpoint:
                self.uncertainty_loss.load_state_dict(checkpoint["uncertainty_loss_state_dict"])
        except (CheckpointError,):
            raise
        except Exception as e:
            raise CheckpointError(path, "load", str(e)) from e

    def _check_divergence(self, loss_value: float) -> None:
        """Track consecutive high-loss batches."""
        if loss_value > self.divergence_threshold:
            self._consecutive_high_loss += 1
            self._high_loss_batches.append(self._batch_count)
            self._high_loss_values.append(loss_value)
            if self._consecutive_high_loss >= self.divergence_patience:
                raise DivergenceError(
                    self._high_loss_batches[-self.divergence_patience:],
                    self._high_loss_values[-self.divergence_patience:],
                    self.divergence_threshold,
                )
        else:
            self._consecutive_high_loss = 0
            self._high_loss_batches.clear()
            self._high_loss_values.clear()

    def _safe_backward(self, loss: torch.Tensor) -> bool:
        """Backward pass with NaN/Inf gradient detection."""
        loss.backward()

        if has_bad_gradients(self.model):
            self.optimizer.zero_grad()
            self._clipping_active = True
            return True

        if self._clipping_active:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        self.optimizer.step()
        return False
