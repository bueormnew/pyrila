"""Training utilities: gradient safety, divergence detection, etc."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from pyrila.exceptions import DivergenceError, GradientError


def has_bad_gradients(model: nn.Module) -> bool:
    """Check if any parameter has NaN or Inf gradients.

    Args:
        model: Model to check.

    Returns:
        True if bad gradients detected.
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                return True
    return False


def find_bad_gradient_param(model: nn.Module) -> Optional[str]:
    """Find the first parameter with NaN or Inf gradients.

    Args:
        model: Model to check.

    Returns:
        Name of the first parameter with bad gradients, or None.
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                return name
    return None


def compute_answer_correctness(
    logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    """Compute per-sample binary correctness.

    Args:
        logits: (batch, seq_len, vocab_size).
        targets: (batch, seq_len) target IDs.
        ignore_index: Token to ignore.

    Returns:
        (batch,) binary tensor.
    """
    predictions = logits.argmax(dim=-1)
    mask = targets != ignore_index
    matches = (predictions == targets) | ~mask
    return matches.all(dim=-1).float()
