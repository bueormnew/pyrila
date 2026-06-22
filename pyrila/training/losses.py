"""RILA loss functions.

Implements the composite loss L = L_answer + λ₁·L_retrieval + λ₂·L_confidence + λ₃·L_budget
and the uncertainty-weighted variant (Kendall et al., 2018) that learns optimal
loss weights automatically.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyWeightedLoss(nn.Module):
    """Learns optimal loss weights via homoscedastic uncertainty.

    Instead of fixed lambdas, learns log(σ²) per auxiliary task.
    Loss becomes: L = L_answer + Σ (1/2σ²_i)·L_i + (1/2)·log(σ²_i)

    This automatically balances loss magnitudes during training.

    Reference: Kendall et al., "Multi-Task Learning Using Uncertainty
    to Weigh Losses for Scene Geometry and Semantics", CVPR 2018.
    """

    def __init__(self) -> None:
        super().__init__()
        # Initialize log_var to 0 → initial σ²=1 → initial precision=1
        self.log_var_retrieval = nn.Parameter(torch.zeros(1))
        self.log_var_confidence = nn.Parameter(torch.zeros(1))
        self.log_var_budget = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        l_answer: torch.Tensor,
        l_retrieval: torch.Tensor,
        l_confidence: torch.Tensor,
        l_budget: torch.Tensor,
    ) -> torch.Tensor:
        """Compute uncertainty-weighted composite loss.

        L = L_answer + (1/2σ²_r)·L_ret + (1/2)·log(σ²_r)
                     + (1/2σ²_c)·L_conf + (1/2)·log(σ²_c)
                     + (1/2σ²_b)·L_budget + (1/2)·log(σ²_b)

        Args:
            l_answer: Scalar answer loss.
            l_retrieval: Scalar retrieval loss.
            l_confidence: Scalar confidence loss.
            l_budget: Scalar budget loss.

        Returns:
            Scalar composite loss with learned weighting.
        """
        precision_r = torch.exp(-self.log_var_retrieval)
        precision_c = torch.exp(-self.log_var_confidence)
        precision_b = torch.exp(-self.log_var_budget)

        loss = l_answer
        loss = loss + precision_r * l_retrieval + 0.5 * self.log_var_retrieval
        loss = loss + precision_c * l_confidence + 0.5 * self.log_var_confidence
        loss = loss + precision_b * l_budget + 0.5 * self.log_var_budget

        return loss.squeeze()

    @property
    def effective_weights(self) -> dict:
        """Current effective weights (precision = 1/σ² for each task)."""
        with torch.no_grad():
            return {
                "retrieval": torch.exp(-self.log_var_retrieval).item(),
                "confidence": torch.exp(-self.log_var_confidence).item(),
                "budget": torch.exp(-self.log_var_budget).item(),
            }


def compute_answer_loss(
    logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    """Cross-entropy loss between predicted logits and target tokens.

    Args:
        logits: (batch, seq_len, vocab_size).
        targets: (batch, seq_len) target token IDs.
        ignore_index: Token ID to ignore (padding).

    Returns:
        Scalar cross-entropy loss.
    """
    vocab_size = logits.shape[-1]
    return F.cross_entropy(
        logits.reshape(-1, vocab_size), targets.reshape(-1), ignore_index=ignore_index
    )


def compute_retrieval_loss(num_cells_retrieved: int, alpha: float = 0.01) -> torch.Tensor:
    """Retrieval cost: α * R (cells retrieved).

    Args:
        num_cells_retrieved: Total cells retrieved.
        alpha: Cost coefficient in [0.0001, 1.0].

    Returns:
        Scalar retrieval loss.
    """
    return torch.tensor(alpha * num_cells_retrieved, dtype=torch.float32)


def compute_confidence_loss(
    predicted_confidence: torch.Tensor, correctness: torch.Tensor
) -> torch.Tensor:
    """BCE between predicted confidence and correctness.

    Args:
        predicted_confidence: (batch,) or (batch, 1) in [0, 1].
        correctness: (batch,) binary labels.

    Returns:
        Scalar BCE loss.
    """
    pred = predicted_confidence.reshape(-1).clamp(1e-7, 1.0 - 1e-7)
    target = correctness.reshape(-1).float()
    return F.binary_cross_entropy(pred, target)


def compute_budget_loss(budget_used: int, beta: float = 0.005) -> torch.Tensor:
    """Budget cost: β * B_used (reasoning cycles consumed).

    Args:
        budget_used: Number of cycles consumed.
        beta: Cost coefficient in [0.0001, 1.0].

    Returns:
        Scalar budget loss.
    """
    return torch.tensor(beta * budget_used, dtype=torch.float32)


def compute_composite_loss(
    l_answer: torch.Tensor,
    l_retrieval: torch.Tensor,
    l_confidence: torch.Tensor,
    l_budget: torch.Tensor,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    lambda3: float = 1.0,
) -> torch.Tensor:
    """Fixed-weight composite loss.

    L = L_answer + λ₁·L_retrieval + λ₂·L_confidence + λ₃·L_budget

    Args:
        l_answer, l_retrieval, l_confidence, l_budget: Individual loss terms.
        lambda1, lambda2, lambda3: Fixed weights.

    Returns:
        Scalar composite loss.
    """
    return l_answer + lambda1 * l_retrieval + lambda2 * l_confidence + lambda3 * l_budget
