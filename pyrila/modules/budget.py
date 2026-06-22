"""Reasoning Budget Controller: manages computational budget for reasoning loop.

Determines how many reasoning cycles the system may perform before
committing to the best available output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from pyrila.config import RILAConfig


@dataclass
class BudgetState:
    """Tracks reasoning budget consumption.

    Attributes:
        max_cycles: Maximum reasoning cycles allowed.
        current_cycle: Cycles completed so far.
        total_retrievals: Cumulative retrieval operations.
        best_confidence: Highest confidence observed.
        best_pre_output: Pre-output associated with best confidence.
    """

    max_cycles: int
    current_cycle: int = 0
    total_retrievals: int = 0
    best_confidence: float = 0.0
    best_pre_output: Optional[torch.Tensor] = None

    @property
    def exhausted(self) -> bool:
        """True when budget is exhausted."""
        return self.current_cycle >= self.max_cycles

    @property
    def remaining(self) -> int:
        """Remaining cycles."""
        return max(0, self.max_cycles - self.current_cycle)


class ReasoningBudgetController(nn.Module):
    """Controls reasoning budget allocation.

    Uses a learned predictor to estimate budget from cognitive state
    when no explicit budget is provided.

    Args:
        config: RILAConfig with budget parameters.
    """

    def __init__(self, config: RILAConfig) -> None:
        super().__init__()
        self.max_budget = config.max_budget
        self.default_budget = config.default_budget
        self.confidence_threshold = config.confidence_threshold

        self.budget_predictor = nn.Sequential(
            nn.Linear(config.cognitive_dim, config.cognitive_dim // 2),
            nn.GELU(),
            nn.Linear(config.cognitive_dim // 2, 1),
            nn.Sigmoid(),
        )

    def create_budget(
        self, budget: Optional[int] = None, cognitive_state: Optional[torch.Tensor] = None
    ) -> BudgetState:
        """Create a new BudgetState.

        Args:
            budget: Explicit budget (capped at max_budget).
            cognitive_state: Used to predict budget if none given.

        Returns:
            Fresh BudgetState.
        """
        if budget is not None:
            max_cycles = min(budget, self.max_budget)
        elif cognitive_state is not None:
            with torch.no_grad():
                fraction = self.budget_predictor(cognitive_state).mean().item()
            max_cycles = max(1, min(round(fraction * self.max_budget), self.max_budget))
        else:
            max_cycles = self.default_budget

        return BudgetState(max_cycles=max_cycles)

    def step(self, budget_state: BudgetState, confidence: float, pre_output: torch.Tensor) -> BudgetState:
        """Record completion of one reasoning cycle."""
        budget_state.current_cycle += 1
        if confidence > budget_state.best_confidence:
            budget_state.best_confidence = confidence
            budget_state.best_pre_output = pre_output
        return budget_state

    def should_continue(self, budget_state: BudgetState, confidence: float) -> bool:
        """Determine if reasoning should continue."""
        if budget_state.exhausted:
            return False
        if confidence >= self.confidence_threshold:
            return False
        return True
