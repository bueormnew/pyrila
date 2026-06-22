"""Recursive Verification Engine (RVE): multi-dimensional confidence evaluation.

Evaluates pre-output across four dimensions (logical, contextual, semantic,
consistency) and produces a final confidence score in [0, 1].
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from pyrila.config import RILAConfig


class RecursiveVerificationEngine(nn.Module):
    """Verify pre-output quality: c_t = V(query, WC, s_t, p_t).

    Four independent evaluator heads produce sub-scores in [0, 1].
    A learned aggregation network combines them into final confidence.

    Args:
        config: RILAConfig with cognitive_dim and confidence_threshold.
    """

    def __init__(self, config: RILAConfig) -> None:
        super().__init__()
        self.config = config
        self.threshold = config.confidence_threshold

        input_dim = config.cognitive_dim * 4
        H = config.cognitive_dim

        self.logical_evaluator = self._build_evaluator(input_dim, H)
        self.contextual_evaluator = self._build_evaluator(input_dim, H)
        self.semantic_evaluator = self._build_evaluator(input_dim, H)
        self.consistency_evaluator = self._build_evaluator(input_dim, H)

        self.confidence_aggregator = nn.Sequential(
            nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 1), nn.Sigmoid(),
        )

    @staticmethod
    def _build_evaluator(input_dim: int, hidden_dim: int) -> nn.Sequential:
        """Build a single evaluator head."""
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 4), nn.GELU(),
            nn.Linear(hidden_dim // 4, 1), nn.Sigmoid(),
        )

    def forward(
        self,
        query: torch.Tensor,
        working_context: torch.Tensor,
        cognitive_state: torch.Tensor,
        pre_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, bool]:
        """Evaluate pre-output confidence.

        Args:
            query: (batch, cognitive_dim).
            working_context: (batch, cognitive_dim).
            cognitive_state: (batch, cognitive_dim).
            pre_output: (batch, cognitive_dim).

        Returns:
            confidence: (batch, 1) in [0, 1].
            accept: True if confidence >= threshold for all batch items.
        """
        combined = torch.cat([query, working_context, cognitive_state, pre_output], dim=-1)

        logical = self.logical_evaluator(combined)
        contextual = self.contextual_evaluator(combined)
        semantic = self.semantic_evaluator(combined)
        consistency = self.consistency_evaluator(combined)

        sub_scores = torch.cat([logical, contextual, semantic, consistency], dim=-1)
        confidence = self.confidence_aggregator(sub_scores)

        accept = bool((confidence >= self.threshold).all().item())
        return confidence, accept
