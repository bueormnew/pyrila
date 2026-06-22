"""Relevance Gate: filters Knowledge Units by learned relevance score.

Computes r_i = σ(f(KU_i)) ∈ [0, 1] and applies threshold-based gating
with a fallback guaranteeing at least one unit passes per batch element.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from pyrila.config import RILAConfig


class RelevanceGate(nn.Module):
    """Gate Knowledge Units based on learned relevance.

    Args:
        config: RILAConfig with knowledge_dim and relevance_gate_threshold.
    """

    def __init__(self, config: RILAConfig) -> None:
        super().__init__()
        self.config = config
        self.gate_network = nn.Sequential(
            nn.Linear(config.knowledge_dim, config.knowledge_dim // 2),
            nn.GELU(),
            nn.Linear(config.knowledge_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.threshold = config.relevance_gate_threshold

    def forward(self, knowledge_units: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply relevance gating.

        Args:
            knowledge_units: (batch, num_units, knowledge_dim).

        Returns:
            gated_units: (batch, num_units, knowledge_dim) scaled by relevance.
            gate_scores: (batch, num_units) relevance scores.
        """
        gate_scores = self.gate_network(knowledge_units).squeeze(-1)
        pass_mask = gate_scores >= self.threshold

        # Fallback: if all below threshold, pass highest-scoring unit
        all_below = ~pass_mask.any(dim=1)
        if all_below.any():
            max_indices = gate_scores.argmax(dim=1)
            batch_indices = torch.where(all_below)[0]
            pass_mask[batch_indices, max_indices[all_below]] = True

        scale = torch.where(
            pass_mask.unsqueeze(-1),
            gate_scores.unsqueeze(-1),
            torch.zeros_like(gate_scores.unsqueeze(-1)),
        )
        return knowledge_units * scale, gate_scores
