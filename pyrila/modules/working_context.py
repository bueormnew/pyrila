"""Working Context: aggregates retrieved cells into a cognitive representation.

Uses multi-head attention weighted by relevance scores to produce a single
(batch, cognitive_dim) tensor from the set of retrieved Context Cells.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from pyrila.config import RILAConfig


class WorkingContext(nn.Module):
    """Aggregates retrieved cell encodings into a cognitive-dim representation.

    Within a reasoning session the working context only grows (monotonic).
    Cells are added but never removed until reset().

    Args:
        config: RILAConfig with cell_encoding_dim, cognitive_dim, num_heads.
    """

    def __init__(self, config: RILAConfig) -> None:
        super().__init__()
        self.config = config

        self.context_aggregator = nn.MultiheadAttention(
            embed_dim=config.cell_encoding_dim,
            num_heads=config.num_heads,
            dropout=config.attention_dropout,
            batch_first=True,
        )

        self.output_projection = nn.Linear(config.cell_encoding_dim, config.cognitive_dim)

        self._stored_encodings: Optional[torch.Tensor] = None
        self._stored_scores: Optional[torch.Tensor] = None

    def forward(
        self, cell_encodings: torch.Tensor, relevance_scores: torch.Tensor
    ) -> torch.Tensor:
        """Aggregate retrieved cells.

        Args:
            cell_encodings: (batch, num_retrieved, cell_encoding_dim).
            relevance_scores: (batch, num_retrieved) in [0, 1].

        Returns:
            (batch, cognitive_dim) aggregated representation.
        """
        self._update_stored(cell_encodings, relevance_scores)
        return self._aggregate(cell_encodings, relevance_scores)

    def reset(self) -> None:
        """Clear stored cells for a new session."""
        self._stored_encodings = None
        self._stored_scores = None

    @property
    def num_stored_cells(self) -> int:
        """Number of cells currently stored."""
        return 0 if self._stored_encodings is None else self._stored_encodings.shape[1]

    def _update_stored(self, cell_encodings: torch.Tensor, relevance_scores: torch.Tensor) -> None:
        """Monotonically grow stored state."""
        if self._stored_encodings is None:
            self._stored_encodings = cell_encodings.detach()
            self._stored_scores = relevance_scores.detach()
        elif cell_encodings.shape[1] > self._stored_encodings.shape[1]:
            self._stored_encodings = cell_encodings.detach()
            self._stored_scores = relevance_scores.detach()

    def _aggregate(
        self, cell_encodings: torch.Tensor, relevance_scores: torch.Tensor
    ) -> torch.Tensor:
        """Core aggregation via relevance-weighted attention."""
        batch_size, num_cells, _ = cell_encodings.shape

        weights = relevance_scores.unsqueeze(-1)
        weight_sum = weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        normalized_weights = weights / weight_sum
        query = (cell_encodings * normalized_weights).sum(dim=1, keepdim=True)

        attn_bias = torch.log(relevance_scores.clamp(min=1e-8))
        attn_bias = attn_bias.unsqueeze(1).repeat(self.config.num_heads, 1, 1)

        attended, _ = self.context_aggregator(
            query=query, key=cell_encodings, value=cell_encodings, attn_mask=attn_bias
        )

        attended = attended.squeeze(1)
        return self.output_projection(attended)
