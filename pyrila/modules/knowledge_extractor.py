"""Knowledge Extractor: extracts Knowledge Units from Context Cells.

Uses learned cross-attention queries to extract structured knowledge from
cell token embeddings. Each cell produces max_knowledge_units_per_cell
KU slots with activation scores.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from pyrila.config import RILAConfig


class KnowledgeExtractor(nn.Module):
    """Extract Knowledge Units from Context Cells via cross-attention.

    Architecture:
        1. Project cell tokens from hidden_dim to knowledge_dim
        2. Learned queries attend over projected tokens
        3. Activation predictor assigns scores in [0, 1] per slot

    Args:
        config: RILAConfig with knowledge extraction parameters.
    """

    def __init__(self, config: RILAConfig) -> None:
        super().__init__()
        self.config = config

        self.extraction_queries = nn.Parameter(
            torch.randn(config.max_knowledge_units_per_cell, config.knowledge_dim)
        )

        self.extraction_attention = nn.MultiheadAttention(
            embed_dim=config.knowledge_dim,
            num_heads=max(1, config.num_heads // 2),
            dropout=config.dropout,
            batch_first=True,
        )

        self.content_projection = nn.Linear(config.hidden_dim, config.knowledge_dim)

        self.activation_predictor = nn.Sequential(
            nn.Linear(config.knowledge_dim, config.knowledge_dim // 2),
            nn.GELU(),
            nn.Linear(config.knowledge_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, cell_tokens: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract Knowledge Units from cell tokens.

        Args:
            cell_tokens: (batch, cell_size, hidden_dim).
            lengths: (batch,) actual token counts.

        Returns:
            knowledge_units: (batch, max_ku_per_cell, knowledge_dim).
            activations: (batch, max_ku_per_cell) in [0, 1].
        """
        batch_size, cell_size, _ = cell_tokens.shape
        device = cell_tokens.device

        projected = self.content_projection(cell_tokens)

        positions = torch.arange(cell_size, device=device).unsqueeze(0)
        key_padding_mask = positions >= lengths.unsqueeze(1)

        all_padded = lengths == 0
        if all_padded.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_padded] = False

        queries = self.extraction_queries.unsqueeze(0).expand(batch_size, -1, -1)

        knowledge_units, _ = self.extraction_attention(
            query=queries, key=projected, value=projected,
            key_padding_mask=key_padding_mask,
        )

        activations = self.activation_predictor(knowledge_units).squeeze(-1)

        if all_padded.any():
            activations = activations.clone()
            activations[all_padded] = 0.0

        return knowledge_units, activations
