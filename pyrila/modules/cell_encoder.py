"""Cell Encoder: transforms Context Cells into four specialized vectors.

CCE_i = concat(S_i, R_i, T_i, I_i) ∈ R^(4D)
- S: Semantic vector
- R: Relation vector
- T: Structural vector
- I: Importance vector
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyrila.config import RILAConfig
from pyrila.modules.cell_builder import ContextCellData


class CellEncoder(nn.Module):
    """Produce 4D encoded representation per Context Cell.

    Architecture:
        1. Shared 2-layer TransformerEncoder contextualizes tokens
        2. Mean pooling over non-padded positions
        3. Four projection heads produce S, R, T, I vectors (L2-normalized)
        4. Concatenation → CCE

    Args:
        config: RILAConfig with hidden_dim, num_heads, dropout.
    """

    def __init__(self, config: RILAConfig) -> None:
        super().__init__()
        self.config = config
        D = config.hidden_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=config.num_heads,
            dim_feedforward=D * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.cell_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.semantic_head = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
        self.relation_head = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
        self.structural_head = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
        self.importance_head = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))

    def forward(self, cell_tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Encode batch of cells into CCE vectors.

        Args:
            cell_tokens: (batch, cell_size, hidden_dim).
            lengths: (batch,) actual token counts per cell.

        Returns:
            CCE tensor of shape (batch, 4 * hidden_dim).
        """
        batch_size, cell_size, D = cell_tokens.shape

        positions = torch.arange(cell_size, device=cell_tokens.device).unsqueeze(0)
        padding_mask = positions >= lengths.unsqueeze(1)

        all_padded = lengths == 0
        if all_padded.any():
            padding_mask = padding_mask.clone()
            padding_mask[all_padded] = False

        contextualized = self.cell_transformer(cell_tokens, src_key_padding_mask=padding_mask)
        pooled = self._mean_pool(contextualized, lengths)

        S = F.normalize(self.semantic_head(pooled), p=2, dim=-1)
        R = F.normalize(self.relation_head(pooled), p=2, dim=-1)
        T = F.normalize(self.structural_head(pooled), p=2, dim=-1)
        I = F.normalize(self.importance_head(pooled), p=2, dim=-1)

        return torch.cat([S, R, T, I], dim=-1)

    def _mean_pool(self, contextualized: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Mean pool over non-padded positions."""
        batch_size, cell_size, D = contextualized.shape
        positions = torch.arange(cell_size, device=contextualized.device).unsqueeze(0)
        valid_mask = (positions < lengths.unsqueeze(1)).float()

        all_padded = lengths == 0
        if all_padded.any():
            valid_mask[all_padded] = 1.0

        mask_expanded = valid_mask.unsqueeze(-1)
        summed = (contextualized * mask_expanded).sum(dim=1)
        counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return summed / counts

    def encode_batch(self, cells: List[ContextCellData]) -> torch.Tensor:
        """Encode multiple ContextCellData objects in parallel.

        Args:
            cells: List of ContextCellData to encode.

        Returns:
            (num_cells, 4 * hidden_dim) CCE tensor.
        """
        if not cells:
            return torch.zeros(0, self.config.cell_encoding_dim)

        cell_tokens = torch.stack([c.tokens for c in cells], dim=0)
        lengths = torch.tensor(
            [c.length for c in cells], dtype=torch.long, device=cell_tokens.device
        )
        return self.forward(cell_tokens, lengths)

    def encode_batch_with_tokens(self, cells: List[ContextCellData]) -> tuple:
        """Encode cells AND return token-level contextualized embeddings.

        This gives the decoder access to fine-grained token representations
        while the rest of the RILA pipeline uses the pooled cell encodings.

        Args:
            cells: List of ContextCellData to encode.

        Returns:
            Tuple of:
                cell_encodings: (num_cells, 4 * hidden_dim) pooled CCE vectors
                token_embeddings: (num_cells, cell_size, hidden_dim) contextualized tokens
        """
        if not cells:
            return (
                torch.zeros(0, self.config.cell_encoding_dim),
                torch.zeros(0, self.config.cell_size, self.config.hidden_dim),
            )

        cell_tokens = torch.stack([c.tokens for c in cells], dim=0)
        lengths = torch.tensor(
            [c.length for c in cells], dtype=torch.long, device=cell_tokens.device
        )

        batch_size, cell_size, D = cell_tokens.shape
        positions = torch.arange(cell_size, device=cell_tokens.device).unsqueeze(0)
        padding_mask = positions >= lengths.unsqueeze(1)

        all_padded = lengths == 0
        if all_padded.any():
            padding_mask = padding_mask.clone()
            padding_mask[all_padded] = False

        # Contextualize tokens (this is the key token-level representation)
        contextualized = self.cell_transformer(cell_tokens, src_key_padding_mask=padding_mask)

        # Pooled cell encodings (for RILA pipeline: index, retrieval, CLP)
        pooled = self._mean_pool(contextualized, lengths)
        S = F.normalize(self.semantic_head(pooled), p=2, dim=-1)
        R = F.normalize(self.relation_head(pooled), p=2, dim=-1)
        T = F.normalize(self.structural_head(pooled), p=2, dim=-1)
        I = F.normalize(self.importance_head(pooled), p=2, dim=-1)
        cell_encodings = torch.cat([S, R, T, I], dim=-1)

        return cell_encodings, contextualized
