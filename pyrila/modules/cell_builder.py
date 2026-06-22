"""Context Cell Builder: partitions token sequences into fixed-size cells.

Each Context Cell is a contiguous block of tokens with positional metadata
and structural references to adjacent cells.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from pyrila.config import RILAConfig


@dataclass
class ContextCellData:
    """A fixed-size block of token embeddings with metadata.

    Attributes:
        tokens: Token embeddings of shape (cell_size, hidden_dim).
        position: Zero-based global cell index.
        length: Actual (unpadded) token count.
        prev_ref: Index of previous adjacent cell, or None.
        next_ref: Index of next adjacent cell, or None.
        metadata: Additional internal metadata.
    """

    tokens: torch.Tensor
    position: int
    length: int
    prev_ref: Optional[int] = None
    next_ref: Optional[int] = None
    metadata: Dict[str, object] = field(default_factory=dict)


class ContextCellBuilder(nn.Module):
    """Partition token sequences into fixed-size Context Cells.

    Args:
        config: RILAConfig with cell_size and hidden_dim.
    """

    def __init__(self, config: RILAConfig) -> None:
        super().__init__()
        self.config = config
        self.cell_size = config.cell_size

    def forward(
        self, embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> List[List[ContextCellData]]:
        """Partition embeddings into Context Cells.

        Args:
            embeddings: (batch_size, seq_len, hidden_dim).
            attention_mask: (batch_size, seq_len), 1=valid, 0=padding.

        Returns:
            List of lists of ContextCellData, one per batch element.
        """
        batch_size, seq_len, hidden_dim = embeddings.shape
        num_cells = math.ceil(seq_len / self.cell_size)
        batch_cells: List[List[ContextCellData]] = []

        for b in range(batch_size):
            cells: List[ContextCellData] = []
            for i in range(num_cells):
                start = i * self.cell_size
                end = min(start + self.cell_size, seq_len)
                actual_length = end - start
                cell_tokens = embeddings[b, start:end, :]

                if actual_length < self.cell_size:
                    padding = torch.zeros(
                        self.cell_size - actual_length, hidden_dim,
                        dtype=cell_tokens.dtype, device=cell_tokens.device,
                    )
                    cell_tokens = torch.cat([cell_tokens, padding], dim=0)

                cells.append(ContextCellData(
                    tokens=cell_tokens,
                    position=i,
                    length=actual_length,
                    metadata={"actual_length": actual_length},
                ))

            # Assign structural references
            self._assign_refs(cells, attention_mask[b], seq_len)
            batch_cells.append(cells)

        return batch_cells

    def _assign_refs(
        self, cells: List[ContextCellData], mask: torch.Tensor, seq_len: int
    ) -> None:
        """Link adjacent cells if no mask gap separates them."""
        for i in range(len(cells) - 1):
            start_i = i * self.cell_size
            end_next = min((i + 2) * self.cell_size, seq_len)
            if torch.all(mask[start_i:end_next] == 1):
                cells[i].next_ref = i + 1
                cells[i + 1].prev_ref = i
