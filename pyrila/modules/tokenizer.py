"""RILA Tokenizer: text → token IDs → embeddings.

Converts raw text into token IDs using a character-level vocabulary,
then produces embeddings via token embedding + positional encoding +
layer norm + dropout.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from pyrila.config import RILAConfig
from pyrila.exceptions import SequenceTooLongError

SOS_TOKEN_ID = 0
UNK_TOKEN_ID = 1


class RILATokenizer(nn.Module):
    """Token embedding module with character-level vocabulary.

    Args:
        config: RILAConfig instance.
    """

    def __init__(self, config: RILAConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_encoding = nn.Embedding(config.max_sequence_length, config.hidden_dim)
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        self._vocab: Dict[str, int] = self._build_vocab()

    def _build_vocab(self) -> Dict[str, int]:
        """Build character-level vocabulary mapping."""
        vocab: Dict[str, int] = {}
        next_id = 2  # 0=SOS, 1=UNK
        for code in range(32, 127):
            if next_id >= self.config.vocab_size:
                break
            vocab[chr(code)] = next_id
            next_id += 1
        return vocab

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Convert token IDs to embeddings.

        Args:
            token_ids: (batch_size, seq_len) with values in [0, vocab_size).

        Returns:
            Embeddings of shape (batch_size, seq_len, hidden_dim).
        """
        seq_len = token_ids.shape[1]
        if seq_len > self.config.max_sequence_length:
            raise SequenceTooLongError(seq_len, self.config.max_sequence_length)
        positions = torch.arange(seq_len, device=token_ids.device)
        embeddings = self.embedding(token_ids) + self.position_encoding(positions)
        return self.dropout(self.layer_norm(embeddings))

    def tokenize(self, text: str) -> torch.Tensor:
        """Convert text string to token IDs tensor.

        Args:
            text: Raw input text.

        Returns:
            Token IDs of shape (1, seq_len).
        """
        if not text:
            return torch.tensor([[SOS_TOKEN_ID]], dtype=torch.long)
        token_ids = [SOS_TOKEN_ID] + [
            self._vocab.get(c, UNK_TOKEN_ID) for c in text
        ]
        if len(token_ids) > self.config.max_sequence_length:
            raise SequenceTooLongError(len(token_ids), self.config.max_sequence_length)
        return torch.tensor([token_ids], dtype=torch.long)
