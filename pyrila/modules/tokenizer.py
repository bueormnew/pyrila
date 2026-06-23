"""RILA Tokenizer: token IDs → embeddings.

This module converts token IDs into embeddings. It is the EMBEDDING layer
of the architecture, not a text tokenizer. The user is responsible for
converting text to token IDs using their own tokenizer (BPE, SentencePiece,
tiktoken, char-level, etc.) before passing to the model.

The model accepts any integer tensor as input_ids — it does not care how
those IDs were produced.

Includes an optional character-level text→IDs utility for simple use cases.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from pyrila.config import RILAConfig
from pyrila.exceptions import SequenceTooLongError


class RILATokenizer(nn.Module):
    """Token embedding layer: converts token IDs to dense embeddings.

    This is an EMBEDDING module, not a text tokenizer. It takes integer
    token IDs (from any external tokenizer) and produces embeddings.

    Optionally provides a simple char-level tokenize() method for testing.

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
        self._vocab: Optional[Dict[str, int]] = None  # Lazy-built for tokenize()

    def _build_vocab(self) -> Dict[str, int]:
        """Build character-level vocabulary mapping (for tokenize() utility only)."""
        vocab: Dict[str, int] = {}
        next_id = 4  # 0=PAD, 1=UNK, 2=BOS, 3=EOS (reserved)
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
        """Convert text string to token IDs tensor (simple char-level utility).

        This is a convenience method for testing. In production, use your
        own tokenizer (BPE, SentencePiece, etc.) and pass IDs directly.

        Args:
            text: Raw input text.

        Returns:
            Token IDs of shape (1, seq_len).
        """
        if self._vocab is None:
            self._vocab = self._build_vocab()

        bos_id = self.config.bos_token_id
        unk_id = 1  # UNK

        if not text:
            return torch.tensor([[bos_id]], dtype=torch.long)
        token_ids = [bos_id] + [
            self._vocab.get(c, unk_id) for c in text
        ]
        if len(token_ids) > self.config.max_sequence_length:
            raise SequenceTooLongError(len(token_ids), self.config.max_sequence_length)
        return torch.tensor([token_ids], dtype=torch.long)
