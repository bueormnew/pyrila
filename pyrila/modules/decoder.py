"""Final Decoder: transforms cognitive state + encoder memory into tokens.

Autoregressive transformer decoder with cross-attention over the full
encoder memory (cell encodings). Token IDs (BOS, EOS, PAD) are read from
RILAConfig — the decoder does NOT hardcode any tokenizer-specific values.

Supports teacher forcing (training) and configurable sampling (inference)
including temperature, top-k, and top-p.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyrila.config import RILAConfig
from pyrila.exceptions import TeacherForcingError


class FinalDecoder(nn.Module):
    """Autoregressive transformer decoder with full encoder memory.

    The decoder receives:
    - encoder_memory: (batch, num_cells, cell_encoding_dim) — from RILA encoder
    - cognitive_state: (batch, cognitive_dim) — global context from CLP

    The cognitive_state is projected and prepended to the projected cell
    encodings, giving the decoder both global reasoning context AND
    fine-grained cell-level information for cross-attention.

    Token IDs (pad, bos, eos) are taken from config, NOT hardcoded.
    This makes the architecture tokenizer-agnostic.

    Args:
        config: RILAConfig with decoder parameters and token IDs.
    """

    def __init__(self, config: RILAConfig) -> None:
        super().__init__()
        self.config = config
        D = config.hidden_dim

        # Token IDs from config (set by user's tokenizer)
        self.pad_token_id = config.pad_token_id
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id

        # Project encoder memory to decoder dim
        # Accepts either token-level embeddings (hidden_dim) or cell encodings (4*hidden_dim)
        self.memory_projection = nn.Linear(config.hidden_dim, D)
        self.memory_norm = nn.LayerNorm(D)

        # Project cognitive_state to a context token prepended to memory
        self.cognitive_to_memory = nn.Linear(config.cognitive_dim, D)

        # Target token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, D)
        self.position_encoding = nn.Embedding(config.max_sequence_length, D)
        self.layer_norm = nn.LayerNorm(D)
        self.dropout = nn.Dropout(config.dropout)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=D,
            nhead=config.num_heads,
            dim_feedforward=D * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.decoder_layers = nn.TransformerDecoder(
            decoder_layer, num_layers=max(1, config.num_layers // 2)
        )

        # Output projection with weight tying
        self.output_projection = nn.Linear(D, config.vocab_size, bias=False)
        self.output_projection.weight = self.token_embedding.weight

    def _build_memory(
        self,
        cognitive_state: torch.Tensor,
        encoder_memory: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build decoder memory from cognitive state + token-level encoder output.

        Returns (batch, 1+seq_len, hidden_dim) when encoder_memory is provided.
        The cognitive_state is prepended as a global reasoning summary token.
        """
        cog_token = self.cognitive_to_memory(cognitive_state).unsqueeze(1)  # (B, 1, D)

        if encoder_memory is not None:
            projected = self.memory_norm(self.memory_projection(encoder_memory))
            memory = torch.cat([cog_token, projected], dim=1)  # (B, 1+seq_len, D)
        else:
            memory = cog_token

        return memory

    def _embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embed token IDs with positional encoding."""
        seq_len = token_ids.shape[1]
        positions = torch.arange(seq_len, device=token_ids.device)
        emb = self.token_embedding(token_ids) + self.position_encoding(positions)
        return self.dropout(self.layer_norm(emb))

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask (upper triangular)."""
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1
        )

    def forward(
        self,
        cognitive_state: torch.Tensor,
        target_tokens: Optional[torch.Tensor] = None,
        encoder_memory: Optional[torch.Tensor] = None,
        max_length: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Decode using cognitive state + encoder memory.

        Training (target_tokens provided): returns logits (batch, target_len, vocab_size).
        Inference (target_tokens=None): returns generated IDs (batch, gen_len).
        """
        if self.training and target_tokens is None:
            raise TeacherForcingError()

        memory = self._build_memory(cognitive_state, encoder_memory)

        if target_tokens is not None:
            return self._forward_training(memory, target_tokens)
        return self._forward_inference(memory, max_length, temperature, top_k, top_p)

    def _forward_training(self, memory: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
        """Training with teacher forcing."""
        tgt = self._embed_tokens(target_tokens)
        tgt_mask = self._generate_causal_mask(target_tokens.shape[1], target_tokens.device)
        decoded = self.decoder_layers(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
        return self.output_projection(decoded)

    def _forward_inference(
        self, memory: torch.Tensor, max_length: int,
        temperature: float, top_k: int, top_p: float,
    ) -> torch.Tensor:
        """Autoregressive generation with cross-attention over encoder memory."""
        batch_size = memory.shape[0]
        device = memory.device

        generated = torch.full(
            (batch_size, 1), self.bos_token_id, dtype=torch.long, device=device
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length - 1):
            tgt = self._embed_tokens(generated)
            tgt_mask = self._generate_causal_mask(generated.shape[1], device)
            decoded = self.decoder_layers(tgt=tgt, memory=memory, tgt_mask=tgt_mask)

            last_logits = self.output_projection(decoded[:, -1, :]) / temperature

            # Block special tokens from being generated
            last_logits[:, self.pad_token_id] = float("-inf")
            last_logits[:, self.bos_token_id] = float("-inf")

            last_logits = self._top_k_filter(last_logits, top_k)
            last_logits = self._top_p_filter(last_logits, top_p)

            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_token = next_token.masked_fill(finished.unsqueeze(1), self.eos_token_id)

            generated = torch.cat([generated, next_token], dim=1)
            finished = finished | (next_token.squeeze(1) == self.eos_token_id)

            if finished.all():
                break

        return generated

    def _top_k_filter(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Keep only top-k logits."""
        if top_k >= logits.shape[-1]:
            return logits
        values, _ = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1)
        return logits.masked_fill(logits < values[:, -1:], float("-inf"))

    def _top_p_filter(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Nucleus sampling filter."""
        if top_p >= 1.0:
            return logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        return logits.masked_fill(indices_to_remove, float("-inf"))

    def generate(
        self,
        cognitive_state: torch.Tensor,
        encoder_memory: Optional[torch.Tensor] = None,
        max_length: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Convenience method for inference generation."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            result = self.forward(
                cognitive_state,
                target_tokens=None,
                encoder_memory=encoder_memory,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        if was_training:
            self.train()
        return result
