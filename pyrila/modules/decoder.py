"""Final Decoder: transforms cognitive state into natural language tokens.

Autoregressive transformer decoder with teacher forcing (training) and
configurable sampling (inference) including temperature, top-k, and top-p.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyrila.config import RILAConfig
from pyrila.exceptions import TeacherForcingError

EOS_TOKEN_ID = 0
SOS_TOKEN_ID = 0


class FinalDecoder(nn.Module):
    """Autoregressive transformer decoder for text generation.

    Args:
        config: RILAConfig with decoder parameters.
    """

    def __init__(self, config: RILAConfig) -> None:
        super().__init__()
        self.config = config

        self.state_to_decoder = nn.Linear(config.cognitive_dim, config.hidden_dim)
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_encoding = nn.Embedding(config.max_sequence_length, config.hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.decoder_layers = nn.TransformerDecoder(
            decoder_layer, num_layers=max(1, config.num_layers // 2)
        )

        self.output_projection = nn.Linear(config.hidden_dim, config.vocab_size)
        self.layer_norm = nn.LayerNorm(config.hidden_dim)

    def _embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embed token IDs with positional encoding."""
        seq_len = token_ids.shape[1]
        positions = torch.arange(seq_len, device=token_ids.device)
        return self.layer_norm(self.token_embedding(token_ids) + self.position_encoding(positions))

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)

    def forward(
        self,
        cognitive_state: torch.Tensor,
        target_tokens: torch.Tensor | None = None,
        max_length: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Decode cognitive state into tokens.

        Training (target_tokens provided): returns logits (batch, target_len, vocab_size).
        Inference (target_tokens=None): returns generated IDs (batch, gen_len).
        """
        if self.training and target_tokens is None:
            raise TeacherForcingError()

        if target_tokens is not None:
            return self._forward_training(cognitive_state, target_tokens)
        return self._forward_inference(cognitive_state, max_length, temperature, top_k, top_p)

    def _forward_training(self, cognitive_state: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
        """Training with teacher forcing."""
        memory = self.state_to_decoder(cognitive_state).unsqueeze(1)
        tgt = self._embed_tokens(target_tokens)
        tgt_mask = self._generate_causal_mask(target_tokens.shape[1], target_tokens.device)
        decoded = self.decoder_layers(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
        return self.output_projection(decoded)

    def _forward_inference(
        self, cognitive_state: torch.Tensor, max_length: int,
        temperature: float, top_k: int, top_p: float
    ) -> torch.Tensor:
        """Autoregressive generation."""
        batch_size = cognitive_state.shape[0]
        device = cognitive_state.device
        memory = self.state_to_decoder(cognitive_state).unsqueeze(1)

        generated = torch.full((batch_size, 1), SOS_TOKEN_ID, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length - 1):
            tgt = self._embed_tokens(generated)
            tgt_mask = self._generate_causal_mask(generated.shape[1], device)
            decoded = self.decoder_layers(tgt=tgt, memory=memory, tgt_mask=tgt_mask)

            last_logits = self.output_projection(decoded[:, -1, :]) / temperature
            last_logits = self._top_k_filter(last_logits, top_k)
            last_logits = self._top_p_filter(last_logits, top_p)

            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_token = next_token.clamp(0, self.config.vocab_size - 1)
            next_token = next_token.masked_fill(finished.unsqueeze(1), EOS_TOKEN_ID)

            generated = torch.cat([generated, next_token], dim=1)
            finished = finished | (next_token.squeeze(1) == EOS_TOKEN_ID)

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
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        return logits.masked_fill(indices_to_remove, float("-inf"))

    def generate(
        self, cognitive_state: torch.Tensor, max_length: int = 512,
        temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9,
    ) -> torch.Tensor:
        """Convenience method for inference generation."""
        was_training = self.training
        self.eval()
        with torch.no_grad():
            result = self.forward(
                cognitive_state, target_tokens=None,
                max_length=max_length, temperature=temperature, top_k=top_k, top_p=top_p,
            )
        if was_training:
            self.train()
        return result
