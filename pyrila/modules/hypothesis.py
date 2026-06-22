"""Hypothesis Generator and Evidence Tracker.

Produces candidate response hypotheses from cognitive state, scores them
on four criteria, and tracks supporting Knowledge Units.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from pyrila.config import RILAConfig


class HypothesisGenerator(nn.Module):
    """Generate and score response hypotheses: H_t = Γ(s_t).

    Args:
        config: RILAConfig with hypothesis parameters.
    """

    def __init__(self, config: RILAConfig) -> None:
        super().__init__()
        self.config = config
        H = config.cognitive_dim
        max_hyp = config.max_hypotheses

        self.hypothesis_projector = nn.Sequential(nn.Linear(H, H * max_hyp), nn.GELU())

        self.coherence_scorer = nn.Sequential(
            nn.Linear(H * 2, H), nn.GELU(), nn.Linear(H, 1), nn.Sigmoid(),
        )
        self.evidence_scorer = nn.Sequential(
            nn.Linear(H * 2, H), nn.GELU(), nn.Linear(H, 1), nn.Sigmoid(),
        )
        self.query_compat_scorer = nn.Sequential(
            nn.Linear(H * 2, H), nn.GELU(), nn.Linear(H, 1), nn.Sigmoid(),
        )
        self.consistency_scorer = nn.Sequential(
            nn.Linear(H * 2, H), nn.GELU(), nn.Linear(H, 1), nn.Sigmoid(),
        )

    def forward(
        self,
        cognitive_state: torch.Tensor,
        query: Optional[torch.Tensor] = None,
        evidence: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate hypotheses and select winner.

        Args:
            cognitive_state: (batch, cognitive_dim).
            query: Optional query context.
            evidence: Optional evidence summary.

        Returns:
            winner: (batch, cognitive_dim).
            all_hypotheses: (batch, max_hyp, cognitive_dim).
            scores: (batch, max_hyp, 4).
        """
        batch_size = cognitive_state.shape[0]
        H = self.config.cognitive_dim
        max_hyp = self.config.max_hypotheses

        projected = self.hypothesis_projector(cognitive_state)
        all_hypotheses = projected.view(batch_size, max_hyp, H)

        ctx = query if query is not None else cognitive_state
        ctx_expanded = ctx.unsqueeze(1).expand(-1, max_hyp, -1)
        hyp_ctx = torch.cat([all_hypotheses, ctx_expanded], dim=-1)
        hyp_ctx_flat = hyp_ctx.view(batch_size * max_hyp, H * 2)

        coherence = self.coherence_scorer(hyp_ctx_flat).view(batch_size, max_hyp, 1)
        evidence_score = self.evidence_scorer(hyp_ctx_flat).view(batch_size, max_hyp, 1)
        query_compat = self.query_compat_scorer(hyp_ctx_flat).view(batch_size, max_hyp, 1)
        consistency = self.consistency_scorer(hyp_ctx_flat).view(batch_size, max_hyp, 1)

        scores = torch.cat([coherence, evidence_score, query_compat, consistency], dim=-1)
        combined_scores = scores.mean(dim=-1)
        winner_indices = combined_scores.argmax(dim=-1)

        winner_expanded = winner_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, H)
        winner = all_hypotheses.gather(1, winner_expanded).squeeze(1)

        return winner, all_hypotheses, scores


class EvidenceTracker(nn.Module):
    """Track which Knowledge Units support each hypothesis.

    Args:
        config: RILAConfig with hypothesis and knowledge parameters.
    """

    SUPPORT_THRESHOLD: float = 0.5

    def __init__(self, config: RILAConfig) -> None:
        super().__init__()
        self.config = config
        H = config.cognitive_dim
        K = config.knowledge_dim

        self.ku_projection = nn.Linear(K, H)
        self.evidence_scorer = nn.Sequential(
            nn.Linear(H + K, H), nn.GELU(), nn.Linear(H, 1), nn.Sigmoid(),
        )

    def forward(
        self, hypotheses: torch.Tensor, knowledge_units: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute evidence support matrix.

        Args:
            hypotheses: (batch, max_hyp, cognitive_dim).
            knowledge_units: (batch, num_ku, knowledge_dim).

        Returns:
            evidence_map: (batch, max_hyp, num_ku) scores in [0, 1].
            valid_mask: (batch, max_hyp) True if hypothesis has support.
        """
        batch_size = hypotheses.shape[0]
        max_hyp = hypotheses.shape[1]
        num_ku = knowledge_units.shape[1]
        H = self.config.cognitive_dim
        K = self.config.knowledge_dim

        hyp_expanded = hypotheses.unsqueeze(2).expand(-1, -1, num_ku, -1)
        ku_expanded = knowledge_units.unsqueeze(1).expand(-1, max_hyp, -1, -1)
        paired = torch.cat([hyp_expanded, ku_expanded], dim=-1)

        paired_flat = paired.view(batch_size * max_hyp * num_ku, H + K)
        scores_flat = self.evidence_scorer(paired_flat)
        evidence_map = scores_flat.view(batch_size, max_hyp, num_ku)

        has_support = (evidence_map > self.SUPPORT_THRESHOLD).any(dim=-1)
        return evidence_map, has_support
