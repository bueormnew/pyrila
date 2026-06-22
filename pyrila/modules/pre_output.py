"""Pre-Output Generator: transforms Cognitive State into a candidate response.

Produces a candidate response embedding compatible with the RVE,
an evidence summary, and a cognitive state snapshot.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from pyrila.config import RILAConfig


class PreOutputGenerator(nn.Module):
    """Generate pre-output from Cognitive State: p_t = G(s_t).

    Args:
        config: RILAConfig with cognitive_dim and knowledge_dim.
    """

    def __init__(self, config: RILAConfig) -> None:
        super().__init__()
        self.config = config

        self.response_generator = nn.Sequential(
            nn.Linear(config.cognitive_dim, config.cognitive_dim * 2),
            nn.GELU(),
            nn.Linear(config.cognitive_dim * 2, config.cognitive_dim),
            nn.LayerNorm(config.cognitive_dim),
        )

        self.evidence_summarizer = nn.Sequential(
            nn.Linear(config.cognitive_dim, config.cognitive_dim),
            nn.GELU(),
            nn.Linear(config.cognitive_dim, config.cognitive_dim // 2),
        )

        self.evidence_projector = nn.Sequential(
            nn.Linear(config.knowledge_dim, config.cognitive_dim),
            nn.GELU(),
            nn.Linear(config.cognitive_dim, config.cognitive_dim // 2),
        )

    def forward(
        self,
        cognitive_state: torch.Tensor,
        evidence: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate pre-output for verification.

        Args:
            cognitive_state: (batch, cognitive_dim).
            evidence: Optional (batch, num_evidence, knowledge_dim).

        Returns:
            candidate: (batch, cognitive_dim).
            evidence_summary: (batch, cognitive_dim // 2).
            cognitive_snapshot: (batch, cognitive_dim) detached copy.
        """
        candidate = self.response_generator(cognitive_state)

        if evidence is not None and evidence.shape[1] > 0:
            projected = self.evidence_projector(evidence)
            evidence_summary = projected.mean(dim=1)
        else:
            evidence_summary = self.evidence_summarizer(cognitive_state)

        cognitive_snapshot = cognitive_state.clone().detach()
        return candidate, evidence_summary, cognitive_snapshot
