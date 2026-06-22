"""Core Language Processor (CLP): cognitive engine of RILA.

Transforms Working Context into a stable Cognitive State through iterative
knowledge integration with convergence detection and stability constraints.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from pyrila.config import RILAConfig
from pyrila.modules.knowledge_extractor import KnowledgeExtractor
from pyrila.modules.knowledge_graph import CognitiveCompression, KnowledgeGraphBuilder
from pyrila.modules.relevance_gate import RelevanceGate


class CoreLanguageProcessor(nn.Module):
    """Main cognitive processing unit.

    Transforms working context (batch, cognitive_dim) into refined cognitive
    state via iterative GRU-based updates with stability constraints and
    convergence detection.

    Args:
        config: RILAConfig with CLP hyperparameters.
    """

    def __init__(self, config: RILAConfig) -> None:
        super().__init__()
        self.config = config

        self.knowledge_extractor = KnowledgeExtractor(config)
        self.relevance_gate = RelevanceGate(config)
        self.knowledge_graph_builder = KnowledgeGraphBuilder(config)
        self.cognitive_compression = CognitiveCompression(config)

        self.state_initializer = nn.Sequential(
            nn.Linear(config.cognitive_dim, config.cognitive_dim),
            nn.GELU(),
            nn.Linear(config.cognitive_dim, config.cognitive_dim),
        )

        self.cognitive_update = nn.GRUCell(
            input_size=config.knowledge_dim,
            hidden_size=config.cognitive_dim,
        )

        self.integration_transform = nn.Sequential(
            nn.Linear(config.cognitive_dim, config.knowledge_dim),
            nn.GELU(),
            nn.Linear(config.knowledge_dim, config.knowledge_dim),
        )

        self.state_to_knowledge = nn.Linear(config.cognitive_dim, config.knowledge_dim)

        self.convergence_detector = nn.Sequential(
            nn.Linear(config.cognitive_dim * 2, config.cognitive_dim),
            nn.GELU(),
            nn.Linear(config.cognitive_dim, 1),
            nn.Sigmoid(),
        )

        self.memory_update = nn.LSTMCell(config.cognitive_dim, config.cognitive_dim)
        self.stability_norm = config.cognitive_stability_lambda

    def forward(
        self,
        working_context: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Full CLP forward pass with iterative refinement.

        Args:
            working_context: (batch, cognitive_dim).
            initial_state: Optional previous cognitive state.

        Returns:
            cognitive_state: (batch, cognitive_dim).
            metadata: dict with iterations, convergence_score, converged, memory_state.
        """
        batch_size = working_context.shape[0]
        device = working_context.device
        dtype = working_context.dtype

        s_0 = initial_state if initial_state is not None else self.state_initializer(working_context)

        # Empty WC check
        if (working_context.norm(dim=-1) < 1e-7).all():
            h_mem = torch.zeros(batch_size, self.config.cognitive_dim, device=device, dtype=dtype)
            c_mem = torch.zeros(batch_size, self.config.cognitive_dim, device=device, dtype=dtype)
            return s_0, {
                "iterations": 0, "convergence_score": torch.zeros(batch_size, device=device),
                "converged": False, "memory_state": (h_mem, c_mem),
            }

        knowledge_input = self.integration_transform(working_context)

        h_mem = torch.zeros(batch_size, self.config.cognitive_dim, device=device, dtype=dtype)
        c_mem = torch.zeros(batch_size, self.config.cognitive_dim, device=device, dtype=dtype)

        s_t = s_0
        best_score = torch.zeros(batch_size, device=device, dtype=dtype)
        best_state = s_0.clone()
        converged = False
        final_score = torch.zeros(batch_size, device=device, dtype=dtype)
        num_iterations = 0

        for iteration in range(self.config.max_cognitive_iterations):
            num_iterations = iteration + 1

            state_knowledge = self.state_to_knowledge(s_t)
            gru_input = knowledge_input + state_knowledge

            s_next = self.cognitive_update(gru_input, s_t)
            s_next = self._enforce_stability(s_t, s_next)

            h_mem, c_mem = self.memory_update(s_next, (h_mem, c_mem))

            score = self._check_convergence(s_t, s_next)
            final_score = score

            if score.mean() > best_score.mean():
                best_score = score.clone()
                best_state = s_next.clone()

            s_t = s_next

            if score.mean().item() >= self.config.convergence_threshold:
                converged = True
                break

        cognitive_state_final = s_t if converged else best_state

        return cognitive_state_final, {
            "iterations": num_iterations,
            "convergence_score": final_score,
            "converged": converged,
            "memory_state": (h_mem, c_mem),
        }

    def _enforce_stability(self, s_prev: torch.Tensor, s_next: torch.Tensor) -> torch.Tensor:
        """Enforce ||s_{t+1} - s_t|| ≤ lambda."""
        delta = s_next - s_prev
        delta_norm = delta.norm(dim=-1, keepdim=True)
        needs_clamping = delta_norm > self.stability_norm
        direction = delta / delta_norm.clamp(min=1e-8)
        clamped = s_prev + self.stability_norm * direction
        return torch.where(needs_clamping, clamped, s_next)

    def _check_convergence(self, s_prev: torch.Tensor, s_curr: torch.Tensor) -> torch.Tensor:
        """Compute convergence score from consecutive states."""
        combined = torch.cat([s_prev, s_curr], dim=-1)
        return self.convergence_detector(combined).squeeze(-1)
