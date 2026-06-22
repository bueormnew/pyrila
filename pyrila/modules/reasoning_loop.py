"""Recursive Reasoning Loop: iterative refinement when confidence is low.

Orchestrates cycles of failure analysis, query refinement, retrieval,
context expansion, CLP re-run, and verification until confidence is met
or budget is exhausted.
"""

from __future__ import annotations

from typing import Optional, Set, Tuple

import torch
import torch.nn as nn

from pyrila.config import RILAConfig
from pyrila.modules.budget import BudgetState, ReasoningBudgetController
from pyrila.modules.clp import CoreLanguageProcessor
from pyrila.modules.context_index import RecursiveContextIndex
from pyrila.modules.pre_output import PreOutputGenerator
from pyrila.modules.rce import RecursiveContextExplorer
from pyrila.modules.rve import RecursiveVerificationEngine
from pyrila.modules.working_context import WorkingContext


class RecursiveReasoningLoop(nn.Module):
    """Iterative reasoning loop with budget control.

    Args:
        config: RILAConfig with reasoning parameters.
    """

    def __init__(self, config: RILAConfig) -> None:
        super().__init__()
        self.config = config
        self.budget_controller = ReasoningBudgetController(config)

        self.failure_analyzer = nn.Sequential(
            nn.Linear(config.cognitive_dim * 2, config.cognitive_dim),
            nn.GELU(),
            nn.Linear(config.cognitive_dim, config.cognitive_dim),
        )

        self.confidence_expander = nn.Linear(1, config.cognitive_dim)
        self.query_refiner = nn.GRUCell(config.cognitive_dim, config.cognitive_dim)

    def forward(
        self,
        query: torch.Tensor,
        initial_cognitive_state: torch.Tensor,
        initial_pre_output: torch.Tensor,
        initial_confidence: torch.Tensor,
        rce: RecursiveContextExplorer,
        index: RecursiveContextIndex,
        working_context_module: WorkingContext,
        clp: CoreLanguageProcessor,
        pre_output_gen: PreOutputGenerator,
        rve: RecursiveVerificationEngine,
        budget: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, BudgetState]:
        """Run the recursive reasoning loop.

        Args:
            query: (batch, cognitive_dim).
            initial_cognitive_state: (batch, cognitive_dim).
            initial_pre_output: (batch, cognitive_dim).
            initial_confidence: (batch, 1).
            rce, index, working_context_module, clp, pre_output_gen, rve: Shared modules.
            budget: Optional explicit budget.

        Returns:
            final_pre_output: (batch, cognitive_dim).
            final_confidence: (batch, 1).
            budget_state: Final BudgetState.
        """
        batch_size = query.shape[0]
        device = query.device

        budget_state = self.budget_controller.create_budget(
            budget=budget, cognitive_state=initial_cognitive_state
        )
        budget_state.best_confidence = initial_confidence.mean().item()
        budget_state.best_pre_output = initial_pre_output.clone().detach()

        current_query = query
        cognitive_state = initial_cognitive_state
        current_pre_output = initial_pre_output
        current_confidence = initial_confidence

        retrieved_indices: Set[int] = set()
        accumulated_encodings: Optional[torch.Tensor] = None
        accumulated_scores: Optional[torch.Tensor] = None

        while self.budget_controller.should_continue(
            budget_state, current_confidence.mean().item()
        ):
            # Failure analysis
            confidence_expanded = self.confidence_expander(current_confidence)
            failure_input = torch.cat([cognitive_state, confidence_expanded], dim=-1)
            analysis = self.failure_analyzer(failure_input)

            # Query refinement
            current_query = self.query_refiner(analysis, current_query)

            # Retrieve new cells
            retrieval_result = rce(current_query[0], index)

            new_encodings_list = []
            new_scores_list = []
            for i, idx in enumerate(retrieval_result.cell_indices.tolist()):
                if idx not in retrieved_indices:
                    new_encodings_list.append(retrieval_result.cell_encodings[i])
                    new_scores_list.append(retrieval_result.relevance_scores[i])
                    retrieved_indices.add(idx)

            if new_encodings_list:
                new_enc = torch.stack(new_encodings_list, dim=0).unsqueeze(0).expand(batch_size, -1, -1)
                new_scr = torch.stack(new_scores_list, dim=0).unsqueeze(0).expand(batch_size, -1)

                if accumulated_encodings is None:
                    accumulated_encodings = new_enc
                    accumulated_scores = new_scr
                else:
                    accumulated_encodings = torch.cat([accumulated_encodings, new_enc], dim=1)
                    accumulated_scores = torch.cat([accumulated_scores, new_scr], dim=1)

            if accumulated_encodings is not None:
                working_ctx = working_context_module(accumulated_encodings, accumulated_scores)
            else:
                working_ctx = current_query

            cognitive_state, _ = clp(working_ctx, initial_state=cognitive_state)
            current_pre_output, _, _ = pre_output_gen(cognitive_state)
            current_confidence, _ = rve(current_query, working_ctx, cognitive_state, current_pre_output)

            budget_state = self.budget_controller.step(
                budget_state, current_confidence.mean().item(), current_pre_output
            )

        if current_confidence.mean().item() >= self.config.confidence_threshold:
            return current_pre_output, current_confidence, budget_state
        else:
            final_conf = torch.tensor(
                [[budget_state.best_confidence]], device=device
            ).expand(batch_size, 1)
            return budget_state.best_pre_output, final_conf, budget_state
