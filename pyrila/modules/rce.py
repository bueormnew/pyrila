"""Recursive Context Explorer: iterative retrieval from the Context Index.

Implements progressive discovery of relevant Context Cells with soft top-k
selection during training for improved gradient flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyrila.config import RILAConfig
from pyrila.exceptions import IndexNotBuiltError
from pyrila.modules.context_index import RecursiveContextIndex


@dataclass
class RetrievalResult:
    """Result of a retrieval session.

    Attributes:
        cell_indices: (num_retrieved,) selected cell indices.
        cell_encodings: (num_retrieved, cell_encoding_dim) encodings.
        relevance_scores: (num_retrieved,) scores in [0, 1].
        num_retrievals: Total retrieval steps taken.
    """

    cell_indices: torch.Tensor
    cell_encodings: torch.Tensor
    relevance_scores: torch.Tensor
    num_retrievals: int


class RecursiveContextExplorer(nn.Module):
    """Iterative retrieval with soft top-k training and hard top-k inference.

    Process:
        1. Project cognitive state into query space.
        2. Score all cells via learned scoring network.
        3. Select initial_top_k (soft in training, hard in inference).
        4. Check sufficiency; expand via graph neighbors if needed.
        5. Return unique retrieved cells.

    Args:
        config: RILAConfig with retrieval hyperparameters.
    """

    def __init__(self, config: RILAConfig) -> None:
        super().__init__()
        self.config = config

        self.query_projector = nn.Sequential(
            nn.Linear(config.cognitive_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.cell_encoding_dim),
        )

        self.scoring_network = nn.Sequential(
            nn.Linear(config.cell_encoding_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.sufficiency_predictor = nn.Sequential(
            nn.Linear(config.cognitive_dim + config.cell_encoding_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid(),
        )

        self._soft_aggregated: Optional[torch.Tensor] = None

    def forward(
        self,
        cognitive_state: torch.Tensor,
        index: RecursiveContextIndex,
        max_expansions: Optional[int] = None,
    ) -> RetrievalResult:
        """Retrieve relevant cells from the index.

        Uses soft top-k in training mode (differentiable selection) and
        hard top-k in inference mode (exact selection).

        Args:
            cognitive_state: (cognitive_dim,) current cognitive state.
            index: Built RecursiveContextIndex.
            max_expansions: Override for max expansion iterations.

        Returns:
            RetrievalResult with unique retrieved cells.
        """
        if index.cell_encodings is None:
            raise IndexNotBuiltError("retrieve")

        if max_expansions is None:
            max_expansions = self.config.max_retrieval_expansions

        cell_encodings = index.cell_encodings
        num_cells = cell_encodings.shape[0]

        # Project to query space
        query = self.query_projector(cognitive_state)

        # Score all cells
        scores = self._compute_scores(query, cell_encodings)

        # Initial selection
        k = min(self.config.initial_top_k, num_cells)

        if self.training and self.config.use_soft_topk:
            # Soft selection: differentiable weights for gradient flow
            soft_weights = F.softmax(scores / self.config.soft_topk_temperature, dim=-1)
            self._soft_aggregated = (soft_weights.unsqueeze(-1) * cell_encodings).sum(dim=0)
            _, topk_indices = torch.topk(scores, k)
            topk_scores = scores[topk_indices]
        else:
            topk_scores, topk_indices = torch.topk(scores, k)
            self._soft_aggregated = None

        if k == 0:
            topk_indices = torch.tensor([0], dtype=torch.long, device=scores.device)
            topk_scores = scores[:1]

        retrieved_set = set(topk_indices.tolist())
        all_indices = topk_indices.tolist()
        all_scores = topk_scores.tolist()
        num_retrievals = 1

        # Iterative expansion
        for _ in range(max_expansions):
            if self._should_expand(cognitive_state, cell_encodings, all_indices):
                current_tensor = torch.tensor(
                    all_indices, dtype=torch.long, device=cell_encodings.device
                )
                neighbor_indices = index.expand_from(current_tensor, hops=1)

                if neighbor_indices.numel() == 0:
                    break

                new_candidates = [i for i in neighbor_indices.tolist() if i not in retrieved_set]
                if not new_candidates:
                    break

                candidate_tensor = torch.tensor(
                    new_candidates, dtype=torch.long, device=cell_encodings.device
                )
                candidate_encodings = cell_encodings[candidate_tensor]
                candidate_scores = self._compute_scores(query, candidate_encodings)

                sorted_order = torch.argsort(candidate_scores, descending=True)
                for sort_idx in sorted_order:
                    cell_idx = new_candidates[sort_idx.item()]
                    if cell_idx not in retrieved_set:
                        retrieved_set.add(cell_idx)
                        all_indices.append(cell_idx)
                        all_scores.append(candidate_scores[sort_idx].item())

                num_retrievals += 1
            else:
                break

        if not all_indices:
            best_idx = torch.argmax(scores).item()
            all_indices = [best_idx]
            all_scores = [scores[best_idx].item()]

        result_indices = torch.tensor(all_indices, dtype=torch.long, device=cell_encodings.device)
        result_encodings = cell_encodings[result_indices]
        result_scores = torch.tensor(all_scores, dtype=torch.float, device=cell_encodings.device)

        return RetrievalResult(
            cell_indices=result_indices,
            cell_encodings=result_encodings,
            relevance_scores=result_scores,
            num_retrievals=num_retrievals,
        )

    def _compute_scores(self, query: torch.Tensor, cell_encodings: torch.Tensor) -> torch.Tensor:
        """Score cells against query vector."""
        num_cells = cell_encodings.shape[0]
        query_expanded = query.unsqueeze(0).expand(num_cells, -1)
        combined = torch.cat([query_expanded, cell_encodings], dim=-1)
        return self.scoring_network(combined).squeeze(-1)

    def _should_expand(
        self, cognitive_state: torch.Tensor, cell_encodings: torch.Tensor, retrieved_indices: list
    ) -> bool:
        """Check if more retrieval is needed via sufficiency predictor."""
        if not retrieved_indices:
            return True

        if self.training and self.config.use_soft_topk and self._soft_aggregated is not None:
            aggregated = self._soft_aggregated
        else:
            indices_tensor = torch.tensor(
                retrieved_indices, dtype=torch.long, device=cell_encodings.device
            )
            aggregated = cell_encodings[indices_tensor].mean(dim=0)

        combined = torch.cat([cognitive_state, aggregated], dim=-1)
        sufficiency = self.sufficiency_predictor(combined).squeeze(-1)
        return sufficiency.item() < self.config.retrieval_score_threshold
