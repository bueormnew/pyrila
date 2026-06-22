"""Recursive Context Index: dynamic affinity graph over encoded Context Cells.

Implements graph G = (V, E) where V = cells and E = learned affinity edges,
enabling fast retrieval without processing the full context.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from pyrila.config import RILAConfig
from pyrila.exceptions import IndexNotBuiltError


class RecursiveContextIndex(nn.Module):
    """Graph-based index over encoded Context Cells.

    Builds an affinity graph where nodes are cell encodings and edges
    represent learned affinity scores. Supports top-K querying and
    graph-based neighbor expansion.

    Args:
        config: RILAConfig with cell_encoding_dim, hidden_dim, etc.
    """

    def __init__(self, config: RILAConfig) -> None:
        super().__init__()
        self.config = config

        self.affinity_network = nn.Sequential(
            nn.Linear(config.cell_encoding_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.cell_encodings: Optional[torch.Tensor] = None
        self.adjacency: Optional[torch.Tensor] = None

    def build_index(self, cell_encodings: torch.Tensor) -> None:
        """Build the affinity graph from cell encodings.

        Args:
            cell_encodings: (num_cells, cell_encoding_dim).
        """
        self.cell_encodings = cell_encodings
        num_cells = cell_encodings.shape[0]

        if num_cells <= 1:
            self.adjacency = torch.zeros(num_cells, num_cells, device=cell_encodings.device)
            return

        affinity_matrix = self._compute_pairwise_affinity(cell_encodings)
        affinity_matrix = affinity_matrix * (1.0 - torch.eye(num_cells, device=cell_encodings.device))

        adjacency = affinity_matrix.clone()
        adjacency[adjacency < self.config.index_affinity_threshold] = 0.0
        adjacency = self._enforce_max_degree(adjacency)
        adjacency = self._ensure_connectivity(adjacency, affinity_matrix)
        self.adjacency = adjacency

    def _compute_pairwise_affinity(self, cell_encodings: torch.Tensor) -> torch.Tensor:
        """Compute pairwise affinity scores."""
        num_cells = cell_encodings.shape[0]
        enc_i = cell_encodings.unsqueeze(1).expand(-1, num_cells, -1)
        enc_j = cell_encodings.unsqueeze(0).expand(num_cells, -1, -1)
        pairs = torch.cat([enc_i, enc_j], dim=-1)
        return self.affinity_network(pairs).squeeze(-1)

    def compute_affinity(self, encoding_i: torch.Tensor, encoding_j: torch.Tensor) -> torch.Tensor:
        """Compute affinity between two encodings."""
        if encoding_i.dim() == 1:
            encoding_i = encoding_i.unsqueeze(0)
            encoding_j = encoding_j.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        score = self.affinity_network(torch.cat([encoding_i, encoding_j], dim=-1)).squeeze(-1)
        return score.squeeze(0) if squeeze else score

    def _enforce_max_degree(self, adjacency: torch.Tensor) -> torch.Tensor:
        """Keep only top max_index_connections edges per node."""
        max_conn = self.config.max_index_connections
        num_cells = adjacency.shape[0]
        if num_cells <= max_conn:
            return adjacency

        result = adjacency.clone()
        for i in range(num_cells):
            row = result[i]
            if (row > 0).sum().item() > max_conn:
                _, topk_idx = torch.topk(row, max_conn)
                mask = torch.zeros_like(row)
                mask[topk_idx] = 1.0
                result[i] = row * mask

        return torch.max(result, result.t())

    def _ensure_connectivity(self, adjacency: torch.Tensor, full_affinity: torch.Tensor) -> torch.Tensor:
        """Connect disconnected components with highest-affinity edges."""
        num_cells = adjacency.shape[0]
        if num_cells <= 1:
            return adjacency

        components = self._find_components(adjacency)
        if len(components) <= 1:
            return adjacency

        result = adjacency.clone()
        connected = set(components[0])
        for comp in components[1:]:
            best_score, best_i, best_j = -1.0, -1, -1
            for i in connected:
                for j in comp:
                    score = full_affinity[i, j].item()
                    if score > best_score:
                        best_score, best_i, best_j = score, i, j
            if best_i >= 0:
                result[best_i, best_j] = full_affinity[best_i, best_j]
                result[best_j, best_i] = full_affinity[best_j, best_i]
                connected.update(comp)
        return result

    def _find_components(self, adjacency: torch.Tensor) -> list:
        """Find connected components via BFS."""
        num_cells = adjacency.shape[0]
        visited: set = set()
        components = []
        for start in range(num_cells):
            if start in visited:
                continue
            component = []
            queue = [start]
            visited.add(start)
            while queue:
                node = queue.pop(0)
                component.append(node)
                neighbors = (adjacency[node] > 0).nonzero(as_tuple=True)[0]
                for n in neighbors.tolist():
                    if n not in visited:
                        visited.add(n)
                        queue.append(n)
            components.append(component)
        return components

    def query(self, query_vector: torch.Tensor, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve top-K most relevant cell indices and scores.

        Args:
            query_vector: (cell_encoding_dim,) query.
            top_k: Number of cells to retrieve.

        Returns:
            (indices, scores) tensors of shape (k,).
        """
        if self.cell_encodings is None:
            raise IndexNotBuiltError("query")

        num_cells = self.cell_encodings.shape[0]
        k = min(top_k, num_cells)
        query_expanded = query_vector.unsqueeze(0).expand(num_cells, -1)
        scores = self.compute_affinity(query_expanded, self.cell_encodings)
        return torch.topk(scores, k)

    def expand_from(self, cell_indices: torch.Tensor, hops: int = 1) -> torch.Tensor:
        """Expand retrieval to neighbors within hop distance.

        Args:
            cell_indices: Starting cell indices.
            hops: Number of hops.

        Returns:
            Neighboring cell indices (excluding originals).
        """
        if self.adjacency is None:
            raise IndexNotBuiltError("expand_from")

        if cell_indices.dim() == 0:
            cell_indices = cell_indices.unsqueeze(0)

        original_set = set(cell_indices.tolist())
        current_frontier = set(cell_indices.tolist())
        all_visited = set(cell_indices.tolist())

        for _ in range(hops):
            next_frontier: set = set()
            for node in current_frontier:
                neighbors = (self.adjacency[node] > 0).nonzero(as_tuple=True)[0]
                for n in neighbors.tolist():
                    if n not in all_visited:
                        next_frontier.add(n)
                        all_visited.add(n)
            current_frontier = next_frontier

        expanded = sorted(all_visited - original_set)
        if not expanded:
            return torch.tensor([], dtype=torch.long, device=self.adjacency.device)
        return torch.tensor(expanded, dtype=torch.long, device=self.adjacency.device)
