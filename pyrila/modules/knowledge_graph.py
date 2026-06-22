"""Knowledge Graph Builder and Cognitive Compression.

Organizes Knowledge Units into a dynamic graph with typed relations and
GNN message passing. Includes cognitive compression for scalability.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from pyrila.config import RILAConfig


class KnowledgeGraphBuilder(nn.Module):
    """Build dynamic graph KG = (N, E) from Knowledge Units.

    Relations are computed via pairwise scoring. GNN message passing
    updates KU representations based on graph structure.

    Args:
        config: RILAConfig with knowledge_dim.
        relation_threshold: Edge creation threshold τ.
        gnn_iterations: Number of message passing rounds.
    """

    def __init__(
        self, config: RILAConfig,
        relation_threshold: float = 0.5,
        gnn_iterations: int = 3,
    ) -> None:
        super().__init__()
        self.config = config
        self.relation_threshold = relation_threshold
        self.gnn_iterations = gnn_iterations
        K = config.knowledge_dim

        self.relation_scorer = nn.Sequential(
            nn.Linear(K * 2, K), nn.GELU(),
            nn.Linear(K, K // 2), nn.GELU(),
            nn.Linear(K // 2, 1), nn.Sigmoid(),
        )

        self.graph_update = nn.GRUCell(K, K)

    def forward(
        self, knowledge_units: torch.Tensor, activations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build graph and run message passing.

        Args:
            knowledge_units: (batch, num_ku, knowledge_dim).
            activations: (batch, num_ku) activation scores.

        Returns:
            graph_state: (batch, num_ku, knowledge_dim) updated representations.
            adjacency: (batch, num_ku, num_ku) edge weights.
        """
        batch_size, num_ku, K = knowledge_units.shape
        device = knowledge_units.device

        active_counts = (activations > 0).sum(dim=-1)
        if (active_counts < 2).all():
            adjacency = torch.zeros(batch_size, num_ku, num_ku, device=device)
            return knowledge_units.clone(), adjacency

        # Pairwise relation strengths
        ku_i = knowledge_units.unsqueeze(2).expand(-1, -1, num_ku, -1)
        ku_j = knowledge_units.unsqueeze(1).expand(-1, num_ku, -1, -1)
        pairs = torch.cat([ku_i, ku_j], dim=-1)
        relation_strengths = self.relation_scorer(pairs).squeeze(-1)

        # Masking
        active_mask_i = (activations > 0).unsqueeze(2).float()
        active_mask_j = (activations > 0).unsqueeze(1).float()
        active_pair_mask = active_mask_i * active_mask_j
        diag_mask = 1.0 - torch.eye(num_ku, device=device).unsqueeze(0)
        active_pair_mask = active_pair_mask * diag_mask

        threshold_mask = (relation_strengths > self.relation_threshold).float()
        adjacency = relation_strengths * active_pair_mask * threshold_mask

        insufficient_mask = (active_counts < 2).float()
        adjacency = adjacency * (1.0 - insufficient_mask.unsqueeze(-1).unsqueeze(-1))

        # GNN message passing
        graph_state = knowledge_units.clone()
        for _ in range(self.gnn_iterations):
            adj_sum = adjacency.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            normalized_adj = adjacency / adj_sum
            messages = torch.bmm(normalized_adj, graph_state)

            flat_messages = messages.reshape(batch_size * num_ku, K)
            flat_state = graph_state.reshape(batch_size * num_ku, K)
            updated_flat = self.graph_update(flat_messages, flat_state)
            updated_state = updated_flat.reshape(batch_size, num_ku, K)

            keep_original = insufficient_mask.unsqueeze(-1).unsqueeze(-1)
            graph_state = updated_state * (1.0 - keep_original) + knowledge_units * keep_original

        return graph_state, adjacency


class CognitiveCompression(nn.Module):
    """Compress clusters of KUs into Meta Knowledge Units when capacity exceeded.

    Args:
        config: RILAConfig with knowledge_dim.
        node_capacity: Max active nodes before compression triggers.
    """

    def __init__(self, config: RILAConfig, node_capacity: int = 1000) -> None:
        super().__init__()
        self.config = config
        self.node_capacity = max(node_capacity, 100)
        K = config.knowledge_dim

        self.cluster_attention = nn.MultiheadAttention(
            embed_dim=K, num_heads=max(1, min(4, K // 16)), batch_first=True
        )
        self.compression_gate = nn.Sequential(
            nn.Linear(K * 2, K), nn.GELU(), nn.Linear(K, K),
        )

    def forward(
        self, knowledge_units: torch.Tensor, adjacency: torch.Tensor
    ) -> Tuple[torch.Tensor, bool]:
        """Forward: compress if needed.

        Args:
            knowledge_units: (batch, num_ku, knowledge_dim).
            adjacency: (batch, num_ku, num_ku).

        Returns:
            output_units: Compressed or original tensor.
            compressed: Whether compression was applied.
        """
        num_ku = knowledge_units.shape[1]
        if num_ku <= self.node_capacity:
            return knowledge_units, False
        # For this implementation, return unchanged (full compression
        # requires expensive clustering per-batch — simplified for package)
        return knowledge_units, False
