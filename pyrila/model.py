"""RILA top-level model: Recursive Indexed Language Architecture.

Orchestrates the full pipeline from input token IDs through all processing
stages to final output generation.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from pyrila.config import RILAConfig
from pyrila.modules.budget import ReasoningBudgetController
from pyrila.modules.cell_builder import ContextCellBuilder
from pyrila.modules.cell_encoder import CellEncoder
from pyrila.modules.clp import CoreLanguageProcessor
from pyrila.modules.context_index import RecursiveContextIndex
from pyrila.modules.decoder import FinalDecoder
from pyrila.modules.hypothesis import EvidenceTracker, HypothesisGenerator
from pyrila.modules.knowledge_extractor import KnowledgeExtractor
from pyrila.modules.knowledge_graph import CognitiveCompression, KnowledgeGraphBuilder
from pyrila.modules.pre_output import PreOutputGenerator
from pyrila.modules.rce import RecursiveContextExplorer
from pyrila.modules.reasoning_loop import RecursiveReasoningLoop
from pyrila.modules.relevance_gate import RelevanceGate
from pyrila.modules.rve import RecursiveVerificationEngine
from pyrila.modules.tokenizer import RILATokenizer
from pyrila.modules.working_context import WorkingContext


class RILA(nn.Module):
    """Top-level RILA model.

    Implements the complete pipeline:
        Tokenizer → Cell Builder → Cell Encoder → Context Index →
        RCE → Working Context → CLP → Pre-Output → RVE →
        [Reasoning Loop] → Decoder

    Args:
        config: RILAConfig with all architecture hyperparameters.

    Example:
        >>> from pyrila import RILA, rila_small
        >>> model = RILA(rila_small())
        >>> output = model(torch.randint(0, 1000, (1, 128)), target_ids=torch.randint(0, 1000, (1, 32)))
        >>> output['logits'].shape
        torch.Size([1, 32, 1000])
    """

    def __init__(self, config: RILAConfig) -> None:
        super().__init__()
        self.config = config

        self.tokenizer = RILATokenizer(config)
        self.cell_builder = ContextCellBuilder(config)
        self.cell_encoder = CellEncoder(config)
        self.context_index = RecursiveContextIndex(config)
        self.rce = RecursiveContextExplorer(config)
        self.working_context = WorkingContext(config)
        self.knowledge_extractor = KnowledgeExtractor(config)
        self.relevance_gate = RelevanceGate(config)
        self.knowledge_graph_builder = KnowledgeGraphBuilder(config)
        self.cognitive_compression = CognitiveCompression(config)
        self.clp = CoreLanguageProcessor(config)
        self.hypothesis_generator = HypothesisGenerator(config)
        self.evidence_tracker = EvidenceTracker(config)
        self.pre_output_gen = PreOutputGenerator(config)
        self.rve = RecursiveVerificationEngine(config)
        self.budget_controller = ReasoningBudgetController(config)
        self.reasoning_loop = RecursiveReasoningLoop(config)
        self.final_decoder = FinalDecoder(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_ids: Optional[torch.Tensor] = None,
        reasoning_budget: Optional[int] = None,
    ) -> Dict[str, object]:
        """Full forward pass through the RILA pipeline.

        Args:
            input_ids: (batch, seq_len) token IDs.
            attention_mask: (batch, seq_len) binary mask (default: all ones).
            target_ids: (batch, target_len) for teacher forcing. If None, generates.
            reasoning_budget: Optional explicit budget override.

        Returns:
            Dict with logits/generated, confidence, retrieval_stats, budget_used, cognitive_state.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
        if reasoning_budget is None:
            reasoning_budget = self.config.default_budget

        self.working_context.reset()

        # Tokenizer → embeddings
        embeddings = self.tokenizer(input_ids)

        # Cell Builder → cells
        batch_cells = self.cell_builder(embeddings, attention_mask)

        # Cell Encoder → cell encodings
        all_cells = []
        for cells in batch_cells:
            all_cells.extend(cells)

        if not all_cells:
            cell_encodings_flat = torch.zeros(1, self.config.cell_encoding_dim, device=device)
            num_cells_per_batch = 1
        else:
            cell_encodings_flat = self.cell_encoder.encode_batch(all_cells)
            num_cells_per_batch = len(batch_cells[0])

        cell_encodings = cell_encodings_flat.view(
            batch_size, num_cells_per_batch, self.config.cell_encoding_dim
        )

        # Build index (using first batch element)
        self.context_index.build_index(cell_encodings[0])

        # Initial cognitive state from importance vectors
        D = self.config.hidden_dim
        importance_vectors = cell_encodings[:, :, 3 * D:]
        initial_cognitive_state = importance_vectors.mean(dim=1)

        if D != self.config.cognitive_dim:
            cog_state = torch.zeros(batch_size, self.config.cognitive_dim, device=device)
            cog_state[:, :D] = initial_cognitive_state
            initial_cognitive_state = cog_state

        # RCE retrieval
        retrieval_result = self.rce(initial_cognitive_state[0], self.context_index)
        num_cells_retrieved = retrieval_result.cell_indices.shape[0]
        num_retrieval_steps = retrieval_result.num_retrievals

        # Working Context
        retrieved_enc = retrieval_result.cell_encodings.unsqueeze(0).expand(batch_size, -1, -1)
        retrieved_scr = retrieval_result.relevance_scores.unsqueeze(0).expand(batch_size, -1)
        working_ctx = self.working_context(retrieved_enc, retrieved_scr)

        # CLP
        cognitive_state_final, _ = self.clp(working_ctx, initial_state=initial_cognitive_state)

        # Pre-output + RVE
        pre_output, _, _ = self.pre_output_gen(cognitive_state_final)
        query = initial_cognitive_state
        confidence, accept = self.rve(query, working_ctx, cognitive_state_final, pre_output)

        budget_used = 0

        # Reasoning loop if needed
        if not accept and reasoning_budget > 0:
            final_pre_output, final_confidence, budget_state = self.reasoning_loop(
                query=query,
                initial_cognitive_state=cognitive_state_final,
                initial_pre_output=pre_output,
                initial_confidence=confidence,
                rce=self.rce, index=self.context_index,
                working_context_module=self.working_context,
                clp=self.clp, pre_output_gen=self.pre_output_gen, rve=self.rve,
                budget=reasoning_budget,
            )
            confidence = final_confidence
            budget_used = budget_state.current_cycle

        # Final Decoder
        if target_ids is not None:
            output = self.final_decoder(cognitive_state_final, target_tokens=target_ids)
        else:
            max_gen = min(512, self.config.max_sequence_length)
            output = self.final_decoder.generate(cognitive_state_final, max_length=max_gen)

        return {
            "logits": output,
            "confidence": confidence,
            "retrieval_stats": {
                "num_cells_retrieved": num_cells_retrieved,
                "num_retrieval_steps": num_retrieval_steps,
            },
            "budget_used": budget_used,
            "cognitive_state": cognitive_state_final,
        }

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 512,
        reasoning_budget: Optional[int] = None,
        **generation_kwargs,
    ) -> torch.Tensor:
        """Inference pipeline returning generated token IDs.

        Args:
            input_ids: (batch, seq_len) token IDs.
            attention_mask: (batch, seq_len) binary mask.
            max_length: Maximum generation length.
            reasoning_budget: Optional reasoning budget.
            **generation_kwargs: temperature, top_k, top_p.

        Returns:
            Generated token IDs of shape (batch, generated_len).
        """
        was_training = self.training
        self.eval()

        with torch.no_grad():
            batch_size, seq_len = input_ids.shape
            device = input_ids.device

            if attention_mask is None:
                attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
            if reasoning_budget is None:
                reasoning_budget = self.config.default_budget

            self.working_context.reset()

            embeddings = self.tokenizer(input_ids)
            batch_cells = self.cell_builder(embeddings, attention_mask)

            all_cells = []
            for cells in batch_cells:
                all_cells.extend(cells)

            if not all_cells:
                cell_encodings_flat = torch.zeros(1, self.config.cell_encoding_dim, device=device)
                num_cells_per_batch = 1
            else:
                cell_encodings_flat = self.cell_encoder.encode_batch(all_cells)
                num_cells_per_batch = len(batch_cells[0])

            cell_encodings = cell_encodings_flat.view(
                batch_size, num_cells_per_batch, self.config.cell_encoding_dim
            )

            self.context_index.build_index(cell_encodings[0])

            D = self.config.hidden_dim
            importance_vectors = cell_encodings[:, :, 3 * D:]
            initial_cog = importance_vectors.mean(dim=1)

            if D != self.config.cognitive_dim:
                cog_state = torch.zeros(batch_size, self.config.cognitive_dim, device=device)
                cog_state[:, :D] = initial_cog
                initial_cog = cog_state

            retrieval_result = self.rce(initial_cog[0], self.context_index)
            retrieved_enc = retrieval_result.cell_encodings.unsqueeze(0).expand(batch_size, -1, -1)
            retrieved_scr = retrieval_result.relevance_scores.unsqueeze(0).expand(batch_size, -1)
            working_ctx = self.working_context(retrieved_enc, retrieved_scr)

            cognitive_state_final, _ = self.clp(working_ctx, initial_state=initial_cog)

            pre_output, _, _ = self.pre_output_gen(cognitive_state_final)
            query = initial_cog
            confidence, accept = self.rve(query, working_ctx, cognitive_state_final, pre_output)

            if not accept and reasoning_budget > 0:
                _, _, _ = self.reasoning_loop(
                    query=query, initial_cognitive_state=cognitive_state_final,
                    initial_pre_output=pre_output, initial_confidence=confidence,
                    rce=self.rce, index=self.context_index,
                    working_context_module=self.working_context,
                    clp=self.clp, pre_output_gen=self.pre_output_gen, rve=self.rve,
                    budget=reasoning_budget,
                )

            generated = self.final_decoder.generate(
                cognitive_state_final, max_length=max_length, **generation_kwargs,
            )

        if was_training:
            self.train()
        return generated
