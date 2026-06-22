"""Tests for individual RILA modules."""

import torch
import pytest

from pyrila.presets import rila_small
from pyrila.modules.tokenizer import RILATokenizer
from pyrila.modules.cell_builder import ContextCellBuilder
from pyrila.modules.cell_encoder import CellEncoder
from pyrila.modules.context_index import RecursiveContextIndex
from pyrila.modules.rce import RecursiveContextExplorer
from pyrila.modules.working_context import WorkingContext
from pyrila.modules.clp import CoreLanguageProcessor
from pyrila.modules.pre_output import PreOutputGenerator
from pyrila.modules.rve import RecursiveVerificationEngine
from pyrila.modules.decoder import FinalDecoder
from pyrila.modules.knowledge_extractor import KnowledgeExtractor
from pyrila.modules.relevance_gate import RelevanceGate
from pyrila.modules.knowledge_graph import KnowledgeGraphBuilder
from pyrila.modules.budget import ReasoningBudgetController


@pytest.fixture
def config():
    return rila_small()


class TestTokenizer:
    def test_forward(self, config):
        tokenizer = RILATokenizer(config)
        ids = torch.randint(0, config.vocab_size, (2, 32))
        out = tokenizer(ids)
        assert out.shape == (2, 32, config.hidden_dim)

    def test_tokenize_text(self, config):
        tokenizer = RILATokenizer(config)
        ids = tokenizer.tokenize("hello")
        assert ids.shape[0] == 1
        assert ids.shape[1] == 6  # SOS + 5 chars


class TestCellBuilder:
    def test_forward(self, config):
        builder = ContextCellBuilder(config)
        embeddings = torch.randn(2, 128, config.hidden_dim)
        mask = torch.ones(2, 128, dtype=torch.long)
        cells = builder(embeddings, mask)
        assert len(cells) == 2
        assert len(cells[0]) == 128 // config.cell_size


class TestCellEncoder:
    def test_forward(self, config):
        encoder = CellEncoder(config)
        tokens = torch.randn(4, config.cell_size, config.hidden_dim)
        lengths = torch.full((4,), config.cell_size, dtype=torch.long)
        cce = encoder(tokens, lengths)
        assert cce.shape == (4, config.cell_encoding_dim)


class TestContextIndex:
    def test_build_and_query(self, config):
        index = RecursiveContextIndex(config)
        encodings = torch.randn(8, config.cell_encoding_dim)
        index.build_index(encodings)

        query = torch.randn(config.cell_encoding_dim)
        scores, indices = index.query(query, top_k=3)
        assert indices.shape[0] == 3
        assert scores.shape[0] == 3

    def test_expand_from(self, config):
        index = RecursiveContextIndex(config)
        encodings = torch.randn(8, config.cell_encoding_dim)
        index.build_index(encodings)

        start = torch.tensor([0])
        expanded = index.expand_from(start, hops=1)
        assert expanded.dtype == torch.long


class TestRCE:
    def test_forward(self, config):
        rce = RecursiveContextExplorer(config)
        index = RecursiveContextIndex(config)
        encodings = torch.randn(8, config.cell_encoding_dim)
        index.build_index(encodings)

        cog_state = torch.randn(config.cognitive_dim)
        result = rce(cog_state, index)
        assert result.cell_indices.numel() >= 1
        assert result.relevance_scores.shape == result.cell_indices.shape


class TestWorkingContext:
    def test_forward(self, config):
        wc = WorkingContext(config)
        encodings = torch.randn(2, 4, config.cell_encoding_dim)
        scores = torch.rand(2, 4)
        out = wc(encodings, scores)
        assert out.shape == (2, config.cognitive_dim)


class TestKnowledgeExtractor:
    def test_forward(self, config):
        ke = KnowledgeExtractor(config)
        tokens = torch.randn(2, config.cell_size, config.hidden_dim)
        lengths = torch.full((2,), config.cell_size, dtype=torch.long)
        kus, acts = ke(tokens, lengths)
        assert kus.shape == (2, config.max_knowledge_units_per_cell, config.knowledge_dim)
        assert acts.shape == (2, config.max_knowledge_units_per_cell)
        assert (acts >= 0).all() and (acts <= 1).all()


class TestRelevanceGate:
    def test_forward(self, config):
        gate = RelevanceGate(config)
        kus = torch.randn(2, 4, config.knowledge_dim)
        gated, scores = gate(kus)
        assert gated.shape == kus.shape
        assert scores.shape == (2, 4)


class TestKnowledgeGraph:
    def test_forward(self, config):
        builder = KnowledgeGraphBuilder(config)
        kus = torch.randn(2, 4, config.knowledge_dim)
        acts = torch.rand(2, 4)
        state, adj = builder(kus, acts)
        assert state.shape == kus.shape
        assert adj.shape == (2, 4, 4)


class TestCLP:
    def test_forward(self, config):
        clp = CoreLanguageProcessor(config)
        wc = torch.randn(2, config.cognitive_dim)
        state, meta = clp(wc)
        assert state.shape == (2, config.cognitive_dim)
        assert "iterations" in meta
        assert "converged" in meta


class TestPreOutput:
    def test_forward(self, config):
        gen = PreOutputGenerator(config)
        state = torch.randn(2, config.cognitive_dim)
        candidate, evidence, snapshot = gen(state)
        assert candidate.shape == (2, config.cognitive_dim)
        assert evidence.shape == (2, config.cognitive_dim // 2)
        assert snapshot.shape == (2, config.cognitive_dim)


class TestRVE:
    def test_forward(self, config):
        rve = RecursiveVerificationEngine(config)
        q = torch.randn(2, config.cognitive_dim)
        wc = torch.randn(2, config.cognitive_dim)
        s = torch.randn(2, config.cognitive_dim)
        p = torch.randn(2, config.cognitive_dim)
        conf, accept = rve(q, wc, s, p)
        assert conf.shape == (2, 1)
        assert isinstance(accept, bool)
        assert (conf >= 0).all() and (conf <= 1).all()


class TestDecoder:
    def test_training_forward(self, config):
        decoder = FinalDecoder(config)
        decoder.train()
        state = torch.randn(2, config.cognitive_dim)
        targets = torch.randint(0, config.vocab_size, (2, 16))
        logits = decoder(state, target_tokens=targets)
        assert logits.shape == (2, 16, config.vocab_size)

    def test_generate(self, config):
        decoder = FinalDecoder(config)
        state = torch.randn(1, config.cognitive_dim)
        generated = decoder.generate(state, max_length=10)
        assert generated.shape[0] == 1
        assert generated.shape[1] <= 10


class TestBudgetController:
    def test_create_budget(self, config):
        controller = ReasoningBudgetController(config)
        state = controller.create_budget(budget=5)
        assert state.max_cycles == 5
        assert state.current_cycle == 0

    def test_should_continue(self, config):
        controller = ReasoningBudgetController(config)
        state = controller.create_budget(budget=3)
        assert controller.should_continue(state, 0.1) is True
        assert controller.should_continue(state, 0.99) is False
