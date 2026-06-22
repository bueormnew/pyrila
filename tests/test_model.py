"""Tests for RILA model instantiation and forward pass."""

import torch
import pytest

from pyrila import RILA, RILAConfig
from pyrila.presets import rila_small, rila_base


class TestModelInstantiation:
    """Test model creation with different configs."""

    def test_small_model_creates(self, small_config):
        """Model instantiates with small config."""
        model = RILA(small_config)
        assert model is not None

    def test_model_has_all_components(self, model):
        """Model has all expected sub-modules."""
        assert hasattr(model, "tokenizer")
        assert hasattr(model, "cell_builder")
        assert hasattr(model, "cell_encoder")
        assert hasattr(model, "context_index")
        assert hasattr(model, "rce")
        assert hasattr(model, "working_context")
        assert hasattr(model, "clp")
        assert hasattr(model, "pre_output_gen")
        assert hasattr(model, "rve")
        assert hasattr(model, "reasoning_loop")
        assert hasattr(model, "final_decoder")


class TestForwardPass:
    """Test model forward pass."""

    def test_forward_training_mode(self, model, sample_batch):
        """Forward pass produces logits in training mode."""
        model.train()
        outputs = model(
            input_ids=sample_batch["input_ids"],
            target_ids=sample_batch["target_ids"],
        )
        assert "logits" in outputs
        assert outputs["logits"].shape == (2, 32, model.config.vocab_size)
        assert "confidence" in outputs
        assert "retrieval_stats" in outputs
        assert "budget_used" in outputs

    def test_forward_with_attention_mask(self, model, sample_batch):
        """Forward pass works with explicit attention mask."""
        model.train()
        outputs = model(
            input_ids=sample_batch["input_ids"],
            attention_mask=sample_batch["attention_mask"],
            target_ids=sample_batch["target_ids"],
        )
        assert outputs["logits"].shape[0] == 2

    def test_confidence_in_range(self, model, sample_batch):
        """Confidence is in [0, 1]."""
        model.train()
        outputs = model(
            input_ids=sample_batch["input_ids"],
            target_ids=sample_batch["target_ids"],
        )
        conf = outputs["confidence"]
        assert (conf >= 0.0).all()
        assert (conf <= 1.0).all()


class TestGeneration:
    """Test inference/generation mode."""

    def test_generate_produces_tokens(self, model, small_config):
        """Generate produces valid token IDs."""
        model.eval()
        input_ids = torch.randint(0, small_config.vocab_size, (1, 64))
        with torch.no_grad():
            generated = model.generate(input_ids, max_length=20)
        assert generated.shape[0] == 1
        assert generated.shape[1] <= 20
        assert (generated >= 0).all()
        assert (generated < small_config.vocab_size).all()

    def test_generate_batch(self, model, small_config):
        """Generate works with batch size > 1."""
        model.eval()
        input_ids = torch.randint(0, small_config.vocab_size, (2, 64))
        with torch.no_grad():
            generated = model.generate(input_ids, max_length=10)
        assert generated.shape[0] == 2


class TestDeviceCompatibility:
    """Test CPU/GPU compatibility."""

    def test_model_on_cpu(self, small_config):
        """Model works on CPU."""
        model = RILA(small_config)
        input_ids = torch.randint(0, small_config.vocab_size, (1, 64))
        target_ids = torch.randint(0, small_config.vocab_size, (1, 16))
        model.train()
        outputs = model(input_ids, target_ids=target_ids)
        assert outputs["logits"].device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_on_cuda(self, small_config):
        """Model works on CUDA."""
        model = RILA(small_config).cuda()
        input_ids = torch.randint(0, small_config.vocab_size, (1, 64)).cuda()
        target_ids = torch.randint(0, small_config.vocab_size, (1, 16)).cuda()
        model.train()
        outputs = model(input_ids, target_ids=target_ids)
        assert outputs["logits"].device.type == "cuda"
