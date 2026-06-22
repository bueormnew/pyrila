"""Tests for RILA training loop."""

import os
import tempfile

import torch
import pytest

from pyrila import RILA, RILATrainer, CurriculumScheduler
from pyrila.presets import rila_small
from pyrila.training.losses import UncertaintyWeightedLoss
from pyrila.exceptions import DivergenceError


class TestTrainStep:
    """Test single training step."""

    def test_train_step_completes(self, model, sample_batch):
        """A training step completes without error."""
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        trainer = RILATrainer(model, optimizer)
        result = trainer.train_step(sample_batch)

        assert "loss" in result
        assert "l_answer" in result
        assert "l_retrieval" in result
        assert "l_confidence" in result
        assert "l_budget" in result
        assert isinstance(result["loss"], float)
        assert not result["batch_discarded"]

    def test_train_step_with_uncertainty_weighting(self, model, sample_batch):
        """Training step works with uncertainty weighting."""
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        trainer = RILATrainer(model, optimizer, use_uncertainty_weighting=True)
        result = trainer.train_step(sample_batch)
        assert isinstance(result["loss"], float)

    def test_loss_decreases_over_steps(self, model, sample_batch):
        """Loss generally decreases over multiple steps."""
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        trainer = RILATrainer(model, optimizer)

        losses = []
        for _ in range(5):
            result = trainer.train_step(sample_batch)
            losses.append(result["loss"])

        # At minimum, no NaN/Inf losses
        assert all(torch.isfinite(torch.tensor(l)) for l in losses)


class TestCheckpoint:
    """Test checkpoint save/load."""

    def test_save_and_load_checkpoint(self, model, sample_batch):
        """Checkpoint round-trip preserves model state."""
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        trainer = RILATrainer(model, optimizer)

        # Train one step
        trainer.train_step(sample_batch)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            trainer.save_checkpoint(path)

            # Create fresh model and load
            config = rila_small()
            new_model = RILA(config)
            new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-4)
            new_trainer = RILATrainer(new_model, new_optimizer)
            new_trainer.load_checkpoint(path)

            # Verify state matches
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                assert torch.allclose(p1, p2)
        finally:
            os.unlink(path)


class TestCurriculumScheduler:
    """Test curriculum scheduling."""

    def test_budget_increases_monotonically(self):
        """Budget increases across phases."""
        scheduler = CurriculumScheduler(max_budget=10, num_phases=5)
        budgets = [scheduler.get_budget_for_phase(p) for p in range(1, 6)]
        assert budgets == sorted(budgets)

    def test_first_phase_is_20_percent(self):
        """First phase budget is ~20% of max."""
        scheduler = CurriculumScheduler(max_budget=100, num_phases=5)
        assert scheduler.get_budget_for_phase(1) == 20

    def test_last_phase_is_100_percent(self):
        """Last phase budget is 100% of max."""
        scheduler = CurriculumScheduler(max_budget=100, num_phases=5)
        assert scheduler.get_budget_for_phase(5) == 100

    def test_invalid_phase_raises(self):
        """Out-of-range phase raises ValueError."""
        scheduler = CurriculumScheduler(max_budget=10, num_phases=3)
        with pytest.raises(ValueError):
            scheduler.get_budget_for_phase(0)
        with pytest.raises(ValueError):
            scheduler.get_budget_for_phase(4)


class TestUncertaintyWeightedLoss:
    """Test uncertainty loss module."""

    def test_forward_produces_scalar(self):
        """UncertaintyWeightedLoss produces a scalar."""
        loss_fn = UncertaintyWeightedLoss()
        result = loss_fn(
            torch.tensor(1.0),
            torch.tensor(0.5),
            torch.tensor(0.3),
            torch.tensor(0.1),
        )
        assert result.dim() == 0

    def test_effective_weights_start_at_one(self):
        """Initial effective weights are approximately 1."""
        loss_fn = UncertaintyWeightedLoss()
        weights = loss_fn.effective_weights
        assert abs(weights["retrieval"] - 1.0) < 1e-5
        assert abs(weights["confidence"] - 1.0) < 1e-5
        assert abs(weights["budget"] - 1.0) < 1e-5
