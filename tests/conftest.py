"""Shared test fixtures for pyrila test suite."""

import torch
import pytest

from pyrila import RILA, RILAConfig
from pyrila.presets import rila_small


@pytest.fixture
def small_config() -> RILAConfig:
    """Small config for fast test execution."""
    return rila_small()


@pytest.fixture
def model(small_config: RILAConfig) -> RILA:
    """Small RILA model instance."""
    return RILA(small_config)


@pytest.fixture
def sample_batch(small_config: RILAConfig):
    """Sample training batch."""
    return {
        "input_ids": torch.randint(0, small_config.vocab_size, (2, 128)),
        "target_ids": torch.randint(0, small_config.vocab_size, (2, 32)),
        "attention_mask": torch.ones(2, 128, dtype=torch.long),
    }


@pytest.fixture
def device():
    """Available compute device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
