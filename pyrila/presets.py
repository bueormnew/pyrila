"""Predefined RILA configurations for common use cases.

Each preset produces a model of a specific approximate parameter count,
suitable for different stages of development and deployment.
"""

from pyrila.config import RILAConfig


def rila_small() -> RILAConfig:
    """~2M parameter configuration for testing and rapid prototyping.

    Uses minimal dimensions while maintaining full architecture fidelity.
    Suitable for unit tests, CI pipelines, and architecture validation.

    Returns:
        RILAConfig configured for ~2M parameters.
    """
    return RILAConfig(
        vocab_size=1000,
        hidden_dim=64,
        cognitive_dim=128,
        knowledge_dim=64,
        num_heads=4,
        num_layers=2,
        cell_size=64,
        max_sequence_length=4096,
        initial_top_k=4,
        max_retrieval_expansions=2,
        max_knowledge_units_per_cell=4,
        max_cognitive_iterations=3,
        max_hypotheses=2,
        default_budget=3,
        max_budget=10,
        dropout=0.1,
        attention_dropout=0.1,
        _skip_validation=True,
    )


def rila_base() -> RILAConfig:
    """~125M parameter configuration for research experiments.

    Balanced dimensions for meaningful training runs on single-GPU setups.

    Returns:
        RILAConfig configured for ~125M parameters.
    """
    return RILAConfig(
        vocab_size=32000,
        hidden_dim=768,
        cognitive_dim=1024,
        knowledge_dim=512,
        num_heads=12,
        num_layers=12,
        cell_size=512,
        max_sequence_length=131072,
        initial_top_k=8,
        max_retrieval_expansions=5,
        max_knowledge_units_per_cell=16,
        max_cognitive_iterations=8,
        max_hypotheses=4,
        default_budget=10,
        max_budget=1000,
        dropout=0.1,
        attention_dropout=0.1,
    )


def rila_large() -> RILAConfig:
    """~350M parameter configuration for production workloads.

    Larger dimensions for higher-quality outputs. Requires multi-GPU
    or high-memory single-GPU training.

    Returns:
        RILAConfig configured for ~350M parameters.
    """
    return RILAConfig(
        vocab_size=50000,
        hidden_dim=1024,
        cognitive_dim=1536,
        knowledge_dim=768,
        num_heads=16,
        num_layers=24,
        cell_size=512,
        max_sequence_length=131072,
        initial_top_k=12,
        max_retrieval_expansions=8,
        max_knowledge_units_per_cell=24,
        max_cognitive_iterations=12,
        max_hypotheses=6,
        default_budget=20,
        max_budget=2000,
        dropout=0.1,
        attention_dropout=0.1,
    )


def rila_xl() -> RILAConfig:
    """~1.3B parameter configuration for maximum capability.

    Large-scale configuration requiring distributed training infrastructure.

    Returns:
        RILAConfig configured for ~1.3B parameters.
    """
    return RILAConfig(
        vocab_size=50000,
        hidden_dim=2048,
        cognitive_dim=2560,
        knowledge_dim=1024,
        num_heads=32,
        num_layers=36,
        cell_size=512,
        max_sequence_length=131072,
        initial_top_k=16,
        max_retrieval_expansions=10,
        max_knowledge_units_per_cell=32,
        max_cognitive_iterations=16,
        max_hypotheses=8,
        default_budget=50,
        max_budget=5000,
        dropout=0.05,
        attention_dropout=0.05,
    )
