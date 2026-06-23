"""RILA configuration with full validation and derived properties.

Central configuration holding all hyperparameters for the RILA architecture,
supporting models from ~2M to 1.3B+ parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

from pyrila.exceptions import ConfigurationError


@dataclass
class RILAConfig:
    """Central configuration for the RILA architecture.

    All architecture components read their hyperparameters from this single
    config object. The architecture structure remains identical regardless
    of scale — only numeric parameters change.

    Args:
        vocab_size: Vocabulary size for token embeddings.
        hidden_dim: Internal embedding dimension (D).
        cognitive_dim: Cognitive state dimension (H).
        knowledge_dim: Knowledge unit dimension (K).
        num_heads: Number of attention heads.
        num_layers: Number of transformer layers.
        cell_size: Tokens per Context Cell (C).
        max_sequence_length: Maximum input sequence length.
        index_affinity_threshold: Threshold for affinity graph edges.
        max_index_connections: Maximum edges per node in the index graph.
        initial_top_k: Initial cells to retrieve.
        max_retrieval_expansions: Maximum retrieval expansion iterations.
        retrieval_score_threshold: Sufficiency threshold for retrieval.
        relevance_gate_threshold: Threshold for knowledge unit filtering.
        max_knowledge_units_per_cell: KU slots per cell extraction.
        cognitive_stability_lambda: Max cognitive state change norm per iteration.
        convergence_threshold: CLP convergence detection threshold.
        max_cognitive_iterations: Maximum CLP refinement iterations.
        max_hypotheses: Number of candidate hypotheses to generate.
        hypothesis_competition_temperature: Temperature for hypothesis scoring.
        confidence_threshold: RVE acceptance threshold (tau).
        default_budget: Default reasoning budget (B).
        max_budget: Maximum allowed reasoning budget.
        dropout: General dropout rate.
        attention_dropout: Attention-specific dropout rate.
        use_soft_topk: Use differentiable soft selection during training.
        soft_topk_temperature: Temperature for soft top-k (lower = more peaked).
    """

    # Core dimensions
    vocab_size: int = 32000
    hidden_dim: int = 768
    cognitive_dim: int = 1024
    knowledge_dim: int = 512
    num_heads: int = 12
    num_layers: int = 12

    # Context Cell parameters
    cell_size: int = 512
    max_sequence_length: int = 131072

    # Recursive Context Index
    index_affinity_threshold: float = 0.3
    max_index_connections: int = 64

    # RCE parameters
    initial_top_k: int = 8
    max_retrieval_expansions: int = 5
    retrieval_score_threshold: float = 0.5

    # CLP parameters
    relevance_gate_threshold: float = 0.4
    max_knowledge_units_per_cell: int = 16
    cognitive_stability_lambda: float = 0.1
    convergence_threshold: float = 0.9
    max_cognitive_iterations: int = 8

    # Hypothesis parameters
    max_hypotheses: int = 4
    hypothesis_competition_temperature: float = 0.7

    # RVE parameters
    confidence_threshold: float = 0.8

    # Reasoning Budget
    default_budget: int = 10
    max_budget: int = 1000

    # Dropout and regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1

    # Training optimization
    use_soft_topk: bool = True
    soft_topk_temperature: float = 0.5

    # Special token IDs — configured by the user's tokenizer.
    # The architecture does NOT assume any specific tokenizer.
    # These are used by the decoder to know when to start/stop generation.
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 3

    # Internal flag to skip validation (useful for testing with small configs)
    _skip_validation: bool = False

    @property
    def max_cells(self) -> int:
        """Maximum number of Context Cells derived from sequence length and cell size."""
        return self.max_sequence_length // self.cell_size

    @property
    def cell_encoding_dim(self) -> int:
        """Dimension of complete cell encoding: concat(S, R, T, I) = 4 * hidden_dim."""
        return self.hidden_dim * 4

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        if not self._skip_validation:
            self.validate()

    def validate(self) -> None:
        """Validate all configuration parameters.

        Raises:
            ConfigurationError: If any parameter is out of valid range or violates constraints.
        """
        if self.hidden_dim % self.num_heads != 0:
            raise ConfigurationError(
                "hidden_dim", self.hidden_dim,
                f"Must be divisible by num_heads ({self.num_heads})",
                f"multiples of {self.num_heads}",
            )

        if self.hidden_dim < 32:
            raise ConfigurationError("hidden_dim", self.hidden_dim, "Must be >= 32", ">= 32")

        if self.cognitive_dim < 32:
            raise ConfigurationError("cognitive_dim", self.cognitive_dim, "Must be >= 32", ">= 32")

        if self.knowledge_dim < 16:
            raise ConfigurationError("knowledge_dim", self.knowledge_dim, "Must be >= 16", ">= 16")

        if self.num_heads < 1:
            raise ConfigurationError("num_heads", self.num_heads, "Must be >= 1", ">= 1")

        if self.num_layers < 1:
            raise ConfigurationError("num_layers", self.num_layers, "Must be >= 1", ">= 1")

        if self.cell_size < 1:
            raise ConfigurationError("cell_size", self.cell_size, "Must be >= 1", ">= 1")

        if self.vocab_size < 2:
            raise ConfigurationError("vocab_size", self.vocab_size, "Must be >= 2", ">= 2")

        if not (0.0 <= self.dropout <= 1.0):
            raise ConfigurationError("dropout", self.dropout, "Must be in [0, 1]", "[0.0, 1.0]")

        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ConfigurationError(
                "confidence_threshold", self.confidence_threshold,
                "Must be in [0, 1]", "[0.0, 1.0]",
            )

        if self.max_budget < 1:
            raise ConfigurationError("max_budget", self.max_budget, "Must be >= 1", ">= 1")

    def estimate_parameters(self) -> int:
        """Estimate total parameter count for this configuration.

        Returns:
            Approximate number of trainable parameters.
        """
        D = self.hidden_dim
        H = self.cognitive_dim
        K = self.knowledge_dim
        L = self.num_layers
        V = self.vocab_size

        token_embed = V * D
        cell_encoder = 2 * 12 * D * D + 4 * (2 * D * D)
        transformer_layers = L * 12 * D * D
        rce = H * D + D * self.cell_encoding_dim + self.cell_encoding_dim * 2 * D
        working_ctx = self.cell_encoding_dim * self.cell_encoding_dim * 3 + self.cell_encoding_dim * H
        knowledge_ext = self.max_knowledge_units_per_cell * K + D * K + K * K * 3
        clp = H * H * 2 + K * H * 3 + H * H * 4
        pre_output = H * H * 3
        rve = 4 * (H * 4 * H + H) + 4 + H + 1
        decoder = max(1, L // 2) * (12 * D * D) + D * V

        total = (
            token_embed + cell_encoder + transformer_layers
            + rce + working_ctx + knowledge_ext + clp
            + pre_output + rve + decoder
        )
        return int(total)

    def to_dict(self) -> dict:
        """Serialize config to a dictionary (excludes internal fields)."""
        from dataclasses import fields
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if not f.name.startswith("_")
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RILAConfig":
        """Create config from a dictionary."""
        return cls(**{k: v for k, v in d.items() if not k.startswith("_")}, _skip_validation=True)
