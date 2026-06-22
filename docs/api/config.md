# Config API Reference

## `pyrila.RILAConfig`

Central configuration dataclass for the RILA architecture.

### Constructor

```python
from pyrila import RILAConfig

config = RILAConfig(
    vocab_size=32000,
    hidden_dim=768,
    cognitive_dim=1024,
    knowledge_dim=512,
    num_heads=12,
    num_layers=12,
    ...
)
```

### Core Dimensions

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vocab_size` | 32000 | Token vocabulary size |
| `hidden_dim` | 768 | Embedding dimension (D) |
| `cognitive_dim` | 1024 | Cognitive state dimension (H) |
| `knowledge_dim` | 512 | Knowledge unit dimension (K) |
| `num_heads` | 12 | Attention heads |
| `num_layers` | 12 | Transformer layers |

### Context Cell Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cell_size` | 512 | Tokens per cell |
| `max_sequence_length` | 131072 | Maximum input length |

### Retrieval Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_top_k` | 8 | Initial cells to retrieve |
| `max_retrieval_expansions` | 5 | Max expansion iterations |
| `retrieval_score_threshold` | 0.5 | Sufficiency threshold |

### CLP Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `relevance_gate_threshold` | 0.4 | KU filtering threshold |
| `max_knowledge_units_per_cell` | 16 | KU slots per cell |
| `cognitive_stability_lambda` | 0.1 | Max state change norm |
| `convergence_threshold` | 0.9 | Convergence detection |
| `max_cognitive_iterations` | 8 | Max refinement iterations |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `confidence_threshold` | 0.8 | RVE acceptance threshold |
| `default_budget` | 10 | Default reasoning budget |
| `max_budget` | 1000 | Maximum budget |
| `use_soft_topk` | True | Differentiable selection in training |
| `soft_topk_temperature` | 0.5 | Temperature for soft selection |

### Properties

- `max_cells`: Maximum Context Cells (sequence_length // cell_size)
- `cell_encoding_dim`: Full cell encoding dimension (4 * hidden_dim)

### Methods

- `validate()`: Validate all parameters (raises ValueError if invalid)
- `estimate_parameters()`: Estimate total model parameter count

## Presets

```python
from pyrila import rila_small, rila_base, rila_large, rila_xl

config = rila_small()   # ~2M params
config = rila_base()    # ~125M params
config = rila_large()   # ~350M params
config = rila_xl()      # ~1.3B params
```

See also: [Model](model.md), [Training](training.md)
