# Configuration Guide

## Presets

pyrila provides four preset configurations:

```python
from pyrila import rila_small, rila_base, rila_large, rila_xl

config = rila_small()   # ~2M params — testing/CI
config = rila_base()    # ~125M params — research
config = rila_large()   # ~350M params — production
config = rila_xl()      # ~1.3B params — maximum capability
```

## Custom Configuration

```python
from pyrila import RILAConfig

config = RILAConfig(
    vocab_size=32000,
    hidden_dim=768,
    cognitive_dim=1024,
    knowledge_dim=512,
    num_heads=12,
    num_layers=12,
    cell_size=512,
    max_sequence_length=131072,
)
```

## Key Parameters

### Dimensions
- `hidden_dim`: Controls token embedding size and cell encoding width
- `cognitive_dim`: Controls reasoning capacity (cognitive state size)
- `knowledge_dim`: Controls knowledge unit granularity

### Retrieval
- `initial_top_k`: More = broader initial retrieval, slower
- `max_retrieval_expansions`: More = deeper context exploration
- `retrieval_score_threshold`: Lower = more aggressive retrieval

### Reasoning
- `confidence_threshold`: Higher = more reasoning before accepting
- `default_budget`: More = deeper default reasoning
- `max_cognitive_iterations`: More = more CLP refinement steps

### Training
- `use_soft_topk`: Enable differentiable selection (better gradients)
- `soft_topk_temperature`: Lower = sharper selection (closer to hard)

## Validation

Configs are validated on creation. To skip (for testing):

```python
config = RILAConfig(..., _skip_validation=True)
```

## Parameter Estimation

```python
config = rila_base()
print(f"Estimated params: {config.estimate_parameters():,}")
```
