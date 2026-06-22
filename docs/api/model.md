# Model API Reference

## `pyrila.RILA`

The top-level RILA model orchestrating the full processing pipeline.

### Constructor

```python
from pyrila import RILA, RILAConfig

model = RILA(config: RILAConfig)
```

**Parameters:**
- `config` (`RILAConfig`): Complete architecture configuration.

### `forward()`

```python
outputs = model(
    input_ids: torch.Tensor,          # (batch, seq_len)
    attention_mask: torch.Tensor = None,  # (batch, seq_len)
    target_ids: torch.Tensor = None,      # (batch, target_len)
    reasoning_budget: int = None,
) -> Dict[str, Any]
```

**Returns dict with:**
- `logits`: `(batch, target_len, vocab_size)` if target_ids given, else `(batch, gen_len)` token IDs
- `confidence`: `(batch, 1)` confidence score in [0, 1]
- `retrieval_stats`: dict with `num_cells_retrieved`, `num_retrieval_steps`
- `budget_used`: int reasoning cycles consumed
- `cognitive_state`: `(batch, cognitive_dim)` final cognitive state

### `generate()`

```python
tokens = model.generate(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor = None,
    max_length: int = 512,
    reasoning_budget: int = None,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
) -> torch.Tensor  # (batch, generated_len)
```

Runs the full pipeline in eval mode and returns generated token IDs.

### Train/Eval Mode

```python
model.train()  # Enable training mode (soft top-k, dropout)
model.eval()   # Enable inference mode (hard top-k, no dropout)
```

See also: [Config](config.md), [Training](training.md)
