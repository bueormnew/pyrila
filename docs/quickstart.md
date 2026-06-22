# Quick Start

## Installation

```bash
pip install pyrila
```

For development:
```bash
pip install -e ".[dev]"
```

## Create a Model

```python
from pyrila import RILA, rila_small

# Use a preset config
config = rila_small()  # ~2M params, good for testing
model = RILA(config)
```

## Training

```python
import torch
from pyrila import RILA, RILATrainer, rila_small

config = rila_small()
model = RILA(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
trainer = RILATrainer(model, optimizer)

# Single training step
batch = {
    "input_ids": torch.randint(0, config.vocab_size, (2, 128)),
    "target_ids": torch.randint(0, config.vocab_size, (2, 32)),
}
result = trainer.train_step(batch)
print(f"Loss: {result['loss']:.4f}")
```

## Generation

```python
model.eval()
input_ids = torch.randint(0, config.vocab_size, (1, 64))
generated = model.generate(input_ids, max_length=50, temperature=0.8)
print(f"Generated {generated.shape[1]} tokens")
```

## Save and Load Models

### Using the Trainer (recommended)

```python
from pyrila import RILA, RILATrainer, rila_small

config = rila_small()
model = RILA(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
trainer = RILATrainer(model, optimizer)

# Train...
trainer.train_step(batch)

# Save checkpoint (includes model + optimizer state)
trainer.save_checkpoint("checkpoint.pt")

# Load into a fresh model
new_model = RILA(config)
new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-4)
new_trainer = RILATrainer(new_model, new_optimizer)
new_trainer.load_checkpoint("checkpoint.pt")
```

### Using PyTorch directly

```python
import torch

# Save just the model weights
torch.save(model.state_dict(), "model_weights.pt")

# Load weights into a new model
new_model = RILA(config)
new_model.load_state_dict(torch.load("model_weights.pt", weights_only=True))
```

## Curriculum Training

Curriculum training progressively increases the reasoning budget across training phases, helping the model learn basic capabilities before attempting complex multi-step reasoning.

```python
from pyrila import RILA, RILATrainer, CurriculumScheduler, rila_small

config = rila_small()
model = RILA(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
trainer = RILATrainer(model, optimizer)

# Create scheduler: 5 phases, max budget of 10
scheduler = CurriculumScheduler(max_budget=10, num_phases=5)

# Phase 1: budget = 2 (simple tasks, no deep reasoning)
# Phase 2: budget = 4
# Phase 3: budget = 6
# Phase 4: budget = 8
# Phase 5: budget = 10 (full reasoning depth)

for phase in range(1, 6):
    budget = scheduler.get_budget_for_phase(phase)
    print(f"Phase {phase}: reasoning_budget = {budget}")
    
    for batch in phase_dataset:
        batch["reasoning_budget"] = budget
        result = trainer.train_step(batch)
```

## Configuring Reasoning Budget

The reasoning budget controls how many iterative reasoning cycles the model can perform when its initial confidence is below the acceptance threshold.

```python
from pyrila import RILA, RILAConfig

# Higher budget = deeper reasoning but slower inference
config = RILAConfig(
    vocab_size=1000,
    hidden_dim=64,
    cognitive_dim=128,
    knowledge_dim=64,
    num_heads=4,
    num_layers=2,
    cell_size=64,
    max_sequence_length=4096,
    default_budget=10,       # Default cycles per forward pass
    max_budget=100,          # Hard cap on reasoning depth
    confidence_threshold=0.8, # Accept output when confidence >= 0.8
    _skip_validation=True,
)

model = RILA(config)

# Override budget at inference time
output = model(input_ids, reasoning_budget=20)
print(f"Budget used: {output['budget_used']} cycles")
print(f"Confidence: {output['confidence'].mean().item():.4f}")
```

**Guidelines:**
- Start with `default_budget=3-5` for prototyping
- Use `default_budget=10-20` for production
- Higher budgets help with complex multi-step reasoning but increase latency
- The model may terminate early if confidence threshold is reached

## Troubleshooting Common Errors

### ConfigurationError

```python
from pyrila import RILAConfig, ConfigurationError

try:
    config = RILAConfig(hidden_dim=100, num_heads=12)  # 100 not divisible by 12
except ConfigurationError as e:
    print(e)
    # "Invalid configuration for 'hidden_dim' = 100. Must be divisible by num_heads (12). Valid range: multiples of 12."
    print(e.param_name)  # "hidden_dim"
    print(e.value)       # 100
```

### IndexNotBuiltError

```python
from pyrila.modules import RecursiveContextIndex
from pyrila.exceptions import IndexNotBuiltError

index = RecursiveContextIndex(config)
try:
    index.query(query_vector, top_k=5)  # Forgot to build_index()!
except IndexNotBuiltError as e:
    print(e)
    # "Cannot query: index has not been built yet. Call model.context_index.build_index(cell_encodings) first..."
```

### SequenceTooLongError

```python
from pyrila.exceptions import SequenceTooLongError

try:
    # Input exceeds max_sequence_length
    output = model(very_long_input_ids)
except SequenceTooLongError as e:
    print(f"Input length: {e.actual_length}, max allowed: {e.max_length}")
    # Truncate and retry
    truncated = very_long_input_ids[:, :e.max_length]
    output = model(truncated)
```

### DivergenceError

```python
from pyrila.exceptions import DivergenceError

try:
    for batch in dataset:
        trainer.train_step(batch)
except DivergenceError as e:
    print(f"Diverged at batches: {e.batch_numbers}")
    print(f"Loss values: {e.loss_values}")
    # Reduce learning rate and retry
```

### Catching all pyrila errors

```python
from pyrila import PyRILAError

try:
    output = model(input_ids, target_ids=targets)
except PyRILAError as e:
    print(f"pyrila error: {e}")
```

## Next Steps

- See [Configuration Guide](guides/configuration.md) for all hyperparameters
- See [Training Guide](guides/training_guide.md) for advanced training
- See [Architecture](../docs/ARCHITECTURE.md) for how the pipeline works
- See [API Reference](api/model.md) for full API docs
