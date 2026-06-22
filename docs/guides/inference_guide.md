# Inference Guide

## Basic Generation

```python
from pyrila import RILA, rila_small
import torch

model = RILA(rila_small())
model.eval()

input_ids = torch.randint(0, 1000, (1, 128))
generated = model.generate(input_ids, max_length=100)
```

## Generation Parameters

```python
generated = model.generate(
    input_ids,
    max_length=200,        # Maximum tokens to generate
    temperature=0.8,       # Lower = more deterministic
    top_k=50,              # Top-k filtering
    top_p=0.9,             # Nucleus sampling threshold
    reasoning_budget=20,   # Reasoning cycles allowed
)
```

## Batch Generation

```python
# Generate for multiple inputs simultaneously
input_ids = torch.randint(0, 1000, (4, 128))
generated = model.generate(input_ids, max_length=50)
# generated.shape: (4, <=50)
```

## Accessing Intermediate States

Use `forward()` directly to access confidence and retrieval stats:

```python
model.eval()
with torch.no_grad():
    outputs = model(input_ids)

print(f"Confidence: {outputs['confidence'].mean().item():.4f}")
print(f"Cells retrieved: {outputs['retrieval_stats']['num_cells_retrieved']}")
print(f"Cognitive state shape: {outputs['cognitive_state'].shape}")
```

## Controlling Reasoning Depth

```python
# Quick response (minimal reasoning)
generated = model.generate(input_ids, reasoning_budget=0)

# Deep reasoning
generated = model.generate(input_ids, reasoning_budget=100)
```
