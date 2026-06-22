# Training Guide

## Basic Training Loop

```python
import torch
from pyrila import RILA, RILATrainer, rila_small

config = rila_small()
model = RILA(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
trainer = RILATrainer(model, optimizer)

# Your data loading here
for epoch in range(num_epochs):
    for batch in dataloader:
        result = trainer.train_step(batch)
        if result["batch_discarded"]:
            print("Bad gradients detected, batch discarded")
```

## Batch Format

```python
batch = {
    "input_ids": torch.Tensor,       # (batch, seq_len) required
    "target_ids": torch.Tensor,      # (batch, target_len) required
    "attention_mask": torch.Tensor,  # (batch, seq_len) optional
    "reasoning_budget": int,         # optional
}
```

## Uncertainty Weighting

Instead of manually tuning λ₁, λ₂, λ₃, use learned weights:

```python
trainer = RILATrainer(model, optimizer, use_uncertainty_weighting=True)
```

The model automatically learns the optimal balance between loss terms.

## Curriculum Training

Progressively increase the reasoning budget:

```python
from pyrila import CurriculumScheduler

scheduler = CurriculumScheduler(max_budget=50, num_phases=5)

for phase in range(1, 6):
    budget = scheduler.get_budget_for_phase(phase)
    for batch in phase_data:
        batch["reasoning_budget"] = budget
        trainer.train_step(batch)
```

## Gradient Safety

The trainer automatically:
- Detects NaN/Inf gradients → discards the batch
- Activates gradient clipping after first detection
- Detects divergence (loss > 1e6 for 3 consecutive batches) → raises DivergenceError

## Checkpointing

```python
# Save after training
trainer.save_checkpoint("checkpoint.pt")

# Resume later
trainer.load_checkpoint("checkpoint.pt")
```

## GPU Training

```python
device = torch.device("cuda")
model = RILA(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
trainer = RILATrainer(model, optimizer)

# Move batches to GPU
batch = {k: v.to(device) for k, v in batch.items()}
result = trainer.train_step(batch)
```
