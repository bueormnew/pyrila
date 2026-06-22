# Training API Reference

## `pyrila.RILATrainer`

Full training loop with composite loss and gradient safety.

### Constructor

```python
from pyrila import RILATrainer

trainer = RILATrainer(
    model,
    optimizer,
    alpha=0.01,                    # Retrieval cost coefficient
    beta=0.005,                    # Budget cost coefficient
    lambda1=1.0,                   # Retrieval loss weight
    lambda2=1.0,                   # Confidence loss weight
    lambda3=1.0,                   # Budget loss weight
    max_grad_norm=1.0,             # Gradient clipping threshold
    divergence_threshold=1e6,      # Loss divergence detection
    divergence_patience=3,         # Consecutive batches before halt
    use_uncertainty_weighting=False,  # Learned loss weights
)
```

### Methods

#### `train_step(batch) -> Dict`
Single training step. Returns loss components and gradient safety info.

#### `train(dataset, epochs=1) -> List[Dict]`
Full training loop over a dataset (list of batch dicts).

#### `evaluate(dataset) -> Dict`
Evaluation mode returning avg_loss and avg_confidence.

#### `save_checkpoint(path)` / `load_checkpoint(path)`
Model persistence.

## `pyrila.CurriculumScheduler`

Progressive budget scheduling.

```python
from pyrila import CurriculumScheduler

scheduler = CurriculumScheduler(max_budget=10, num_phases=5)
budget = scheduler.get_budget_for_phase(1)  # 2 (20% of max)
budget = scheduler.get_budget_for_phase(5)  # 10 (100% of max)
```

## `pyrila.training.UncertaintyWeightedLoss`

Learned loss weighting (Kendall et al., 2018).

```python
from pyrila.training import UncertaintyWeightedLoss

loss_fn = UncertaintyWeightedLoss()
total = loss_fn(l_answer, l_retrieval, l_confidence, l_budget)
print(loss_fn.effective_weights)  # Current learned weights
```

## Loss Function

The composite loss is:

```
L = L_answer + λ₁·L_retrieval + λ₂·L_confidence + λ₃·L_budget
```

With uncertainty weighting:
```
L = L_answer + (1/2σ²_r)·L_ret + log(σ_r) + (1/2σ²_c)·L_conf + log(σ_c) + ...
```

See also: [Model](model.md), [Config](config.md)
