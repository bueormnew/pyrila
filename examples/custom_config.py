"""Custom configuration example: show how to customize all parameters."""

import torch
from pyrila import RILA, RILAConfig
from pyrila.presets import rila_small, rila_base, rila_large, rila_xl

# --- Method 1: Use a preset ---
config = rila_small()
model = RILA(config)
print(f"rila_small: {sum(p.numel() for p in model.parameters()):,} params")

# --- Method 2: Customize a preset ---
config = rila_small()
config.confidence_threshold = 0.9  # Higher acceptance bar
config.default_budget = 5          # More reasoning by default
config.max_retrieval_expansions = 3

# Note: After modifying, re-validate
# config.validate()  # Would raise if invalid

model = RILA(config)
print(f"Customized small: confidence_threshold={config.confidence_threshold}")

# --- Method 3: Full custom config ---
custom_config = RILAConfig(
    vocab_size=5000,
    hidden_dim=128,
    cognitive_dim=256,
    knowledge_dim=128,
    num_heads=4,
    num_layers=4,
    cell_size=128,
    max_sequence_length=8192,
    # Retrieval
    initial_top_k=6,
    max_retrieval_expansions=3,
    retrieval_score_threshold=0.4,
    # CLP
    relevance_gate_threshold=0.3,
    max_knowledge_units_per_cell=8,
    cognitive_stability_lambda=0.2,
    convergence_threshold=0.85,
    max_cognitive_iterations=5,
    # Hypothesis
    max_hypotheses=3,
    # RVE
    confidence_threshold=0.75,
    # Budget
    default_budget=8,
    max_budget=100,
    # Training
    dropout=0.05,
    attention_dropout=0.05,
    use_soft_topk=True,
    soft_topk_temperature=0.3,
    # Skip validation for non-standard configs
    _skip_validation=True,
)

model = RILA(custom_config)
params = sum(p.numel() for p in model.parameters())
print(f"Custom config: {params:,} params")

# --- Method 4: Show all presets ---
print("\nAll presets:")
for name, fn in [("small", rila_small), ("base", rila_base), ("large", rila_large), ("xl", rila_xl)]:
    cfg = fn()
    print(f"  {name}: hidden_dim={cfg.hidden_dim}, cognitive_dim={cfg.cognitive_dim}, "
          f"layers={cfg.num_layers}, heads={cfg.num_heads}")
    print(f"    Estimated params: {cfg.estimate_parameters():,}")
