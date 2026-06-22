"""Basic usage: create a RILA model and generate text."""

import torch
from pyrila import RILA, rila_small

# Create a small model for demonstration
config = rila_small()
model = RILA(config)
print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"Config: hidden_dim={config.hidden_dim}, cognitive_dim={config.cognitive_dim}")

# Prepare input
input_ids = torch.randint(0, config.vocab_size, (1, 128))
print(f"\nInput shape: {input_ids.shape}")

# Generate tokens (inference mode)
model.eval()
with torch.no_grad():
    generated = model.generate(input_ids, max_length=20, temperature=0.8)
    print(f"Generated shape: {generated.shape}")
    print(f"Generated tokens: {generated[0].tolist()[:20]}")

# Forward pass (training mode)
model.train()
target_ids = torch.randint(0, config.vocab_size, (1, 32))
outputs = model(input_ids, target_ids=target_ids)

print(f"\nTraining outputs:")
print(f"  Logits shape: {outputs['logits'].shape}")
print(f"  Confidence: {outputs['confidence'].mean().item():.4f}")
print(f"  Cells retrieved: {outputs['retrieval_stats']['num_cells_retrieved']}")
print(f"  Budget used: {outputs['budget_used']}")
