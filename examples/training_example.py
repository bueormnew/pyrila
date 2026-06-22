"""Training example: create model, dummy data, train 3 steps."""

import torch
from pyrila import RILA, RILATrainer, CurriculumScheduler, rila_small

# Setup
config = rila_small()
model = RILA(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# Create trainer with uncertainty weighting
trainer = RILATrainer(
    model, optimizer,
    use_uncertainty_weighting=True,
    max_grad_norm=1.0,
)

# Create dummy training data
def make_batch(batch_size=2, seq_len=128, target_len=32):
    return {
        "input_ids": torch.randint(0, config.vocab_size, (batch_size, seq_len)),
        "target_ids": torch.randint(0, config.vocab_size, (batch_size, target_len)),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
    }

# Train 3 steps
print("Training RILA model...")
for step in range(3):
    batch = make_batch()
    result = trainer.train_step(batch)
    print(
        f"  Step {step + 1}: loss={result['loss']:.4f} "
        f"(answer={result['l_answer']:.4f}, retrieval={result['l_retrieval']:.4f}, "
        f"conf={result['l_confidence']:.4f}, budget={result['l_budget']:.4f})"
    )

# Check learned loss weights
if trainer.uncertainty_loss:
    weights = trainer.uncertainty_loss.effective_weights
    print(f"\nLearned loss weights: {weights}")

# Curriculum scheduling demo
print("\nCurriculum schedule (max_budget=10, 5 phases):")
scheduler = CurriculumScheduler(max_budget=10, num_phases=5)
for phase in range(1, 6):
    budget = scheduler.get_budget_for_phase(phase)
    print(f"  Phase {phase}: budget = {budget}")

# Save checkpoint
trainer.save_checkpoint("rila_checkpoint.pt")
print("\nCheckpoint saved to rila_checkpoint.pt")
