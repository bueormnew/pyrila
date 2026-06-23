# pyrila

**RILA: Recursive Indexed Language Architecture**

A production-ready PyTorch implementation of the RILA architecture — a novel language model that combines contextual indexing, adaptive retrieval, recursive internal reasoning, and autonomous result verification.

## Key Features

- **Contextual Indexing** — Organizes input into Context Cells with a dynamic affinity graph for fast retrieval
- **Adaptive Retrieval** — Learns what, when, and how much context to retrieve without specialized datasets
- **Recursive Reasoning** — Internal reasoning loop with budget control, no intermediate text generation
- **Self-Verification** — Multi-dimensional confidence evaluation before committing to output
- **Scale-Invariant** — Same architecture from 2M to 1.3B+ parameters

## How RILA Differs from Traditional Transformers

Traditional transformers apply dense attention over all tokens at every layer, scaling quadratically with context length. RILA replaces this with structured context cells and a learned affinity graph — only relevant cells are retrieved and processed deeply, enabling sub-quadratic scaling. The recursive reasoning loop allows the model to iteratively refine its understanding with autonomous budget control, something fixed-depth transformers cannot do.

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                     RILA PIPELINE                                   │
│                                                                    │
│  Input Tokens                                                      │
│       │                                                            │
│       ▼                                                            │
│  [Tokenizer] → [Cell Builder] → [Cell Encoder]                    │
│                                       │                            │
│                                       ▼                            │
│                              [Context Index]  (affinity graph)     │
│                                       │                            │
│                                       ▼                            │
│                         [Recursive Context Explorer]               │
│                                       │                            │
│                                       ▼                            │
│                             [Working Context]                      │
│                                       │                            │
│                                       ▼                            │
│                        [Core Language Processor]                   │
│                           (iterative GRU + convergence)            │
│                                       │                            │
│                                       ▼                            │
│                         [Pre-Output Generator]                     │
│                                       │                            │
│                                       ▼                            │
│                   [Recursive Verification Engine]                  │
│                          confidence >= τ ?                         │
│                         /              \                           │
│                       Yes               No                        │
│                        │                 │                         │
│                        │      [Reasoning Loop]                    │
│                        │      (budget-limited)                    │
│                        │           │                              │
│                        ▼           ▼                              │
│                     [Final Decoder]                                │
│                          │                                        │
│                          ▼                                        │
│                    Output Tokens                                   │
└────────────────────────────────────────────────────────────────────┘
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full architecture explanation with mathematical formulas.

## Installation

```bash
pip install pyrila
```

For development:

```bash
git clone https://github.com/bueormnew/pyrila.git
cd pyrila
pip install -e ".[dev]"
```

## Quick Start

```python
import torch
from pyrila import RILA, RILAConfig, rila_small

# Create a small model for testing
config = rila_small()
model = RILA(config)

# Forward pass with teacher forcing
input_ids = torch.randint(0, config.vocab_size, (1, 1024))
target_ids = torch.randint(0, config.vocab_size, (1, 64))
outputs = model(input_ids, target_ids=target_ids)

print(f"Logits shape: {outputs['logits'].shape}")
print(f"Confidence: {outputs['confidence'].mean().item():.4f}")
print(f"Budget used: {outputs['budget_used']} cycles")
```

## Preset Configurations

| Preset | Parameters | Use Case |
|--------|-----------|----------|
| `rila_small()` | ~2M | Testing and prototyping |
| `rila_base()` | ~125M | Research experiments |
| `rila_large()` | ~350M | Production workloads |
| `rila_xl()` | ~1.3B | Maximum capability |

## Training

```python
from pyrila import RILA, RILATrainer, rila_small

config = rila_small()
model = RILA(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
trainer = RILATrainer(model, optimizer, use_uncertainty_weighting=True)

batch = {
    "input_ids": torch.randint(0, config.vocab_size, (2, 1024)),
    "target_ids": torch.randint(0, config.vocab_size, (2, 64)),
}
result = trainer.train_step(batch)
print(f"Loss: {result['loss']:.4f}")
```

## Save & Load

```python
# Save model (config + weights)
model.save("./my_model")

# Load model
from pyrila import RILA
model = RILA.load("./my_model", device="cuda")
```

## Tokenizer-Agnostic

RILA accepts any tokenizer. Configure special token IDs in the config:

```python
from pyrila import RILA, RILAConfig

config = RILAConfig(
    vocab_size=32000,       # Match your tokenizer's vocab
    pad_token_id=0,         # Your tokenizer's PAD
    bos_token_id=1,         # Your tokenizer's BOS
    eos_token_id=2,         # Your tokenizer's EOS
    # ... other params
)
model = RILA(config)

# Use any tokenizer (HuggingFace, SentencePiece, tiktoken, etc.)
input_ids = your_tokenizer.encode("Hello world")
output = model(torch.tensor([input_ids]))
```

## Error Handling

pyrila provides a hierarchy of custom exceptions for clear, actionable error messages:

```python
from pyrila import PyRILAError, ConfigurationError, SequenceTooLongError

# All pyrila exceptions inherit from PyRILAError
try:
    output = model(input_ids)
except PyRILAError as e:
    print(f"pyrila error: {e}")

# Specific exceptions provide context
try:
    config = RILAConfig(hidden_dim=100, num_heads=12)
except ConfigurationError as e:
    print(e.param_name, e.value)  # "hidden_dim", 100
```

Available exceptions: `ConfigurationError`, `IndexNotBuiltError`, `DivergenceError`, `GradientError`, `TeacherForcingError`, `SequenceTooLongError`, `BudgetExhaustedError`, `CheckpointError`.

## Documentation

- [Quick Start Guide](docs/quickstart.md)
- [Architecture Deep Dive](docs/ARCHITECTURE.md)
- [API Reference](docs/api/)
- [Training Guide](docs/guides/training_guide.md)
- [Configuration Guide](docs/guides/configuration.md)

## Contributing

Contributions are welcome! To get started:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Install development dependencies: `pip install -e ".[dev]"`
4. Run tests: `pytest tests/ -v`
5. Submit a pull request

Please ensure all tests pass and follow the existing code style.

## Citation

```bibtex
@software{pyrila2024,
  title={pyrila: Recursive Indexed Language Architecture},
  author={RILA Team},
  year={2024},
  url={https://github.com/bueormnew/pyrila},
  version={0.2.0}
}
```

## License

MIT — see [LICENSE](LICENSE) for details.
