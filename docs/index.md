# pyrila Documentation

Welcome to the pyrila documentation — the production-ready PyTorch implementation of the **RILA** (Recursive Indexed Language Architecture).

## Overview

RILA is a novel language model architecture that combines:

- **Contextual Indexing** — Input is organized into Context Cells with a dynamic affinity graph
- **Adaptive Retrieval** — The model learns what, when, and how much context to retrieve
- **Recursive Reasoning** — Internal reasoning loop with budget control
- **Self-Verification** — Multi-dimensional confidence evaluation before output

## Contents

- [Quick Start](quickstart.md)
- **API Reference**
  - [Model](api/model.md)
  - [Config](api/config.md)
  - [Modules](api/modules.md)
  - [Training](api/training.md)
- **Guides**
  - [Training Guide](guides/training_guide.md)
  - [Inference Guide](guides/inference_guide.md)
  - [Configuration](guides/configuration.md)

## Installation

```bash
pip install pyrila
```

## Minimum Example

```python
import torch
from pyrila import RILA, rila_small

model = RILA(rila_small())
input_ids = torch.randint(0, 1000, (1, 128))
output = model.generate(input_ids, max_length=50)
```
