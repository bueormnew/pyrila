"""pyrila — Recursive Indexed Language Architecture.

A production-ready PyTorch implementation of the RILA architecture for
contextual indexing, adaptive retrieval, recursive reasoning, and
autonomous verification in language models.

The architecture is tokenizer-agnostic: pass any integer token IDs
(from BPE, SentencePiece, tiktoken, char-level, etc.) as input.
Configure pad/bos/eos token IDs in RILAConfig to match your tokenizer.

Save/load models natively:
    model.save("./my_model")
    model = RILA.load("./my_model")
"""

from pyrila.config import RILAConfig
from pyrila.exceptions import (
    PyRILAError,
    ConfigurationError,
    IndexNotBuiltError,
    TrainingError,
    DivergenceError,
    GradientError,
    GenerationError,
    TeacherForcingError,
    SequenceTooLongError,
    BudgetExhaustedError,
    CheckpointError,
)
from pyrila.model import RILA
from pyrila.presets import rila_base, rila_large, rila_small, rila_xl
from pyrila.training import CurriculumScheduler, RILATrainer

__version__ = "0.3.0"

__all__ = [
    "RILA",
    "RILAConfig",
    "RILATrainer",
    "CurriculumScheduler",
    "rila_small",
    "rila_base",
    "rila_large",
    "rila_xl",
    "PyRILAError",
    "ConfigurationError",
    "IndexNotBuiltError",
    "TrainingError",
    "DivergenceError",
    "GradientError",
    "GenerationError",
    "TeacherForcingError",
    "SequenceTooLongError",
    "BudgetExhaustedError",
    "CheckpointError",
]
