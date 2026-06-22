"""pyrila — Recursive Indexed Language Architecture.

A production-ready PyTorch implementation of the RILA architecture for
contextual indexing, adaptive retrieval, recursive reasoning, and
autonomous verification in language models.
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

__version__ = "0.1.0"

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
