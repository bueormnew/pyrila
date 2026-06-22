"""Custom exception hierarchy for pyrila.

All pyrila exceptions inherit from PyRILAError for easy catching.
Each exception provides a clear explanation of what went wrong and how to fix it.
"""


class PyRILAError(Exception):
    """Base exception for all pyrila errors."""
    pass


class ConfigurationError(PyRILAError):
    """Raised when RILAConfig has invalid parameters.
    
    Examples:
        - hidden_dim not divisible by num_heads
        - Parameters out of valid range
        - Incompatible dimension combinations
    """
    def __init__(self, param_name: str, value, reason: str, valid_range: str = ""):
        msg = f"Invalid configuration for '{param_name}' = {value}. {reason}"
        if valid_range:
            msg += f" Valid range: {valid_range}."
        super().__init__(msg)
        self.param_name = param_name
        self.value = value


class IndexNotBuiltError(PyRILAError):
    """Raised when querying or expanding from an index that hasn't been built.
    
    Fix: Call index.build_index(cell_encodings) before query() or expand_from().
    """
    def __init__(self, operation: str = "query"):
        msg = (
            f"Cannot {operation}: index has not been built yet. "
            f"Call model.context_index.build_index(cell_encodings) first, "
            f"or use model.forward() which builds the index automatically."
        )
        super().__init__(msg)


class TrainingError(PyRILAError):
    """Raised during training when an unrecoverable error occurs."""
    pass


class DivergenceError(TrainingError):
    """Raised when training loss diverges (exceeds threshold for consecutive batches).
    
    Common causes:
        - Learning rate too high
        - Gradient explosion
        - Data preprocessing issues
    """
    def __init__(self, batch_numbers: list, loss_values: list, threshold: float = 1e6):
        msg = (
            f"Training divergence detected: loss exceeded {threshold:.0e} for "
            f"{len(batch_numbers)} consecutive batches.\n"
            f"  Batches: {batch_numbers}\n"
            f"  Losses: {[f'{v:.2f}' for v in loss_values]}\n"
            f"Suggestions:\n"
            f"  - Reduce learning rate\n"
            f"  - Enable gradient clipping (max_grad_norm=1.0)\n"
            f"  - Check data preprocessing"
        )
        super().__init__(msg)
        self.batch_numbers = batch_numbers
        self.loss_values = loss_values


class GradientError(TrainingError):
    """Raised when gradients contain NaN or Inf values.
    
    The affected batch is automatically discarded and gradient clipping is activated.
    """
    def __init__(self, param_name: str = ""):
        msg = "NaN/Inf detected in gradients"
        if param_name:
            msg += f" (first occurrence: {param_name})"
        msg += ". Batch discarded, gradient clipping activated for subsequent batches."
        super().__init__(msg)
        self.param_name = param_name


class GenerationError(PyRILAError):
    """Raised when text generation encounters an error."""
    pass


class TeacherForcingError(GenerationError):
    """Raised when training mode is active but no target tokens are provided.
    
    Fix: Either provide target_ids or set model.eval() for inference mode.
    """
    def __init__(self):
        msg = (
            "Teacher forcing requires target_tokens during training mode. "
            "Either:\n"
            "  1. Provide target_ids: model(input_ids, target_ids=targets)\n"
            "  2. Switch to eval mode: model.eval() then model(input_ids)"
        )
        super().__init__(msg)


class SequenceTooLongError(PyRILAError):
    """Raised when input sequence exceeds max_sequence_length.
    
    Fix: Truncate input or increase config.max_sequence_length.
    """
    def __init__(self, actual_length: int, max_length: int):
        msg = (
            f"Input sequence length ({actual_length}) exceeds "
            f"max_sequence_length ({max_length}). "
            f"Either truncate the input or create a config with "
            f"max_sequence_length >= {actual_length}."
        )
        super().__init__(msg)
        self.actual_length = actual_length
        self.max_length = max_length


class BudgetExhaustedError(PyRILAError):
    """Informational: reasoning budget was exhausted without reaching confidence threshold.
    
    This is not necessarily an error — the system returns the best output found.
    Increase reasoning_budget for deeper reasoning.
    """
    def __init__(self, budget: int, best_confidence: float, threshold: float):
        msg = (
            f"Reasoning budget exhausted ({budget} cycles) without reaching "
            f"confidence threshold ({threshold}). Best confidence: {best_confidence:.4f}. "
            f"The best pre-output was accepted. To improve: increase reasoning_budget."
        )
        super().__init__(msg)
        self.budget = budget
        self.best_confidence = best_confidence
        self.threshold = threshold


class CheckpointError(PyRILAError):
    """Raised when saving or loading a checkpoint fails."""
    def __init__(self, path: str, operation: str = "load", reason: str = ""):
        msg = f"Failed to {operation} checkpoint at '{path}'"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)
        self.path = path
