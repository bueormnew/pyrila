"""Curriculum Scheduler for progressive budget training.

Linearly increases the reasoning budget from 20% (phase 1) to 100%
(final phase), encouraging the model to first learn basic capabilities
before tackling complex multi-step reasoning.
"""

from __future__ import annotations


class CurriculumScheduler:
    """Progressive budget scheduling across training phases.

    Args:
        max_budget: Maximum reasoning budget at full capacity.
        num_phases: Number of training phases in [2, 10].

    Raises:
        ValueError: If parameters are out of range.

    Example:
        >>> scheduler = CurriculumScheduler(max_budget=10, num_phases=5)
        >>> [scheduler.get_budget_for_phase(p) for p in range(1, 6)]
        [2, 4, 6, 8, 10]
    """

    def __init__(self, max_budget: int, num_phases: int) -> None:
        if not (2 <= num_phases <= 10):
            raise ValueError(f"num_phases must be in [2, 10], got {num_phases}")
        if max_budget < 1:
            raise ValueError(f"max_budget must be >= 1, got {max_budget}")

        self.max_budget = max_budget
        self.num_phases = num_phases
        self._start_fraction = 0.2
        self._end_fraction = 1.0

    def get_budget_for_phase(self, phase: int) -> int:
        """Get reasoning budget for a given phase (1-indexed).

        Args:
            phase: Current phase in [1, num_phases].

        Returns:
            Integer reasoning budget (at least 1).

        Raises:
            ValueError: If phase is out of range.
        """
        if not (1 <= phase <= self.num_phases):
            raise ValueError(f"phase must be in [1, {self.num_phases}], got {phase}")

        t = (phase - 1) / (self.num_phases - 1)
        fraction = self._start_fraction + t * (self._end_fraction - self._start_fraction)
        return max(1, int(fraction * self.max_budget))
