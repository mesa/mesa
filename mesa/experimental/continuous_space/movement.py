"""Optional movement mixin for continuous-space agents."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike


class SupportsContinuousMovement(Protocol):
    """Protocol for objects that can use continuous movement helpers."""

    space: object
    position: np.ndarray


class BasicContinuousMovement:
    """Mixin with common movement operations for continuous space agents."""

    def move_by(self: SupportsContinuousMovement, delta: ArrayLike) -> None:
        """Translate the agent position by a vector."""
        self.position = self.position + np.asarray(delta, dtype=float)

    def move_towards(
        self: SupportsContinuousMovement,
        target: ArrayLike,
        distance: float,
        *,
        clamp: bool = True,
    ) -> float:
        """Move toward target by at most ``distance`` and return the moved distance."""
        if distance < 0:
            raise ValueError("distance must be non-negative")

        target = np.asarray(target, dtype=float)
        # calculate_difference_vector returns (agent_position - point); negate to get
        # the shortest vector from the agent to the target.
        delta = -self.space.calculate_difference_vector(target, agents=[self])[0]
        dist = float(np.linalg.norm(delta))

        if dist == 0:
            return 0.0

        step = min(distance, dist) if clamp else distance
        self.position = self.position + (delta / dist) * step
        return step

    def move_in_direction(
        self: SupportsContinuousMovement, direction: ArrayLike, distance: float
    ) -> None:
        """Move along a direction vector scaled to ``distance``."""
        if distance < 0:
            raise ValueError("distance must be non-negative")

        direction = np.asarray(direction, dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm == 0:
            return

        self.position = self.position + (direction / norm) * distance
