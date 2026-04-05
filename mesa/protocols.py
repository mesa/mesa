"""Protocols and utilities for spatial positioning."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

PositionLike = NDArray[np.floating] | tuple[int, ...] | tuple[float, ...]


@runtime_checkable
class Locatable(Protocol):
    """Protocol for any object that has a position in space.

    Satisfied by Cell, CellAgent, FixedAgent, and ContinuousSpaceAgent.
    """

    @property
    def position(self) -> PositionLike | None: 
        """The position of this object in its space."""
        ...


class HasPosition:
    """Mixin providing writable position storage for agents not backed by a space.

    Usage:
        class MyAgent(Agent, HasPosition):
            pass

        agent = MyAgent(model)
        agent.position = np.array([1.0, 2.0])
        assert isinstance(agent, Locatable)  # True
    """

    _position: PositionLike | None = None

    @property
    def position(self) -> PositionLike | None:
        """The position of this object in its space."""
        return self._position

    @position.setter
    def position(self, value: PositionLike | None) -> None:
        self._position = value
