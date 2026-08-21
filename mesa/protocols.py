"""Protocols and composition mixins for Mesa agents and spaces.

This module provides:
- ``Locatable``: A Protocol defining the position interface (like HasCellProtocol)
- ``HasPosition``: A composition mixin providing writable position (like HasCell)

Design philosophy (aligned with quaquel's vision):
    - Composition over inheritance: agents gain position by mixing in HasPosition,
      not by inheriting it from base Agent.
    - Agent-centric API: ``self.position = new_pos`` is the primary way to set position.
    - Space-specific overrides: CellAgent derives position from its cell,
      ContinuousSpaceAgent computes it from numpy arrays.
    - Protocol for structural typing: any object with a ``position`` property
      satisfies Locatable, regardless of class hierarchy.

Pattern reference (parallel to discrete_space.cell_agent)::

    HasCellProtocol (in cell_agent)  <-->  Locatable (in protocols)
    HasCell (in cell_agent)          <-->  HasPosition (in protocols)
    CellAgent(Agent, HasCell)        <-->  MyAgent(Agent, HasPosition)
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

# Type alias for positions across all space types.
# - tuple[int, ...] for discrete grid coordinates
# - tuple[float, ...] for float-based positions
# - NDArray[np.floating] for continuous spaces backed by numpy
PositionLike = tuple[int, ...] | tuple[float, ...] | NDArray[np.floating]


@runtime_checkable
class Locatable(Protocol):
    """Protocol for any object that has a position in space.

    This is the structural typing interface. Any class that implements
    a ``position`` property satisfies this protocol via duck typing,
    without needing to inherit from it.

    Mirrors the relationship of HasCellProtocol to HasCell:
    - Locatable defines the interface
    - HasPosition provides a default implementation
    - Space-specific agents override as needed

    Examples:
        Using as a type hint for space-agnostic functions::

            from mesa.protocols import Locatable
            import numpy as np

            def get_distance(a: Locatable, b: Locatable) -> float:
                pos_a = np.array(a.position)
                pos_b = np.array(b.position)
                return float(np.linalg.norm(pos_a - pos_b))

            # Works with any combination: Agent-Agent, Agent-Cell, Cell-Cell

        Runtime checking::

            from mesa.protocols import Locatable
            from mesa.discrete_space import Cell

            cell = Cell((3, 4))
            assert isinstance(cell, Locatable)

    """

    @property
    def position(self) -> PositionLike | None:
        """The position of this object in its space.

        Returns:
            The position as a coordinate tuple or numpy array.
            None if the object is not placed in any space.

        """
        ...


class HasPosition:
    """Mixin providing a writable ``position`` attribute via composition.

    This follows the same pattern as ``HasCell`` in mesa.discrete_space.cell_agent:
    a mixin class that manages the position lifecycle through a property with
    getter and setter.

    Usage:
        Mix into your agent class to gain a writable position::

            class MyAgent(Agent, HasPosition):
                pass

            agent = MyAgent(model)
            agent.position = (5.0, 3.0)  # writable!
            assert isinstance(agent, Locatable)  # satisfies protocol

        Space-specific subclasses can override the setter to notify their
        space when position changes (just like HasCell notifies cells)::

            class SpaceAwarePosition(HasPosition):
                @HasPosition.position.setter
                def position(self, value):
                    old = self._position
                    self._position = value
                    if hasattr(self, 'space'):
                        self.space._on_position_changed(self, old, value)

    Notes:
        - Base ``Agent`` does NOT include this mixin. This is intentional:
          only agents that need spatial position should mix it in (composition).
        - ``CellAgent`` does NOT use this mixin either; it derives position
          from ``cell.position`` (np.ndarray) via its own property override.
          See #3268 for the distinction between logical ``coordinate`` and
          physical ``position`` on cells.
        - ``ContinuousSpaceAgent`` has its own position property backed by
          numpy arrays in the space.

    """

    _position: PositionLike | None = None

    @property
    def position(self) -> PositionLike | None:
        """The position of this object in its space."""
        return self._position

    @position.setter
    def position(self, value: PositionLike | None) -> None:
        if value is not None and not isinstance(value, (tuple, np.ndarray)):
            raise TypeError(
                f"position must be a tuple or numpy array, got {type(value).__name__}"
            )
        self._position = value
