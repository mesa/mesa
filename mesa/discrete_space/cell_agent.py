"""Agents that understand how to exist in and move through cell spaces.

Provides specialized agent classes that handle cell occupation, movement, and
proper registration:
- CellAgent: Mobile agents that can move between cells
- FixedAgent: Immobile agents permanently fixed to cells
- Grid2DMovingAgent: Agents with grid-specific movement capabilities

These classes ensure consistent agent-cell relationships and proper state management
as agents move through the space. They can be used directly or as examples for
creating custom cell-aware agents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from mesa.agent import Agent

if TYPE_CHECKING:
    from mesa.discrete_space import Cell


class HasCellProtocol(Protocol):
    """Protocol for discrete space cell holders."""

    cell: Cell


class HasCell:
    """Descriptor for agents have a cell."""

    def __get__(self, obj: Agent, type=None) -> Cell | None:  # noqa: D105
        try:
            return getattr(obj, self._private_name)
        except AttributeError:
            return None

    def __set__(self, obj: Agent, value: Cell) -> None:  # noqa: D105
        try:
            current_cell = getattr(obj, self._private_name)
        except AttributeError:
            current_cell = None

        # remove from current cell
        if current_cell is not None:
            current_cell.remove_agent(obj)

        setattr(obj, self._private_name, value)

        # add to new cell
        if value is not None:
            value.add_agent(obj)

    def __set_name__(self, owner: Agent, name) -> None:  # noqa: D105
        self._private_name = f"_{name}"


class BasicMovement:
    """Mixin for moving agents in discrete space."""

    def move_to(self: HasCellProtocol, cell: Cell) -> None:
        """Move to a new cell."""
        self.cell = cell

    def move_relative(self: HasCellProtocol, direction: tuple[int, ...]):
        """Move to a cell relative to the current cell.

        Args:
            direction: The direction to move in.
        """
        new_cell = self.cell.connections.get(direction)
        if new_cell is not None:
            self.cell = new_cell
        else:
            raise ValueError(f"No cell in direction {direction}")


class FixedCell(HasCell):
    """Descriptor for agents that are fixed to a cell."""

    def __set__(self, obj: Agent, value: Cell) -> None:  # noqa: D105
        try:
            current_cell = getattr(obj, self._private_name)
        except AttributeError:
            current_cell = None

        if current_cell is not None:
            raise ValueError("Cannot move agent in FixedCell")
        setattr(obj, self._private_name, value)
        value.add_agent(obj)


class CellAgent(Agent, BasicMovement):
    """Cell Agent is an extension of the Agent class and adds behavior for moving in discrete spaces.

    Attributes:
        cell (Cell): The cell the agent is currently in.
    """

    cell = HasCell()

    def remove(self):
        """Remove the agent from the model."""
        super().remove()
        self.cell = None  # ensures that we are also removed from cell


class FixedAgent(Agent):
    """A patch in a 2D grid."""

    cell = FixedCell()

    def remove(self):
        """Remove the agent from the model."""
        super().remove()

        # fixme we leave self._mesa_cell on the original value
        #  so you cannot hijack remove() to move patches
        self.cell.remove_agent(self)


class Grid2DMovingAgent(CellAgent):
    """Mixin for moving agents in 2D grids."""

    # fmt: off
    DIRECTION_MAP = {
        "n": (-1, 0), "north": (-1, 0), "up": (-1, 0),
        "s": (1, 0), "south": (1, 0), "down": (1, 0),
        "e": (0, 1), "east": (0, 1), "right": (0, 1),
        "w": (0, -1), "west": (0, -1), "left": (0, -1),
        "ne": (-1, 1), "northeast": (-1, 1), "upright": (-1, 1),
        "nw": (-1, -1), "northwest": (-1, -1), "upleft": (-1, -1),
        "se": (1, 1), "southeast": (1, 1), "downright": (1, 1),
        "sw": (1, -1), "southwest": (1, -1), "downleft": (1, -1)
    }
    # fmt: on

    def move(self, direction: str, distance: int = 1):
        """Move the agent in a cardinal direction.

        Args:
            direction: The cardinal direction to move in.
            distance: The distance to move.
        """
        direction = direction.lower()  # Convert direction to lowercase

        if direction not in self.DIRECTION_MAP:
            raise ValueError(f"Invalid direction: {direction}")

        move_vector = self.DIRECTION_MAP[direction]
        for _ in range(distance):
            self.move_relative(move_vector)
