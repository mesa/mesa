"""Base class for building cell-based spatial environments.

DiscreteSpace provides the core functionality needed by all cell-based spaces:
- Cell creation and tracking
- Agent-cell relationship management
- Property layer support
- Random selection capabilities
- Capacity management

This serves as the foundation for specific space implementations like grids
and networks, ensuring consistent behavior and shared functionality across
different space types. All concrete cell space implementations (grids, networks, etc.)
inherit from this class.
"""

from __future__ import annotations

import warnings
from functools import cached_property
from random import Random
from typing import TypeVar

import numpy as np

from mesa.agent import Agent, AgentSet
from mesa.discrete_space.cell import Cell
from mesa.discrete_space.cell_collection import CellCollection

T = TypeVar("T", bound=Cell)


class DiscreteSpace[T: Cell]:
    """Base class for all discrete spaces.

    Attributes:
        capacity (int): The capacity of the cells in the discrete space
        all_cells (CellCollection): The cells composing the discrete space
        random (Random): The random number generator
        cell_klass (Type) : the type of cell class
        empties (CellCollection) : collection of all cells that are empty
        property_layers (dict[str, PropertyLayer]): the property layers of the discrete space

    Notes:
        A `UserWarning` is issued if `random=None`. You can resolve this warning by explicitly
        passing a random number generator. In most cases, this will be the seeded random number
        generator in the model. So, you would do `random=self.random` in a `Model` or `Agent` instance.

    """

    def __init__(
        self,
        capacity: int | None = None,
        cell_klass: type[T] = Cell,
        random: Random | None = None,
    ):
        """Instantiate a DiscreteSpace.

        Args:
            capacity: capacity of cells
            cell_klass: base class for all cells
            random: random number generator
        """
        super().__init__()
        self.capacity = capacity
        self._cells: dict[tuple[int, ...], T] = {}
        if random is None:
            warnings.warn(
                "Random number generator not specified, this can make models non-reproducible. Please pass a random number generator explicitly",
                UserWarning,
                stacklevel=2,
            )
            random = Random()
        self.random = random
        self.cell_klass = cell_klass
        self._agent_to_cell: dict[Agent, T] = {}  # NEW: Track agent->cell mapping
        self._agents: set[Agent] = set()  # NEW: Track all agents

        self._empties: dict[tuple[int, ...], None] = {}

    def add(self, agent: Agent) -> None:
        """Register agent in this space."""
        if agent.position is None:
            raise ValueError("Agent must have position before adding to space")

        # Subscribe to position updates
        if self not in agent._discrete_spaces:
            agent._discrete_spaces.append(self)

        # Set initial cell membership
        cell = self.position_to_cell(agent.position)
        self._agent_to_cell[agent] = cell
        cell.add_agent(agent)
        self._agents.add(agent)

    def remove(self, agent: Agent) -> None:
        """Remove agent from this space."""
        # Unsubscribe
        if self in agent._discrete_spaces:
            agent._discrete_spaces.remove(self)

        # Remove from cell
        cell = self._agent_to_cell.get(agent)
        if cell:
            cell.remove_agent(agent)
            del self._agent_to_cell[agent]
        self._agents.discard(agent)

    def cell(self, agent: Agent) -> T:
        """Get the cell this agent is in (read-only)."""
        return self._agent_to_cell.get(agent)

    def move(self, agent: Agent, target: T, align: str = "center") -> None:
        """Move agent to target cell.

        Args:
            agent: Agent to move
            target: Target cell
            align: "center" | "random" - where to position in cell
        """
        # Calculate new position
        if align == "center":
            new_position = self.cell_to_position(target)
        elif align == "random":
            new_position = self.random_position_in_cell(target)
        else:
            raise ValueError(f"Unknown align: {align}")

        # Update position (this triggers _on_agent_position_changed via setter)
        agent.position = new_position

    def _on_agent_position_changed(
        self, agent: Agent, old_position: np.ndarray, new_position: np.ndarray
    ) -> None:
        """Handle agent position change (called automatically)."""
        old_cell = self._agent_to_cell.get(agent)
        new_cell = self.position_to_cell(new_position)

        # Only update if cell actually changed
        if old_cell != new_cell:
            if old_cell:
                old_cell.remove_agent(agent)
            new_cell.add_agent(agent)
            self._agent_to_cell[agent] = new_cell

    def position_to_cell(self, position: np.ndarray) -> T:
        """Convert position to cell (must implement in subclass)."""
        raise NotImplementedError

    def cell_to_position(self, cell: T) -> np.ndarray:
        """Get position for cell center (must implement in subclass)."""
        raise NotImplementedError

    def random_position_in_cell(self, cell: T) -> np.ndarray:
        """Get random position in cell (must implement in subclass)."""
        raise NotImplementedError

    @property
    def cutoff_empties(self):  # noqa
        return 7.953 * len(self._cells) ** 0.384

    @property
    def agents(self) -> AgentSet:
        """Return an AgentSet with the agents in the space."""
        return AgentSet(self.all_cells.agents, random=self.random)

    def _connect_cells(self): ...
    def _connect_single_cell(self, cell: T): ...

    def add_cell(self, cell: T):
        """Add a cell to the space.

        Args:
            cell: cell to add

        Note:
            Discrete spaces rely on caching neighborhood relations for speedups. Adding or removing cells and
            connections at runtime is possible. However, only the caches of cells directly affected will be cleared. So
            if you rely on getting neighborhoods of cells with a radius higher than 1, these might not be cleared
            correctly if you are adding or removing cells and connections at runtime.

        """
        self.__dict__.pop("all_cells", None)
        self._cells[cell.coordinate] = cell

    def remove_cell(self, cell: T):
        """Remove a cell from the space.

        Note:
            Discrete spaces rely on caching neighborhood relations for speedups. Adding or removing cells and
            connections at runtime is possible. However, only the caches of cells directly affected will be cleared. So
            if you rely on getting neighborhoods of cells with a radius higher than 1, these might not be cleared
            correctly if you are adding or removing cells and connections at runtime.


        """
        neighbors = cell.neighborhood
        self._cells.pop(cell.coordinate)
        self.__dict__.pop("all_cells", None)

        # iterate over all neighbors
        for neighbor in neighbors.cells:
            neighbor.disconnect(cell)
            cell.disconnect(neighbor)

    def add_connection(self, cell1: T, cell2: T):
        """Add a connection between the two cells.

        Note:
            Discrete spaces rely on caching neighborhood relations for speedups. Adding or removing cells and
            connections at runtime is possible. However, only the caches of cells directly affected will be cleared. So
            if you rely on getting neighborhoods of cells with a radius higher than 1, these might not be cleared
            correctly if you are adding or removing cells and connections at runtime.

        """
        cell1.connect(cell2)
        cell2.connect(cell1)

    def remove_connection(self, cell1: T, cell2: T):
        """Remove a connection between the two cells.

        Note:
            Discrete spaces rely on caching neighborhood relations for speedups. Adding or removing cells and
            connections at runtime is possible. However, only the caches of cells directly affected will be cleared. So
            if you rely on getting neighborhoods of cells with a radius higher than 1, these might not be cleared
            correctly if you are adding or removing cells and connections at runtime.

        """
        cell1.disconnect(cell2)
        cell2.disconnect(cell1)

    @cached_property
    def all_cells(self):
        """Return all cells in space."""
        return CellCollection(
            {cell: cell._agents for cell in self._cells.values()}, random=self.random
        )

    def __iter__(self):  # noqa
        return iter(self._cells.values())

    def __getitem__(self, key: tuple[int, ...]) -> T:  # noqa: D105
        return self._cells[key]

    @property
    def empties(self) -> CellCollection[T]:
        """Return all empty in spaces."""
        return self.all_cells.select(lambda cell: cell.is_empty)

    def select_random_empty_cell(self) -> T:
        """Select random empty cell."""
        return self.random.choice(list(self.empties))

    def __setstate__(self, state):
        """Set the state of the discrete space and rebuild the connections."""
        self.__dict__ = state
        self._connect_cells()
