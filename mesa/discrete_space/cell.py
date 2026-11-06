"""Cells are positions in space that can have properties and contain agents.

A cell represents a location that can:
- Have properties (like temperature or resources)
- Track and limit the agents it contains
- Connect to neighboring cells
- Provide neighborhood information

Cells form the foundation of the cell space system, enabling rich spatial
environments where both location properties and agent behaviors matter. They're
useful for modeling things like varying terrain, infrastructure capacity, or
environmental conditions.
"""

from __future__ import annotations

from random import Random
from typing import TYPE_CHECKING
from weakref import WeakSet

import numpy as np

from mesa.discrete_space.cell_agent import CellAgent
from mesa.discrete_space.cell_collection import CellCollection
from mesa.exceptions import (
    AgentMissingException,
    CellFullException,
    ConnectionMissingException,
)

if TYPE_CHECKING:
    from mesa.agent import Agent

Coordinate = tuple[int, ...]


class Cell:
    """The cell represents a position in a discrete space.

    Attributes:
        coordinate (Coordinate) : the logical position(or index) of the cell in the discrete space
        position (np.ndarray | None): the physical position of the cell in the discrete space
        agents (List[Agent]): the agents occupying the cell
        capacity (int): the maximum number of agents that can simultaneously occupy the cell
        random (Random): the random number generator

    """

    __slots__ = [
        "__weakref__",
        "_agents",
        "_empty",
        "_neighborhood_cache",
        "_neighborhood_radius1",
        "_neighborhood_radius1_with_center",
        "_neighborhood_radius2",
        "_neighborhood_radius2_with_center",
        "_position",  # physical position
        "capacity",
        "connections",
        "coordinate",  # Logical index
        "properties",
        "random",
    ]

    @property
    def empty(self) -> bool:  # noqa: D102
        return self._empty

    @empty.setter
    def empty(self, value: bool) -> None:
        self._empty = value

    @property
    def position(self) -> np.ndarray:
        """Get the physical position of the cell.

        Returns:
            np.ndarray: Physical position of the cell
        """
        if self._position is not None:
            return self._position
        # Default for implicit grids
        return np.asarray(self.coordinate, dtype=float)

    @position.setter
    def position(self, value: np.ndarray | None) -> None:
        self._position = value

    def __init__(
        self,
        coordinate: Coordinate,
        position: np.ndarray | None = None,
        capacity: int | None = None,
        random: Random | None = None,
    ) -> None:
        """Initialise the cell.

        Args:
            coordinate: coordinates of the cell
            position: physical coordinates of the cell
            capacity (int) : the capacity of the cell. If None, the capacity is infinite
            random (Random) : the random number generator to use

        """
        super().__init__()
        self.coordinate = coordinate  # Logical index
        self._position = position  # Physical position
        self.connections: dict[Coordinate, Cell] = {}
        self._agents: list[
            CellAgent
        ] = []  # TODO:: change to AgentSet or weakrefs? (neither is very performant, )
        self._empty: bool = True  # a freshly created cell holds no agents
        self.capacity: int | None = capacity
        self.properties: dict[
            Coordinate, object
        ] = {}  # fixme still used by voronoi mesh
        self.random = random
        self._neighborhood_cache: (
            dict[tuple[int, bool], CellCollection[Cell]] | None
        ) = None
        self._neighborhood_radius1: CellCollection[Cell] | None = None
        self._neighborhood_radius1_with_center: CellCollection[Cell] | None = None
        self._neighborhood_radius2: CellCollection[Cell] | None = None
        self._neighborhood_radius2_with_center: CellCollection[Cell] | None = None

    def connect(self, other: Cell, key: Coordinate | None = None) -> None:
        """Connects this cell to another cell.

        Args:
            other (Cell): other cell to connect to
            key (Tuple[int, ...]): key for the connection. Should resemble a relative coordinate

        """
        if key is None:
            key = other.coordinate
        self._clear_cache()
        self.connections[key] = other

    def disconnect(self, other: Cell) -> None:
        """Disconnects this cell from another cell.

        Args:
            other (Cell): other cell to remove from connections

        """
        keys_to_remove = [k for k, v in self.connections.items() if v == other]

        if not keys_to_remove:
            raise ConnectionMissingException(self, other)

        for key in keys_to_remove:
            del self.connections[key]
        self._clear_cache()

    def add_agent(self, agent: CellAgent) -> None:
        """Adds an agent to the cell.

        Args:
            agent (CellAgent): agent to add to this Cell

        """
        if self.capacity is not None and len(self._agents) >= self.capacity:
            raise CellFullException(self.coordinate)

        self._agents.append(agent)
        self.empty = False

    def remove_agent(self, agent: CellAgent) -> None:
        """Removes an agent from the cell.

        Args:
            agent (CellAgent): agent to remove from this cell

        """
        try:
            self._agents.remove(agent)
        except ValueError as e:
            raise AgentMissingException(agent, self.coordinate) from e

        self.empty = self.is_empty

    @property
    def is_empty(self) -> bool:
        """Returns a bool of the contents of a cell."""
        return len(self._agents) == 0

    @property
    def is_full(self) -> bool:
        """Returns a bool of the contents of a cell."""
        if self.capacity is None:
            return False
        return len(self._agents) >= self.capacity

    @property
    def agents(self) -> list[CellAgent]:
        """Returns a list of the agents occupying the cell."""
        return self._agents.copy()

    def __repr__(self):  # noqa
        return f"Cell({self.coordinate}, {self.agents})"

    @property
    def neighborhood(self) -> CellCollection[Cell]:
        """Returns the direct neighborhood of the cell.

        This is equivalent to cell.get_neighborhood(radius=1)

        """
        try:
            neighborhood = self._neighborhood_radius1
        except AttributeError:  # Older pickles predate the derived cache slots.
            self._initialize_neighborhood_cache()
            neighborhood = None

        if neighborhood is None:
            neighborhood = CellCollection[Cell](
                self._neighborhood(radius=1, include_center=False),
                random=self.random,
            )
            self._neighborhood_radius1 = neighborhood
            _ACTIVE_NEIGHBORHOOD_CELLS.add(self)

        return neighborhood

    def get_neighborhood(
        self, radius: int = 1, include_center: bool = False
    ) -> CellCollection[Cell]:
        """Returns a CellCollection of all neighboring cells for the given radius.

        For getting the direct neighborhood (i.e., radius=1) you can also use
        the `neighborhood` property.

        Args:
            radius (int): the radius of the neighborhood
            include_center (bool): include the center of the neighborhood

        Returns:
            a CellCollection of all neighboring cells

        """
        try:
            radius1_neighborhood = self._neighborhood_radius1
        except AttributeError:  # Older pickles predate the derived cache slots.
            self._initialize_neighborhood_cache()
            radius1_neighborhood = None

        # Radius 1 and 2 are common model queries; dedicated slots avoid
        # tuple creation and dictionary lookup on these cache-hit paths.
        if radius == 1:
            if include_center:
                neighborhood = self._neighborhood_radius1_with_center
                if neighborhood is not None:
                    return neighborhood
                neighborhood = CellCollection[Cell](
                    self._neighborhood(radius=1, include_center=True),
                    random=self.random,
                )
                self._neighborhood_radius1_with_center = neighborhood
            else:
                if radius1_neighborhood is not None:
                    return radius1_neighborhood
                neighborhood = CellCollection[Cell](
                    self._neighborhood(radius=1, include_center=False),
                    random=self.random,
                )
                self._neighborhood_radius1 = neighborhood
        elif radius == 2:
            if include_center:
                neighborhood = self._neighborhood_radius2_with_center
                if neighborhood is not None:
                    return neighborhood
                neighborhood = CellCollection[Cell](
                    self._neighborhood(radius=2, include_center=True),
                    random=self.random,
                )
                self._neighborhood_radius2_with_center = neighborhood
            else:
                neighborhood = self._neighborhood_radius2
                if neighborhood is not None:
                    return neighborhood
                neighborhood = CellCollection[Cell](
                    self._neighborhood(radius=2, include_center=False),
                    random=self.random,
                )
                self._neighborhood_radius2 = neighborhood
        else:
            neighborhood_cache = self._neighborhood_cache
            if neighborhood_cache is None:
                neighborhood_cache = {}
                self._neighborhood_cache = neighborhood_cache
            cache_key = (radius, include_center)
            try:
                return neighborhood_cache[cache_key]
            except KeyError:
                neighborhood = CellCollection[Cell](
                    self._neighborhood(radius=radius, include_center=include_center),
                    random=self.random,
                )
                neighborhood_cache[cache_key] = neighborhood

        _ACTIVE_NEIGHBORHOOD_CELLS.add(self)
        return neighborhood

    def _initialize_neighborhood_cache(self) -> None:
        """Initialize derived cache slots for cells restored from older pickles."""
        self._neighborhood_cache = None
        self._neighborhood_radius1 = None
        self._neighborhood_radius1_with_center = None
        self._neighborhood_radius2 = None
        self._neighborhood_radius2_with_center = None

    def _neighborhood(
        self, radius: int = 1, include_center: bool = False
    ) -> dict[Cell, list[Agent]]:
        """Return cells within given radius using iterative BFS.

        Note: This implementation uses iterative breadth-first search instead
        of recursion to avoid RecursionError on large radius values.
        """
        if radius < 1:
            raise ValueError("radius must be at least 1")

        # Fast path for radius=1 (most common case) - avoid BFS overhead
        if radius == 1:
            neighborhood = {
                neighbor: neighbor._agents for neighbor in self.connections.values()
            }
            if include_center:
                neighborhood[self] = self._agents
            else:
                neighborhood.pop(self, None)
            return neighborhood

        # Use iterative BFS for radius > 1 to avoid RecursionError
        visited: set[Cell] = {self}
        current_layer: list[Cell] = list(self.connections.values())
        neighborhood: dict[Cell, list[Agent]] = {}

        # Add immediate neighbors (radius=1)
        for neighbor in current_layer:
            if neighbor not in visited:
                visited.add(neighbor)
                neighborhood[neighbor] = neighbor._agents

        # Expand outward for remaining radius levels
        for _ in range(radius - 1):
            next_layer: list[Cell] = []
            for cell in current_layer:
                for neighbor in cell.connections.values():
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_layer.append(neighbor)
                        neighborhood[neighbor] = neighbor._agents
            current_layer = next_layer
            if not current_layer:
                break  # No more cells to explore

        # Handle center inclusion
        if include_center:
            neighborhood[self] = self._agents
        else:
            neighborhood.pop(self, None)

        return neighborhood

    def __getstate__(self):
        """Return state of the Cell.

        Neighbors are replaced with their coordinates to avoid deep recursion
        while preserving the connection keys.
        """
        state = super().__getstate__()
        # Neighborhoods are derived from connections and must not be serialized.
        # Store empty cache slots so they are initialized after unpickling.
        state[1]["_neighborhood_cache"] = None
        state[1]["_neighborhood_radius1"] = None
        state[1]["_neighborhood_radius1_with_center"] = None
        state[1]["_neighborhood_radius2"] = None
        state[1]["_neighborhood_radius2_with_center"] = None
        # Replace neighbor objects with their coordinates to avoid deep recursion
        # while preserving the connection keys.
        state[1]["connections"] = {
            key: neighbor.coordinate for key, neighbor in self.connections.items()
        }
        return state

    def _clear_cache(self) -> None:
        """Invalidate all live neighborhood caches after a topology change.

        A weak registry preserves graph-wide invalidation (including radius > 1)
        without keeping cached Cells or their models alive.
        """
        if not _ACTIVE_NEIGHBORHOOD_CELLS:
            return

        for cell in _ACTIVE_NEIGHBORHOOD_CELLS:
            cell._neighborhood_cache = None
            cell._neighborhood_radius1 = None
            cell._neighborhood_radius1_with_center = None
            cell._neighborhood_radius2 = None
            cell._neighborhood_radius2_with_center = None
        _ACTIVE_NEIGHBORHOOD_CELLS.clear()


# Weak references avoid making this registry a root for Cells or their models.
_ACTIVE_NEIGHBORHOOD_CELLS: WeakSet[Cell] = WeakSet()
