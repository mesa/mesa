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

from functools import cache
from random import Random
from typing import TYPE_CHECKING

from mesa.discrete_space.cell_agent import CellAgent
from mesa.discrete_space.cell_collection import CellCollection

if TYPE_CHECKING:
    from mesa.agent import Agent

Coordinate = tuple[int, ...]


class Cell:
    """The cell represents a position in a discrete space.

    Attributes:
        coordinate (Tuple[int, int]) : the position of the cell in the discrete space
        agents (List[Agent]): the agents occupying the cell
        capacity (int): the maximum number of agents that can simultaneously occupy the cell
        random (Random): the random number generator

    """

    __slots__ = [
        "_agents",
        "_empty",
        "capacity",
        "connections",
        "coordinate",
        "properties",
        "random",
    ]

    @property
    def empty(self) -> bool:  # noqa: D102
        return self._empty

    @empty.setter
    def empty(self, value: bool) -> None:
        self._empty = value

    def __init__(
        self,
        coordinate: Coordinate,
        capacity: int | None = None,
        random: Random | None = None,
    ) -> None:
        """Initialise the cell.

        Args:
            coordinate: coordinates of the cell
            capacity (int) : the capacity of the cell. If None, the capacity is infinite
            random (Random) : the random number generator to use

        """
        super().__init__()
        self.coordinate = coordinate
        self.connections: dict[Coordinate, Cell] = {}
        self._agents: list[
            CellAgent
        ] = []  # TODO:: change to AgentSet or weakrefs? (neither is very performant, )
        self.capacity: int | None = capacity
        self.properties: dict[
            Coordinate, object
        ] = {}  # fixme still used by voronoi mesh
        self.random = random

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
        for key in keys_to_remove:
            del self.connections[key]
        self._clear_cache()

    def add_agent(self, agent: CellAgent) -> None:
        """Adds an agent to the cell.

        Args:
            agent (CellAgent): agent to add to this Cell

        """
        n = len(self._agents)
        self.empty = False

        if self.capacity is not None and n >= self.capacity:
            raise Exception(
                "ERROR: Cell is full"
            )  # FIXME we need MESA errors or a proper error

        self._agents.append(agent)

    def remove_agent(self, agent: CellAgent) -> None:
        """Removes an agent from the cell.

        Args:
            agent (CellAgent): agent to remove from this cell

        """
        self._agents.remove(agent)
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
        return self.get_neighborhood()

    # FIXME: Revisit caching strategy on methods
    @cache  # noqa: B019
    def get_neighborhood(
        self, radius: int = 1, include_center: bool = False
    ) -> CellCollection[Cell]:
        """Returns a list of all neighboring cells for the given radius.

        For getting the direct neighborhood (i.e., radius=1) you can also use
        the `neighborhood` property.

        Args:
            radius (int): the radius of the neighborhood
            include_center (bool): include the center of the neighborhood

        Returns:
            a list of all neighboring cells

        """
        return CellCollection[Cell](
            self._neighborhood(radius=radius, include_center=include_center),
            random=self.random,
        )

    # FIXME: Revisit caching strategy on methods
    @cache  # noqa: B019
    def _neighborhood(
        self, radius: int = 1, include_center: bool = False
    ) -> dict[Cell, list[Agent]]:
        """Return cells within given radius using iterative BFS.

        Note: This implementation uses iterative breadth-first search instead
        of recursion to avoid RecursionError on large radius values.
        """
        if radius < 1:
            raise ValueError("radius must be larger than one")

        # Fast path for radius=1 (most common case) - avoid BFS overhead
        if radius == 1:
            neighborhood = {
                neighbor: neighbor._agents for neighbor in self.connections.values()
            }
            if include_center:
                neighborhood[self] = self._agents
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
        """Return state of the Cell with connections set to empty."""
        state = super().__getstate__()
        # Replace connections with empty dict to avoid infinite recursion error in pickle/deepcopy
        state[1]["connections"] = {}
        return state

    def _clear_cache(self):
        """Helper function to clear local cache."""
        self.get_neighborhood.cache_clear()
        self._neighborhood.cache_clear()
