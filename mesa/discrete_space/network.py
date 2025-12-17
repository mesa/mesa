"""Network-based cell space using arbitrary connection patterns.

Creates spaces where cells connect based on network relationships rather than
spatial proximity. Built on NetworkX graphs, this enables:
- Arbitrary connectivity patterns between cells
- Graph-based neighborhood definitions
- Logical rather than physical distances
- Dynamic connectivity changes
- Integration with NetworkX's graph algorithms

Useful for modeling systems like social networks, transportation systems,
or any environment where connectivity matters more than physical location.
"""

from random import Random
from typing import Any

import numpy as np
from numpy.random import BitGenerator, Generator, RandomState, SeedSequence

from mesa.discrete_space.cell import Cell
from mesa.discrete_space.discrete_space import DiscreteSpace
from mesa.util import deprecate_kwarg

SeedLike = int | np.ndarray[int] | SeedSequence | BitGenerator | Generator | RandomState


class Network(DiscreteSpace[Cell]):
    """A networked discrete space."""

    @deprecate_kwarg("random")
    def __init__(
        self,
        G: Any,  # noqa: N803
        capacity: int | None = None,
        random: Random | None = None,
        rng: SeedLike | None = None,
        cell_klass: type[Cell] = Cell,
    ) -> None:
        """A Networked grid.

        Args:
            G: a NetworkX Graph instance.
            capacity (int) : the capacity of the cell
            random (Random): a random number generator
            rng (SeedLike | None): the random number generator
            cell_klass (type[Cell]): The base Cell class to use in the Network

        """
        super().__init__(
            capacity=capacity, random=random, rng=rng, cell_klass=cell_klass
        )
        self.G = G

        for node_id in self.G.nodes:
            self._cells[node_id] = self.cell_klass(node_id, capacity, rng=self.rng)

        self._connect_cells()

    def _connect_cells(self) -> None:
        for cell in self.all_cells:
            self._connect_single_cell(cell)

    def _connect_single_cell(self, cell: Cell):
        for node_id in self.G.neighbors(cell.coordinate):
            cell.connect(self._cells[node_id], node_id)

    def add_cell(self, cell: Cell):
        """Add a cell to the space."""
        super().add_cell(cell)
        self.G.add_node(cell.coordinate)

    def remove_cell(self, cell: Cell):
        """Remove a cell from the space."""
        super().remove_cell(cell)
        self.G.remove_node(cell.coordinate)

    def add_connection(self, cell1: Cell, cell2: Cell):
        """Add a connection between the two cells."""
        super().add_connection(cell1, cell2)
        self.G.add_edge(cell1.coordinate, cell2.coordinate)

    def remove_connection(self, cell1: Cell, cell2: Cell):
        """Remove a connection between the two cells."""
        super().remove_connection(cell1, cell2)
        self.G.remove_edge(cell1.coordinate, cell2.coordinate)
