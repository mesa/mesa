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

from collections.abc import Callable
from random import Random
from typing import Any

import networkx as nx
import numpy as np
from scipy.spatial import KDTree

from mesa.discrete_space.cell import Cell
from mesa.discrete_space.discrete_space import DiscreteSpace


class Network(DiscreteSpace[Cell]):
    """A networked discrete space."""

    def __init__(
        self,
        G: Any,  # noqa: N803
        capacity: int | None = None,
        random: Random | None = None,
        cell_klass: type[Cell] = Cell,
        layout: Callable | None = nx.spring_layout,
    ) -> None:
        """A Networked grid.

        Args:
            G: a NetworkX Graph instance.
            capacity (int) : the capacity of the cell
            random (Random): a random number generator
            cell_klass (type[Cell]): The base Cell class to use in the Network
            layout (Callable | None): A function that computes positions for the
                nodes if they are missing (e.g. nx.spring_layout).
                Set to None to force a purely topological network (no positions).
                Defaults to nx.spring_layout.

        """
        super().__init__(capacity=capacity, random=random, cell_klass=cell_klass)
        self.G = G

        # Check for existing positions to determine if this is a spatial network
        has_existing_pos = any("pos" in data for _, data in self.G.nodes(data=True))

        # If positions are missing and a layout is provided, generate them.
        if not has_existing_pos and layout is not None:
            positions = layout(self.G)
            nx.set_node_attributes(self.G, positions, "pos")

        for node_id in self.G.nodes:
            # Extract position if spatial network
            pos = None
            if "pos" in self.G.nodes[node_id]:
                pos = np.array(self.G.nodes[node_id]["pos"])

            self._cells[node_id] = self.cell_klass(
                coordinate=node_id,
                capacity=capacity,
                random=self.random,
                position=pos,  # None for topological networks
            )

        self._connect_cells()
        self._build_kdtree()

    def _build_kdtree(self) -> None:
        """Build the KD-Tree for fast nearest-cell lookups."""
        self._kdtree_cells = [
            cell for cell in self._cells.values() if cell._position is not None
        ]

        if self._kdtree_cells:
            positions = np.array([c.position for c in self._kdtree_cells])
            self._kdtree = KDTree(positions)
        else:
            self._kdtree = None

    def _connect_cells(self) -> None:
        for cell in self.all_cells:
            self._connect_single_cell(cell)

    def _connect_single_cell(self, cell: Cell):
        for node_id in self.G.neighbors(cell.coordinate):
            cell.connect(self._cells[node_id], node_id)

    def find_nearest_cell(self, position: np.ndarray) -> Cell:
        """Find the network node nearest to the given position.

        Only works for spatial networks (networks with node positions).

        Args:
            position: Physical position [x, y]

        Returns:
            Cell: The node closest to the position

        Raises:
            ValueError: If network is not spatial
        """
        if getattr(self, "_kdtree", None) is None:
            raise ValueError("No nodes with positions found in network")

        _, index = self._kdtree.query(position)
        return self._kdtree_cells[index]

    def add_cell(self, cell: Cell):
        """Add a cell to the space."""
        super().add_cell(cell)
        self.G.add_node(cell.coordinate)
        self._build_kdtree()

    def remove_cell(self, cell: Cell):
        """Remove a cell from the space."""
        super().remove_cell(cell)
        self.G.remove_node(cell.coordinate)
        self._build_kdtree()

    def add_connection(self, cell1: Cell, cell2: Cell):
        """Add a connection between the two cells."""
        super().add_connection(cell1, cell2)
        self.G.add_edge(cell1.coordinate, cell2.coordinate)

    def remove_connection(self, cell1: Cell, cell2: Cell):
        """Remove a connection between the two cells."""
        super().remove_connection(cell1, cell2)
        self.G.remove_edge(cell1.coordinate, cell2.coordinate)
