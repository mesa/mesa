"""Cell spaces based on Voronoi tessellation around seed points.

Creates irregular spatial divisions by building cells around seed points,
where each cell contains the area closer to its seed than any other.
Features:
- Organic-looking spaces from point sets
- Automatic neighbor determination
- Area-based cell capacities
- Natural regional divisions

Useful for models requiring irregular but mathematically meaningful spatial
divisions, like territories, service areas, or natural regions.
"""

from collections.abc import Callable, Sequence
from itertools import pairwise
from random import Random

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import Delaunay, KDTree, QhullError

from mesa.discrete_space.cell import Cell
from mesa.discrete_space.discrete_space import DiscreteSpace


def round_float(x: float) -> int:  # noqa
    return int(x * 500)


def _clip(polygon: list, p: Sequence, q: Sequence) -> list:
    """Clip a convex polygon to the half-plane that is closer to p than to q."""
    nx, ny = q[0] - p[0], q[1] - p[1]
    offset = (nx * (p[0] + q[0]) + ny * (p[1] + q[1])) / 2
    clipped = []
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        da = nx * a[0] + ny * a[1] - offset
        db = nx * b[0] + ny * b[1] - offset
        if da <= 0:
            clipped.append(a)
        if da * db < 0:
            t = da / (da - db)
            clipped.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
    return clipped


class VoronoiGrid(DiscreteSpace):
    """Voronoi meshed GridSpace."""

    def __init__(
        self,
        centroids_coordinates: Sequence[Sequence[float]],
        capacity: int | Callable | None = None,
        random: Random | None = None,
        cell_klass: type[Cell] = Cell,
        bounds: ArrayLike | None = None,
    ) -> None:
        """A Voronoi Tessellation Grid.

        Given a set of points, this class creates a grid where a cell is centered in each point,
        its neighbors are given by Voronoi Tessellation cells neighbors
        and the capacity by the polygon area.

        Args:
            centroids_coordinates: coordinates of centroids to build the tessellation space
            capacity (int | Callable): capacity of the cells in the discrete space, or a
                callable computing the (int) capacity of a cell from its (float) polygon area
            random (Random): random number generator
            cell_klass (type[Cell]): type of cell class
            bounds: ``[[x_min, x_max], [y_min, y_max]]``, the area the cell polygons
                are clipped to. Defaults to the bounding box of the centroids.

        Raises:
            ValueError: if bounds does not contain every centroid.
        """
        # Separate callable capacity from numeric capacity before passing to base class
        if callable(capacity):
            capacity_function = capacity
            numeric_capacity = None
        else:
            capacity_function = None
            numeric_capacity = capacity

        super().__init__(
            capacity=numeric_capacity, random=random, cell_klass=cell_klass
        )
        self.centroids_coordinates = centroids_coordinates
        self.capacity_function = capacity_function
        self._validate_parameters()

        # Create cells with Integer coordinates and physical positions
        # coordinate is now an integer index
        self._cells = {
            i: cell_klass(
                coordinate=i,  # Integer index
                capacity=capacity,
                random=self.random,
                position=self.centroids_coordinates[i],  # Physical centroid position
            )
            for i in range(len(self.centroids_coordinates))
        }

        # Build KD-tree for fast nearest-centroid lookup
        self._kdtree = KDTree(np.array(centroids_coordinates))
        self._kdtree_cells: list[Cell] = list(self._cells.values())
        self._kdtree_dirty = False

        centroids = np.asarray(centroids_coordinates, dtype=float)
        if bounds is None:
            bounds = np.column_stack((centroids.min(axis=0), centroids.max(axis=0)))
        self.bounds = np.asarray(bounds, dtype=float)
        if self.bounds.shape != (2, 2) or not np.all(
            (self.bounds[:, 0] <= centroids) & (centroids <= self.bounds[:, 1])
        ):
            raise ValueError(
                "bounds must be [[x_min, x_max], [y_min, y_max]] and contain every centroid"
            )
        # Work around the center of the bounds: large absolute coordinates
        # (e.g. UTM) lose floating-point precision in the triangulation.
        self._offset = self.bounds.mean(axis=1)
        self._points = (centroids - self._offset).tolist()

        self._connect_cells()
        self._build_cell_polygons()

    def _rebuild_kdtree(self) -> None:
        """Rebuild the KD-Tree."""
        if self._kdtree_cells:
            positions = np.array([c.position for c in self._kdtree_cells])
            self._kdtree = KDTree(positions)
        else:
            self._kdtree = None
        self._kdtree_dirty = False

    def find_nearest_cell(self, position: np.ndarray) -> Cell:
        """Find the Voronoi cell nearest to the given position.

        Note:
            The KD-Tree index is rebuilt lazily. If cells were added or removed
            since the previous spatial query, this method rebuilds the KD-Tree
            before performing the nearest-neighbor lookup.

        Args:
            position: Physical position [x, y]

        Returns:
            Cell: The Voronoi cell whose centroid is nearest to position

        Raises:
            ValueError: If no cells exist in the VoronoiGrid
        """
        if self._kdtree_dirty:
            self._rebuild_kdtree()

        if getattr(self, "_kdtree", None) is None:
            raise ValueError("No cells found in VoronoiGrid")

        position = np.asarray(position)

        # Find nearest centroid using KD-tree
        _distance, index = self._kdtree.query(position)

        return self._kdtree_cells[index]

    def add_cell(self, cell: Cell):
        """Add a cell to the space."""
        super().add_cell(cell)
        if cell._position is not None:
            self._kdtree_cells.append(cell)
            self._kdtree_dirty = True

    def remove_cell(self, cell: Cell):
        """Remove a cell from the space."""
        super().remove_cell(cell)
        if cell._position is not None and cell in self._kdtree_cells:
            self._kdtree_cells.remove(cell)
            self._kdtree_dirty = True

    def _connect_cells(self) -> None:
        """Connect cells whose Voronoi regions share an edge, using a Delaunay triangulation."""
        try:
            indptr, indices = Delaunay(self._points).vertex_neighbor_vertices
            neighbors = [indices[indptr[i] : indptr[i + 1]] for i in self._cells]
        except QhullError:
            # Fewer than three centroids, or all on one line: each region is a
            # strip that borders the next centroid along the line.
            points = np.asarray(self._points)
            order = np.argsort(points @ (points[-1] - points[0]))
            neighbors = [[] for _ in self._cells]
            for i, j in pairwise(order):
                neighbors[i].append(j)
                neighbors[j].append(i)

        for i, cell in self._cells.items():
            for j in neighbors[i]:
                cell.connect(self._cells[int(j)], int(j))

    def _validate_parameters(self) -> None:
        if self.capacity is not None and not isinstance(self.capacity, float | int):
            raise ValueError("Capacity must be a number or None.")

        if not isinstance(self.centroids_coordinates, Sequence) or not isinstance(
            self.centroids_coordinates[0], Sequence
        ):
            raise ValueError("Centroids should be a list of lists")
        dimension_1 = len(self.centroids_coordinates[0])
        seen = set()
        for coordinate in self.centroids_coordinates:
            if dimension_1 != len(coordinate):
                raise ValueError("Centroid coordinates should be a homogeneous array")
            # Two identical centroids have no border between them, so the
            # triangulation fails or never finishes
            if tuple(coordinate) in seen:
                raise ValueError(
                    f"Centroids must be distinct, {list(coordinate)} appears more than once"
                )
            seen.add(tuple(coordinate))

    @staticmethod
    def _compute_polygon_area(polygon: list) -> float:
        polygon = np.array(polygon)
        # The shoelace formula is translation-invariant, so recentre on the
        # first vertex before summing. At large absolute coordinates (e.g.
        # utm) the cross products reach ~1e13 and their cancellation loses
        # several digits of area precision.
        polygon = polygon - polygon[0]
        x = polygon[:, 0]
        y = polygon[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def _build_cell_polygons(self):
        (x_min, x_max), (y_min, y_max) = self.bounds - self._offset[:, None]
        box = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        for i, cell in self._cells.items():
            # A Voronoi region is the part of the plane closer to its centroid
            # than to any other, and its Delaunay neighbours alone fix its edges.
            polygon = box
            for j in cell.connections:
                polygon = _clip(polygon, self._points[i], self._points[j])
            polygon_area = self._compute_polygon_area(polygon)
            cell.properties["polygon"] = list(np.array(polygon) + self._offset)
            cell.properties["area"] = polygon_area
            if self.capacity is not None:
                # User provided a fixed capacity — use it directly.
                cell.capacity = self.capacity
            elif self.capacity_function is not None:
                # No fixed capacity but a function was provided — derive from area.
                cell.capacity = self.capacity_function(polygon_area)
            # else: both are None — cell capacity stays None (no limit).
