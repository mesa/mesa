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

from collections.abc import Sequence
from itertools import combinations
from random import Random

import numpy as np

from mesa.discrete_space.cell import Cell
from mesa.discrete_space.discrete_space import DiscreteSpace


class Delaunay:
    """Class to compute a Delaunay triangulation in 2D.

    ref: http://github.com/jmespadero/pyDelaunay2D
    """

    def __init__(self, center: tuple = (0, 0), radius: int = 9999) -> None:
        """Init and create a new frame to contain the triangulation.

        Args:
            center: Optional position for the center of the frame. Default (0,0)
            radius: Optional distance from corners to the center.
        """
        center = np.asarray(center)
        # Create coordinates for the corners of the frame
        self.coords = [
            center + radius * np.array((-1, -1)),
            center + radius * np.array((+1, -1)),
            center + radius * np.array((+1, +1)),
            center + radius * np.array((-1, +1)),
        ]

        # Create two dicts to store triangle neighbours and circumcircles.
        self.triangles = {}
        self.circles = {}

        # Create two CCW triangles for the frame
        triangle1 = (0, 1, 3)
        triangle2 = (2, 3, 1)
        self.triangles[triangle1] = [triangle2, None, None]
        self.triangles[triangle2] = [triangle1, None, None]

        # Compute circumcenters and circumradius for each triangle
        for t in self.triangles:
            self.circles[t] = self._circumcenter(t)

    def _circumcenter(self, triangle: list) -> tuple:
        """Compute circumcenter and circumradius of a triangle in 2D."""
        points = np.asarray([self.coords[v] for v in triangle])
        points2 = np.dot(points, points.T)
        a = np.block(
            [[2 * points2, np.array([[1], [1], [1]])], [np.array([[1, 1, 1, 0]])]]
        )

        b = np.hstack((np.sum(points * points, axis=1), [1]))
        x = np.linalg.solve(a, b)
        bary_coords = x[:-1]
        center = np.dot(bary_coords, points)

        radius = np.sum(np.square(points[0] - center))  # squared distance
        return (center, radius)

    def _in_circle(self, triangle: list, point: list) -> bool:
        """Check if point p is inside of precomputed circumcircle of triangle."""
        center, radius = self.circles[triangle]
        return np.sum(np.square(center - point)) <= radius

    def add_point(self, point: Sequence) -> None:
        """Add a point to the current DT, and refine it using Bowyer-Watson."""
        point_index = len(self.coords)
        self.coords.append(np.asarray(point))

        bad_triangles = []
        for triangle in self.triangles:
            if self._in_circle(triangle, point):
                bad_triangles.append(triangle)

        boundary = []
        triangle = bad_triangles[0]
        edge = 0

        while True:
            opposite_triangle = self.triangles[triangle][edge]
            if opposite_triangle not in bad_triangles:
                boundary.append(
                    (
                        triangle[(edge + 1) % 3],
                        triangle[(edge - 1) % 3],
                        opposite_triangle,
                    )
                )
                edge = (edge + 1) % 3
                if boundary[0][0] == boundary[-1][1]:
                    break
            else:
                edge = (self.triangles[opposite_triangle].index(triangle) + 1) % 3
                triangle = opposite_triangle

        for triangle in bad_triangles:
            del self.triangles[triangle]
            del self.circles[triangle]

        new_triangles = []
        for e0, e1, opposite_triangle in boundary:
            triangle = (point_index, e0, e1)
            self.circles[triangle] = self._circumcenter(triangle)
            self.triangles[triangle] = [opposite_triangle, None, None]
            if opposite_triangle:
                for i, neighbor in enumerate(self.triangles[opposite_triangle]):
                    if neighbor and e1 in neighbor and e0 in neighbor:
                        self.triangles[opposite_triangle][i] = triangle

            new_triangles.append(triangle)

        n = len(new_triangles)
        for i, triangle in enumerate(new_triangles):
            self.triangles[triangle][1] = new_triangles[(i + 1) % n]  # next
            self.triangles[triangle][2] = new_triangles[(i - 1) % n]  # previous

    def export_triangles(self) -> list:
        """Export the current list of Delaunay triangles."""
        triangles_list = [
            (a - 4, b - 4, c - 4)
            for (a, b, c) in self.triangles
            if a > 3 and b > 3 and c > 3
        ]
        return triangles_list

    def export_voronoi_regions(self):
        """Export coordinates and regions of Voronoi diagram as indexed data."""
        use_vertex = {i: [] for i in range(len(self.coords))}
        vor_coors = []
        index = {}
        for triangle_index, (a, b, c) in enumerate(sorted(self.triangles)):
            vor_coors.append(self.circles[(a, b, c)][0])
            use_vertex[a] += [(b, c, a)]
            use_vertex[b] += [(c, a, b)]
            use_vertex[c] += [(a, b, c)]

            index[(a, b, c)] = triangle_index
            index[(c, a, b)] = triangle_index
            index[(b, c, a)] = triangle_index

        regions = {}
        for i in range(4, len(self.coords)):
            vertex = use_vertex[i][0][0]
            region = []
            for _ in range(len(use_vertex[i])):
                triangle = next(
                    triangle for triangle in use_vertex[i] if triangle[0] == vertex
                )
                region.append(index[triangle])
                vertex = triangle[1]
            regions[i - 4] = region

        return vor_coors, regions


def round_float(x: float) -> int:  # noqa
    return int(x * 500)


class VoronoiGrid(DiscreteSpace):
    """Voronoi meshed GridSpace."""

    triangulation: Delaunay
    voronoi_coordinates: list
    regions: list

    def __init__(
        self,
        centroids_coordinates: Sequence[Sequence[float]],
        capacity: float | None = None,
        random: Random | None = None,
        cell_klass: type[Cell] = Cell,
        capacity_function: callable = round_float,
    ) -> None:
        """A Voronoi Tessellation Grid.

        Given a set of points, this class creates a grid where a cell is centered in each point,
        its neighbors are given by Voronoi Tessellation cells neighbors
        and the capacity by the polygon area.

        Args:
            centroids_coordinates: coordinates of centroids to build the tessellation space
            capacity (int) : capacity of the cells in the discrete space
            random (Random): random number generator
            cell_klass (type[Cell]): type of cell class
            capacity_function (Callable): function to compute (int) capacity according to (float) area

        """
        super().__init__(capacity=capacity, random=random, cell_klass=cell_klass)
        self.centroids_coordinates = centroids_coordinates
        self._validate_parameters()

        self._cells = {
            i: cell_klass(self.centroids_coordinates[i], capacity, random=self.random)
            for i in range(len(self.centroids_coordinates))
        }

        self.regions = None
        self.triangulation = None
        self.voronoi_coordinates = None
        self.capacity_function = capacity_function

        self._connect_cells()
        self._build_cell_polygons()

    def _connect_cells(self) -> None:
        """Connect cells to neighbors based on given centroids and using Delaunay Triangulation."""
        self.triangulation = Delaunay()
        for centroid in self.centroids_coordinates:
            self.triangulation.add_point(centroid)

        for point in self.triangulation.export_triangles():
            for i, j in combinations(point, 2):
                self._cells[i].connect(self._cells[j], (i, j))
                self._cells[j].connect(self._cells[i], (j, i))

    def _validate_parameters(self) -> None:
        if self.capacity is not None and not isinstance(self.capacity, float | int):
            raise ValueError("Capacity must be a number or None.")
        if not isinstance(self.centroids_coordinates, Sequence) or not isinstance(
            self.centroids_coordinates[0], Sequence
        ):
            raise ValueError("Centroids should be a list of lists")
        dimension_1 = len(self.centroids_coordinates[0])
        for coordinate in self.centroids_coordinates:
            if dimension_1 != len(coordinate):
                raise ValueError("Centroid coordinates should be a homogeneous array")

    def _get_voronoi_regions(self) -> tuple:
        if self.voronoi_coordinates is None or self.regions is None:
            (
                self.voronoi_coordinates,
                self.regions,
            ) = self.triangulation.export_voronoi_regions()
        return self.voronoi_coordinates, self.regions

    @staticmethod
    def _compute_polygon_area(polygon: list) -> float:
        polygon = np.array(polygon)
        x = polygon[:, 0]
        y = polygon[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def _build_cell_polygons(self):
        coordinates, regions = self._get_voronoi_regions()
        for region in regions:
            polygon = [coordinates[i] for i in regions[region]]
            self._cells[region].properties["polygon"] = polygon
            polygon_area = self._compute_polygon_area(polygon)
            self._cells[region].properties["area"] = polygon_area
            self._cells[region].capacity = self.capacity_function(polygon_area)
