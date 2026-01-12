"""Tests for the Voronoi module in discrete_space."""

import random

import numpy as np
import pytest

from mesa.discrete_space.cell import Cell
from mesa.discrete_space.voronoi import Delaunay, VoronoiGrid, round_float


class TestDelaunay:
    """Tests for the Delaunay triangulation class."""

    def test_delaunay_initialization(self):
        """Test Delaunay initializes with frame."""
        delaunay = Delaunay()

        # Should have 4 corner coordinates for frame
        assert len(delaunay.coords) == 4
        # Should have initial triangles
        assert len(delaunay.triangles) > 0

    def test_delaunay_initialization_with_custom_center_radius(self):
        """Test Delaunay with custom center and radius."""
        delaunay = Delaunay(center=(5, 5), radius=100)

        # Should still have 4 corner coordinates
        assert len(delaunay.coords) == 4

    def test_delaunay_add_point(self):
        """Test adding a point to Delaunay triangulation."""
        delaunay = Delaunay()
        initial_coords = len(delaunay.coords)

        delaunay.add_point((0, 0))

        assert len(delaunay.coords) == initial_coords + 1

    def test_delaunay_add_multiple_points(self):
        """Test adding multiple points."""
        delaunay = Delaunay()

        points = [(0, 0), (1, 0), (0.5, 1)]
        for point in points:
            delaunay.add_point(point)

        assert len(delaunay.coords) == 4 + len(points)

    def test_delaunay_export_triangles(self):
        """Test exporting triangles."""
        delaunay = Delaunay()

        # Add points to form a triangle
        points = [(0, 0), (1, 0), (0.5, 1)]
        for point in points:
            delaunay.add_point(point)

        triangles = delaunay.export_triangles()

        # Should have at least one triangle
        assert len(triangles) >= 1

    def test_delaunay_export_voronoi_regions(self):
        """Test exporting Voronoi regions."""
        delaunay = Delaunay()

        # Add points
        points = [(0, 0), (1, 0), (0.5, 1), (0.5, 0.5)]
        for point in points:
            delaunay.add_point(point)

        vor_coords, regions = delaunay.export_voronoi_regions()

        # Should have Voronoi coordinates
        assert len(vor_coords) > 0
        # Should have regions for each point
        assert len(regions) == len(points)


class TestRoundFloat:
    """Tests for the round_float function."""

    def test_round_float_positive(self):
        """Test round_float with positive number."""
        assert round_float(0.5) == 250

    def test_round_float_one(self):
        """Test round_float with 1."""
        assert round_float(1.0) == 500

    def test_round_float_small(self):
        """Test round_float with small number."""
        assert round_float(0.001) == 0

    def test_round_float_zero(self):
        """Test round_float with zero."""
        assert round_float(0) == 0


class TestVoronoiGrid:
    """Tests for the VoronoiGrid class."""

    def test_voronoi_grid_initialization(self):
        """Test VoronoiGrid basic initialization."""
        coordinates = [[0, 0], [1, 0], [0, 1], [1, 1]]
        rng = random.Random(42)
        grid = VoronoiGrid(coordinates, random=rng)

        assert len(grid._cells) == 4

    def test_voronoi_grid_cells_have_coordinates(self):
        """Test VoronoiGrid cells have correct coordinates."""
        coordinates = [[0, 0], [1, 0], [0, 1], [1, 1]]
        rng = random.Random(42)
        grid = VoronoiGrid(coordinates, random=rng)

        for i, coord in enumerate(coordinates):
            assert grid._cells[i].coordinate == coord

    def test_voronoi_grid_cells_connected(self):
        """Test VoronoiGrid cells are properly connected."""
        coordinates = [[0, 0], [1, 0], [0, 1], [1, 1]]
        rng = random.Random(42)
        grid = VoronoiGrid(coordinates, random=rng)

        # Each cell should have at least one neighbor
        for cell in grid.all_cells:
            assert len(cell.connections) > 0

    def test_voronoi_grid_with_capacity(self):
        """Test VoronoiGrid with specified capacity."""
        coordinates = [[0, 0], [1, 0], [0, 1], [1, 1]]
        rng = random.Random(42)
        grid = VoronoiGrid(coordinates, capacity=5, random=rng)

        # Note: capacity gets overwritten by capacity_function in _build_cell_polygons
        # so we just check initialization doesn't fail
        assert grid.capacity == 5

    def test_voronoi_grid_custom_cell_class(self):
        """Test VoronoiGrid with custom cell class."""

        class CustomCell(Cell):
            pass

        coordinates = [[0, 0], [1, 0], [0, 1], [1, 1]]
        rng = random.Random(42)
        grid = VoronoiGrid(coordinates, cell_klass=CustomCell, random=rng)

        for cell in grid.all_cells:
            assert isinstance(cell, CustomCell)

    def test_voronoi_grid_custom_capacity_function(self):
        """Test VoronoiGrid with custom capacity function."""

        def custom_capacity(area):
            return int(area * 100)

        coordinates = [[0, 0], [1, 0], [0, 1], [1, 1]]
        rng = random.Random(42)
        grid = VoronoiGrid(coordinates, capacity_function=custom_capacity, random=rng)

        assert grid.capacity_function == custom_capacity

    def test_voronoi_grid_cells_have_polygon(self):
        """Test VoronoiGrid cells have polygon property."""
        coordinates = [[0, 0], [1, 0], [0, 1], [1, 1]]
        rng = random.Random(42)
        grid = VoronoiGrid(coordinates, random=rng)

        for cell in grid.all_cells:
            assert "polygon" in cell.properties
            assert "area" in cell.properties

    def test_voronoi_grid_all_cells(self):
        """Test VoronoiGrid all_cells property."""
        coordinates = [[0, 0], [1, 0], [0, 1], [1, 1]]
        rng = random.Random(42)
        grid = VoronoiGrid(coordinates, random=rng)

        all_cells = list(grid.all_cells)
        assert len(all_cells) == 4

    def test_voronoi_grid_invalid_capacity_raises_error(self):
        """Test VoronoiGrid raises error with invalid capacity."""
        coordinates = [[0, 0], [1, 0], [0, 1], [1, 1]]
        rng = random.Random(42)

        with pytest.raises(ValueError, match="Capacity must be a number or None"):
            VoronoiGrid(coordinates, capacity="invalid", random=rng)

    def test_voronoi_grid_invalid_coordinates_raises_error(self):
        """Test VoronoiGrid raises error with invalid coordinates."""
        rng = random.Random(42)

        # String as coordinates causes a numpy type error during processing
        with pytest.raises((ValueError, TypeError)):
            VoronoiGrid("invalid", random=rng)

    def test_voronoi_grid_inconsistent_dimensions_raises_error(self):
        """Test VoronoiGrid raises error with inconsistent dimensions."""
        coordinates = [[0, 0], [1, 0, 0]]  # Different dimensions
        rng = random.Random(42)

        with pytest.raises(ValueError, match="homogeneous array"):
            VoronoiGrid(coordinates, random=rng)

    def test_voronoi_grid_triangulation_attribute(self):
        """Test VoronoiGrid has triangulation attribute."""
        coordinates = [[0, 0], [1, 0], [0, 1], [1, 1]]
        rng = random.Random(42)
        grid = VoronoiGrid(coordinates, random=rng)

        assert grid.triangulation is not None
        assert isinstance(grid.triangulation, Delaunay)

    def test_voronoi_grid_compute_polygon_area(self):
        """Test static method _compute_polygon_area."""
        # Unit square polygon
        polygon = [[0, 0], [1, 0], [1, 1], [0, 1]]
        area = VoronoiGrid._compute_polygon_area(polygon)

        assert np.isclose(area, 1.0)

    def test_voronoi_grid_compute_polygon_area_triangle(self):
        """Test polygon area calculation for triangle."""
        # Triangle with area 0.5
        polygon = [[0, 0], [1, 0], [0, 1]]
        area = VoronoiGrid._compute_polygon_area(polygon)

        assert np.isclose(area, 0.5)

    def test_voronoi_grid_many_points(self):
        """Test VoronoiGrid with many points."""
        rng = random.Random(42)
        np.random.seed(42)

        coordinates = [
            [np.random.uniform(0, 10), np.random.uniform(0, 10)] for _ in range(20)
        ]
        grid = VoronoiGrid(coordinates, random=rng)

        assert len(grid._cells) == 20

    def test_voronoi_grid_get_voronoi_regions(self):
        """Test _get_voronoi_regions method."""
        coordinates = [[0, 0], [1, 0], [0, 1], [1, 1]]
        rng = random.Random(42)
        grid = VoronoiGrid(coordinates, random=rng)

        vor_coords, regions = grid._get_voronoi_regions()

        assert len(vor_coords) > 0
        assert len(regions) == 4  # One region per point
