"""Test the coordinate translation between cell and position for different discrete spaces."""

import math
import random

import networkx as nx
import numpy as np
import pytest

from mesa.discrete_space import (
    HexGrid,
    Network,
    OrthogonalMooreGrid,
    VoronoiGrid,
)


def test_grid_transform():
    """Test OrthogonalGrid coordinate conversion."""
    grid = OrthogonalMooreGrid((10, 10), torus=False, random=random.Random(42))

    # Test cell_to_pos (Center of cell)
    cell = grid._cells[(5, 5)]
    pos = grid.cell_to_pos(cell)
    np.testing.assert_array_equal(pos, [5.5, 5.5])

    # Test pos_to_cell (Inside bounds)
    cell = grid.pos_to_cell([5.2, 5.8])
    assert cell.coordinate == (5, 5)

    cell = grid.pos_to_cell([0.1, 0.1])
    assert cell.coordinate == (0, 0)

    # Test Out of Bounds
    with pytest.raises(ValueError):
        grid.pos_to_cell([-1, 5])
    with pytest.raises(ValueError):
        grid.pos_to_cell([10.1, 5])

    # Torus Grid
    torus_grid = OrthogonalMooreGrid((10, 10), torus=True, random=random.Random(42))

    # Wrapping negative
    cell = torus_grid.pos_to_cell([-0.5, 5])
    assert cell.coordinate == (9, 5)

    # Wrapping positive overflow
    cell = torus_grid.pos_to_cell([10.5, 5])
    assert cell.coordinate == (0, 5)


def test_hex_grid_transform():
    """Test HexGrid Pointy-Topped coordinate conversion."""
    grid = HexGrid((10, 10), torus=False, random=random.Random(42))

    # Cell -> Pos -> Cell should return the original cell
    for cell in grid.all_cells:
        pos = grid.cell_to_pos(cell)
        found_cell = grid.pos_to_cell(pos)
        assert cell == found_cell

    # Specific Geometry Checks
    cell_0_0 = grid._cells[(0, 0)]
    np.testing.assert_array_almost_equal(grid.cell_to_pos(cell_0_0), [0.0, 0.0])

    cell_1_0 = grid._cells[(1, 0)]
    np.testing.assert_array_almost_equal(
        grid.cell_to_pos(cell_1_0), [math.sqrt(3), 0.0]
    )

    cell_0_1 = grid._cells[(0, 1)]
    expected_x = math.sqrt(3) * 0.5
    expected_y = 1.5
    np.testing.assert_array_almost_equal(
        grid.cell_to_pos(cell_0_1), [expected_x, expected_y]
    )


def test_voronoi_transform():
    """Test VoronoiGrid."""
    centroids = [(10, 10), (20, 10), (50, 50)]
    grid = VoronoiGrid(centroids, random=random.Random(42))

    # cell_to_pos should return the centroid
    cell_0 = grid._cells[0]
    np.testing.assert_array_equal(grid.cell_to_pos(cell_0), [10, 10])

    # pos_to_cell exactly at centroid should return the cell
    assert grid.pos_to_cell([10, 10]) == cell_0

    # Nearest Neighbor Lookup
    assert grid.pos_to_cell([12, 10]) == grid._cells[0]
    assert grid.pos_to_cell([18, 10]) == grid._cells[1]
    assert grid.pos_to_cell([40, 40]) == grid._cells[2]


def test_network_transform():
    """Test Network."""
    G = nx.Graph()  # noqa: N806
    G.add_node(0, pos=(0, 0))
    G.add_node(1, pos=(10, 0))

    net = Network(G, random=random.Random(42))

    # cell_to_pos
    cell_0 = net._cells[0]
    np.testing.assert_array_equal(net.cell_to_pos(cell_0), [0, 0])

    # pos_to_cell (Nearest Node)
    assert net.pos_to_cell([1, 1]) == cell_0
    assert net.pos_to_cell([9, 1]) == net._cells[1]

    G_for_layout = nx.path_graph(3)  # noqa: N806
    net_layout = Network(
        G_for_layout, layout=nx.spring_layout, random=random.Random(42)
    )

    # Should have generated positions
    assert net_layout._cells[0].position is not None
    # Should support lookup
    assert net_layout.pos_to_cell([0, 0]) is not None

    G_for_topo = nx.path_graph(3)  # noqa: N806
    net_topo = Network(G_for_topo, layout=None, random=random.Random(42))

    # Positions should be None
    assert net_topo._cells[0].position is None
    assert net_topo.cell_to_pos(net_topo._cells[0]) is None

    # Lookup should raise error
    with pytest.raises(ValueError):
        net_topo.pos_to_cell([0, 0])


def test_polymorphism():
    """Test that all spaces adhere to the DiscreteSpace interface."""
    spaces = [
        OrthogonalMooreGrid((5, 5), random=random.Random(42)),
        HexGrid((5, 5), random=random.Random(42)),
        VoronoiGrid([(0, 0), (10, 10)], random=random.Random(42)),
        Network(nx.path_graph(3), layout=nx.spring_layout, random=random.Random(42)),
    ]

    for space in spaces:
        # All should support cell_to_pos
        cell = space.select_random_empty_cell()
        pos = space.cell_to_pos(cell)
        assert isinstance(pos, (np.ndarray, list, tuple))

        # All (except topo network) should support pos_to_cell
        found_cell = space.pos_to_cell(pos)
        assert cell == found_cell
