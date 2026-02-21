"""Test the spatial lookups and physical positioning of different discrete spaces."""

import math
import random

import networkx as nx
import numpy as np
import pytest

from mesa.discrete_space import (
    Cell,
    HexGrid,
    Network,
    OrthogonalMooreGrid,
    VoronoiGrid,
)


def test_grid_lookups():
    """Test OrthogonalGrid positioning and nearest cell lookups."""
    grid = OrthogonalMooreGrid((10, 10), torus=False, random=random.Random(42))

    cell = grid._cells[(5, 5)]
    np.testing.assert_array_equal(cell.position, [5, 5])

    found_cell = grid.find_nearest_cell([5.2, 5.8])
    assert found_cell.coordinate == (5, 5)

    found_cell = grid.find_nearest_cell([0.1, 0.1])
    assert found_cell.coordinate == (0, 0)

    with pytest.raises(ValueError):
        grid.find_nearest_cell([-1, 5])
    with pytest.raises(ValueError):
        grid.find_nearest_cell([10.1, 5])

    torus_grid = OrthogonalMooreGrid((10, 10), torus=True, random=random.Random(42))

    cell = torus_grid.find_nearest_cell([-0.5, 5])
    assert cell.coordinate == (9, 5)

    cell = torus_grid.find_nearest_cell([10.5, 5])
    assert cell.coordinate == (0, 5)


def test_hex_grid_lookups():
    """Test HexGrid Pointy-Topped positioning, KD-Tree, and Torus math."""
    grid = HexGrid((4, 4), torus=False, random=random.Random(42))

    for cell in grid.all_cells:
        pos = cell.position
        found_cell = grid.find_nearest_cell(pos)
        assert cell == found_cell

    # Specific Geometry Checks (Odd-R Pointy-Topped)
    cell_0_0 = grid._cells[(0, 0)]
    np.testing.assert_array_almost_equal(cell_0_0.position, [0.0, 0.0])

    cell_1_0 = grid._cells[(1, 0)]
    np.testing.assert_array_almost_equal(cell_1_0.position, [math.sqrt(3), 0.0])

    cell_0_1 = grid._cells[(0, 1)]
    expected_x = math.sqrt(3) * 0.5
    expected_y = 1.5
    np.testing.assert_array_almost_equal(cell_0_1.position, [expected_x, expected_y])

    torus_hex = HexGrid((4, 4), torus=True, random=random.Random(42))
    width_px = 4 * math.sqrt(3)

    wrapped_cell = torus_hex.find_nearest_cell([-0.1, 0])
    expected_wrapped_cell = torus_hex.find_nearest_cell([width_px - 0.1, 0])
    assert wrapped_cell == expected_wrapped_cell


def test_voronoi_lookups():
    """Test VoronoiGrid positioning and KD-Tree lookups."""
    centroids = [(10, 10), (20, 10), (50, 50)]
    grid = VoronoiGrid(centroids, random=random.Random(42))

    cell_0 = grid._cells[0]
    np.testing.assert_array_equal(cell_0.position, [10, 10])

    assert grid.find_nearest_cell([10, 10]) == cell_0

    assert grid.find_nearest_cell([12, 10]) == grid._cells[0]
    assert grid.find_nearest_cell([18, 10]) == grid._cells[1]
    assert grid.find_nearest_cell([40, 40]) == grid._cells[2]


def test_network_lookups():
    """Test Network spatial vs topological modes and dynamic KD-Tree updates."""
    G = nx.Graph()  # noqa: N806
    G.add_node(0)
    G.add_node(1)

    with pytest.raises(ValueError):
        _ = Network(G, layout=[], random=random.Random(42))

    layout_dict = {0: (0, 0), 1: (10, 0)}
    net = Network(G, layout=layout_dict, random=random.Random(42))

    # Test spatial cell position
    cell_0 = net._cells[0]
    np.testing.assert_array_equal(cell_0.position, [0, 0])

    assert net.find_nearest_cell([1, 1]) == cell_0
    assert net.find_nearest_cell([9, 1]) == net._cells[1]

    new_cell = Cell(
        coordinate=99, position=np.array([100, 100]), random=random.Random(42)
    )
    net.add_cell(new_cell)

    assert net.find_nearest_cell([101, 101]).coordinate == 99

    net.remove_cell(new_cell)
    assert net.find_nearest_cell([101, 101]).coordinate != 99

    G_for_layout = nx.path_graph(3)  # noqa: N806
    net_layout = Network(
        G_for_layout, layout=nx.spring_layout, random=random.Random(42)
    )
    assert net_layout._cells[0].position is not None
    assert net_layout.find_nearest_cell([0, 0]) is not None


def test_all_spaces():
    """Test that all spaces adhere to the DiscreteSpace interface."""
    spaces = [
        OrthogonalMooreGrid((5, 5), random=random.Random(42)),
        HexGrid((4, 4), random=random.Random(42)),
        VoronoiGrid([(0, 0), (10, 10)], random=random.Random(42)),
        Network(nx.path_graph(3), layout=nx.spring_layout, random=random.Random(42)),
    ]

    for space in spaces:
        cell = space.select_random_empty_cell()
        pos = cell.position
        assert isinstance(pos, (np.ndarray, list, tuple))

        found_cell = space.find_nearest_cell(pos)
        assert cell == found_cell
