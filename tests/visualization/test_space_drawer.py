"""Test space drawer classes for various grid types."""

import random
from unittest.mock import MagicMock, patch

import altair as alt
import networkx as nx
import pytest
from matplotlib.collections import LineCollection

from mesa.discrete_space import HexGrid, Network, OrthogonalMooreGrid, VoronoiGrid
from mesa.experimental.continuous_space import ContinuousSpace
from mesa.visualization.space_drawers import (
    ContinuousSpaceDrawer,
    HexSpaceDrawer,
    NetworkSpaceDrawer,
    OrthogonalSpaceDrawer,
    VoronoiSpaceDrawer,
)


@pytest.fixture
def orthogonal_grid():  # noqa: D103
    return OrthogonalMooreGrid([5, 5], random=random.Random(42))


@pytest.fixture
def hex_grid():  # noqa: D103
    return HexGrid([5, 5], random=random.Random(42))


@pytest.fixture
def continuous_space():  # noqa: D103
    return ContinuousSpace(((0, 10), (0, 10)), torus=False, random=random.Random(42))


@pytest.fixture
def network_grid():  # noqa: D103
    G = nx.Graph()  # noqa: N806
    G.add_nodes_from([0, 1, 2])
    G.add_edges_from([(0, 1), (1, 2)])
    return Network(G, random=random.Random(42))


@pytest.fixture
def voronoi_space():  # noqa: D103
    points = [(0, 0), (1, 0), (0, 1), (1, 1)]
    return VoronoiGrid(points, random=random.Random(42))


class TestOrthogonalSpaceDrawer:
    """Test cases for OrthogonalSpaceDrawer class."""

    def test_init(self, orthogonal_grid):  # noqa: D102
        drawer = OrthogonalSpaceDrawer(orthogonal_grid)
        assert drawer.space == orthogonal_grid
        assert hasattr(drawer, "s_default")

    @patch("matplotlib.pyplot.subplots")
    def test_draw_matplotlib(self, mock_subplots, orthogonal_grid):  # noqa: D102
        mock_ax = MagicMock()
        mock_fig = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        drawer = OrthogonalSpaceDrawer(orthogonal_grid)
        ax = drawer.draw_matplotlib()

        assert ax == mock_ax
        # Should draw grid lines for each row/col + 1
        assert mock_ax.axvline.call_count == orthogonal_grid.width + 1
        assert mock_ax.axhline.call_count == orthogonal_grid.height + 1
        mock_ax.set_xlim.assert_called_with(drawer.viz_xmin, drawer.viz_xmax)
        mock_ax.set_ylim.assert_called_with(drawer.viz_ymin, drawer.viz_ymax)

    def test_draw_altair(self, orthogonal_grid):  # noqa: D102
        drawer = OrthogonalSpaceDrawer(orthogonal_grid)
        chart = drawer.draw_altair()
        assert isinstance(chart, alt.Chart)
        assert chart.encoding.x.shorthand == "X:Q"
        assert chart.encoding.y.shorthand == "Y:Q"
        assert chart.mark.type == "point"

    def test_get_viz_limits(self, orthogonal_grid):  # noqa: D102
        drawer = OrthogonalSpaceDrawer(orthogonal_grid)
        limits = drawer.get_viz_limits()
        assert limits == (-0.5, 4.5, -0.5, 4.5)

    def test_draw_matplotlib_with_custom_ax(self, orthogonal_grid):  # noqa: D102
        mock_ax = MagicMock()
        drawer = OrthogonalSpaceDrawer(orthogonal_grid)
        ax = drawer.draw_matplotlib(ax=mock_ax)
        assert ax == mock_ax


class TestHexSpaceDrawer:
    """Test cases for HexSpaceDrawer class."""

    def test_init(self, hex_grid):  # noqa: D102
        drawer = HexSpaceDrawer(hex_grid)
        assert drawer.space == hex_grid
        assert len(drawer.hexagons) == hex_grid.width * hex_grid.height
        assert hasattr(drawer, "x_spacing")
        assert hasattr(drawer, "y_spacing")
        assert hasattr(drawer, "s_default")

    def test_get_unique_edges(self, hex_grid):  # noqa: D102
        drawer = HexSpaceDrawer(hex_grid)
        edges = drawer._get_unique_edges()
        assert isinstance(edges, set)
        assert all(isinstance(e, tuple) for e in edges)

    @patch("matplotlib.pyplot.subplots")
    def test_draw_matplotlib(self, mock_subplots, hex_grid):  # noqa: D102
        mock_ax = MagicMock()
        mock_fig = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        drawer = HexSpaceDrawer(hex_grid)
        ax = drawer.draw_matplotlib(ax=mock_ax)

        assert ax == mock_ax
        assert mock_ax.add_collection.called
        collection = mock_ax.add_collection.call_args[0][0]
        assert isinstance(collection, LineCollection)

    def test_draw_altair(self, hex_grid):  # noqa: D102
        drawer = HexSpaceDrawer(hex_grid)
        chart = drawer.draw_altair()
        assert isinstance(chart, alt.Chart)
        assert chart.mark.type == "line"
        assert chart.encoding.detail.shorthand == "edge_id:N"

    def test_get_viz_limits(self, hex_grid):  # noqa: D102
        drawer = HexSpaceDrawer(hex_grid)
        limits = drawer.get_viz_limits()
        assert len(limits) == 4
        assert all(isinstance(limit, int | float) for limit in limits)


class TestNetworkSpaceDrawer:
    """Test cases for NetworkSpaceDrawer class."""

    def test_init(self, network_grid):  # noqa: D102
        drawer = NetworkSpaceDrawer(network_grid)
        assert drawer.space == network_grid
        assert hasattr(drawer, "s_default")

    @patch("networkx.draw_networkx_nodes")
    @patch("networkx.draw_networkx_edges")
    def test_draw_matplotlib(self, mock_edges, mock_nodes, network_grid):  # noqa: D102
        mock_ax = MagicMock()
        drawer = NetworkSpaceDrawer(network_grid)
        drawer.draw_matplotlib(ax=mock_ax)

        mock_nodes.assert_called_once()
        mock_edges.assert_called_once()
        mock_ax.set_axis_off.assert_called_once()

    def test_draw_altair(self, network_grid):  # noqa: D102
        drawer = NetworkSpaceDrawer(network_grid)
        chart = drawer.draw_altair()
        assert isinstance(chart, alt.LayerChart)
        assert len(chart.layer) == 2

    def test_get_viz_limits(self, network_grid):  # noqa: D102
        drawer = NetworkSpaceDrawer(network_grid)
        limits = drawer.get_viz_limits()
        assert len(limits) == 4
        assert all(isinstance(limit, int | float) for limit in limits)

    def test_network_layout(self, network_grid):  # noqa: D102
        drawer = NetworkSpaceDrawer(network_grid)
        # Test that positions are calculated for all nodes
        assert len(drawer.pos) == len(network_grid.G.nodes)
        # Test that positions are 2D coordinates
        for pos in drawer.pos.values():
            assert len(pos) == 2
            assert all(isinstance(coord, int | float) for coord in pos)


class TestContinuousSpaceDrawer:
    """Test cases for ContinuousSpaceDrawer class."""

    def test_init(self, continuous_space):  # noqa: D102
        drawer = ContinuousSpaceDrawer(continuous_space)
        assert drawer.space == continuous_space
        assert hasattr(drawer, "s_default")

    @patch("matplotlib.pyplot.subplots")
    def test_draw_matplotlib(self, mock_subplots, continuous_space):  # noqa: D102
        mock_ax = MagicMock()
        mock_fig = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        drawer = ContinuousSpaceDrawer(continuous_space)
        ax = drawer.draw_matplotlib(ax=mock_ax)

        assert ax == mock_ax
        mock_ax.set_xlim.assert_called_with(drawer.viz_xmin, drawer.viz_xmax)
        mock_ax.set_ylim.assert_called_with(drawer.viz_ymin, drawer.viz_ymax)

    def test_draw_altair(self, continuous_space):  # noqa: D102
        drawer = ContinuousSpaceDrawer(continuous_space)
        chart = drawer.draw_altair()
        assert isinstance(chart, alt.Chart)
        assert chart.mark.type == "rect"

    def test_get_viz_limits(self, continuous_space):  # noqa: D102
        drawer = ContinuousSpaceDrawer(continuous_space)
        limits = drawer.get_viz_limits()
        assert len(limits) == 4
        assert all(isinstance(limit, int | float) for limit in limits)

    def test_continuous_space_with_custom_bounds(self):  # noqa: D102
        # Test with custom x_min, y_min
        space = ContinuousSpace(
            ((5, 20), (3, 15)), torus=False, random=random.Random(42)
        )
        drawer = ContinuousSpaceDrawer(space)

        expected_xmin = space.x_min - space.width / 20
        expected_xmax = space.x_max + space.width / 20
        expected_ymin = space.y_min - space.height / 20
        expected_ymax = space.y_max + space.height / 20

        assert drawer.viz_xmin == expected_xmin
        assert drawer.viz_xmax == expected_xmax
        assert drawer.viz_ymin == expected_ymin
        assert drawer.viz_ymax == expected_ymax


class TestVoronoiSpaceDrawer:
    """Test cases for VoronoiSpaceDrawer class."""

    def test_init(self, voronoi_space):  # noqa: D102
        drawer = VoronoiSpaceDrawer(voronoi_space)
        assert drawer.space == voronoi_space
        assert hasattr(drawer, "s_default")

    def test_clip_line(self, voronoi_space):  # noqa: D102
        drawer = VoronoiSpaceDrawer(voronoi_space)
        box = (0, 0, 1, 1)

        # Line inside
        result = drawer._clip_line((0.1, 0.1), (0.9, 0.9), box)
        assert result is not None
        assert len(result) == 2

        # Line outside
        result = drawer._clip_line((1.1, 1.1), (1.9, 1.9), box)
        assert result is None

        # Line partially inside
        result = drawer._clip_line((0.5, 0.5), (1.5, 0.5), box)
        assert result is not None

    @patch("matplotlib.pyplot.subplots")
    def test_draw_matplotlib(self, mock_subplots, voronoi_space):  # noqa: D102
        mock_ax = MagicMock()
        mock_fig = MagicMock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        drawer = VoronoiSpaceDrawer(voronoi_space)
        ax = drawer.draw_matplotlib(ax=mock_ax)

        assert ax == mock_ax
        assert mock_ax.add_collection.called
        collection = mock_ax.add_collection.call_args[0][0]
        assert isinstance(collection, LineCollection)

    def test_draw_altair(self, voronoi_space):  # noqa: D102
        drawer = VoronoiSpaceDrawer(voronoi_space)
        chart = drawer.draw_altair()
        assert isinstance(chart, alt.Chart)
        assert chart.mark.type == "line"
        assert "line_id" in chart.encoding.detail.shorthand

    def test_get_viz_limits(self, voronoi_space):  # noqa: D102
        drawer = VoronoiSpaceDrawer(voronoi_space)
        limits = drawer.get_viz_limits()
        assert len(limits) == 4
        assert all(isinstance(limit, int | float) for limit in limits)

    def test_get_clipped_segments(self, voronoi_space):  # noqa: D102
        drawer = VoronoiSpaceDrawer(voronoi_space)
        segments, clip_box = drawer._get_clipped_segments()
        assert isinstance(segments, list)
        assert isinstance(clip_box, tuple)
        assert len(clip_box) == 4


class TestEdgeCases:  # noqa: D101
    def test_single_point_voronoi_grid(self):  # noqa: D102
        single_point_voronoi = VoronoiGrid([(0.5, 0.5)], random=random.Random(42))
        drawer = VoronoiSpaceDrawer(single_point_voronoi)
        assert drawer.viz_xmin is not None
        assert drawer.viz_xmax is not None

    def test_continuous_space_zero_size(self):  # noqa: D102
        space = ContinuousSpace(((0, 0), (0, 0)), torus=False, random=random.Random(42))
        drawer = ContinuousSpaceDrawer(space)
        assert drawer.s_default == 1  # Default when width/height is 0

    def test_network_empty_graph(self):  # noqa: D102
        empty_graph = nx.Graph()
        network = Network(empty_graph, random=random.Random(42))
        drawer = NetworkSpaceDrawer(network)
        assert len(drawer.pos) == 0
        assert drawer.s_default == 1  # Default when no nodes

    def test_hex_grid_single_cell(self):  # noqa: D102
        grid = HexGrid([1, 1], random=random.Random(42))
        drawer = HexSpaceDrawer(grid)
        assert len(drawer.hexagons) == 1
        assert drawer.s_default == (180 / 1) ** 2
