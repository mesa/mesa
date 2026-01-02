"""Test cases for the SpaceRenderer class in Mesa."""

import random
import re
import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import mesa
from mesa.discrete_space import (
    HexGrid,
    Network,
    OrthogonalMooreGrid,
    PropertyLayer,
    VoronoiGrid,
)
from mesa.space import (
    ContinuousSpace,
    HexMultiGrid,
    HexSingleGrid,
    MultiGrid,
    NetworkGrid,
    SingleGrid,
)
from mesa.visualization.backends import altair_backend, matplotlib_backend
from mesa.visualization.components import PropertyLayerStyle
from mesa.visualization.space_drawers import (
    ContinuousSpaceDrawer,
    HexSpaceDrawer,
    NetworkSpaceDrawer,
    OrthogonalSpaceDrawer,
    VoronoiSpaceDrawer,
)
from mesa.visualization.space_renderer import SpaceRenderer


class CustomModel(mesa.Model):
    """A simple model for testing purposes."""

    def __init__(self, seed=None):  # noqa: D107
        super().__init__(seed=seed)
        self.grid = mesa.discrete_space.OrthogonalMooreGrid(
            [2, 2], random=random.Random(42)
        )
        self.layer = PropertyLayer("test", [2, 2], default_value=0, dtype=int)

        self.grid.add_property_layer(self.layer)


def test_backend_selection():
    """Test that the SpaceRenderer selects the correct backend."""
    model = CustomModel()
    sr = SpaceRenderer(model, backend="matplotlib")
    assert isinstance(sr.backend_renderer, matplotlib_backend.MatplotlibBackend)
    sr = SpaceRenderer(model, backend="altair")
    assert isinstance(sr.backend_renderer, altair_backend.AltairBackend)
    with pytest.raises(ValueError):
        SpaceRenderer(model, backend=None)


@pytest.mark.parametrize(
    "grid,expected_drawer",
    [
        (
            OrthogonalMooreGrid([2, 2], random=random.Random(42)),
            OrthogonalSpaceDrawer,
        ),
        (SingleGrid(width=2, height=2, torus=False), OrthogonalSpaceDrawer),
        (MultiGrid(width=2, height=2, torus=False), OrthogonalSpaceDrawer),
        (HexGrid([2, 2], random=random.Random(42)), HexSpaceDrawer),
        (HexSingleGrid(width=2, height=2, torus=False), HexSpaceDrawer),
        (HexMultiGrid(width=2, height=2, torus=False), HexSpaceDrawer),
        (Network(G=MagicMock(), random=random.Random(42)), NetworkSpaceDrawer),
        (NetworkGrid(g=MagicMock()), NetworkSpaceDrawer),
        (ContinuousSpace(x_max=2, y_max=2, torus=False), ContinuousSpaceDrawer),
        (
            VoronoiGrid([[0, 0], [1, 1]], random=random.Random(42)),
            VoronoiSpaceDrawer,
        ),
    ],
)
def test_space_drawer_selection(grid, expected_drawer):
    """Test that the SpaceRenderer selects the correct space drawer based on the grid type."""
    model = CustomModel()
    with patch.object(model, "grid", new=grid):
        sr = SpaceRenderer(model)
        assert isinstance(sr.space_drawer, expected_drawer)


def test_map_coordinates():
    """Test that the SpaceRenderer maps the coordinates correctly based on the grid type."""
    model = CustomModel()

    sr = SpaceRenderer(model)
    arr = np.array([[1, 2], [3, 4]])
    args = {"loc": arr}
    mapped = sr._map_coordinates(args)

    # same for orthogonal grids
    assert np.array_equal(mapped["loc"], arr)

    with patch.object(model, "grid", new=HexGrid([2, 2], random=random.Random(42))):
        sr = SpaceRenderer(model)
        mapped = sr._map_coordinates(args)

        assert not np.array_equal(mapped["loc"], arr)
        assert mapped["loc"].shape == arr.shape

    with patch.object(
        model, "grid", new=Network(G=MagicMock(), random=random.Random(42))
    ):
        sr = SpaceRenderer(model)
        # Patch the space_drawer.pos to provide a mapping for the test
        sr.space_drawer.pos = {0: (0, 0), 1: (1, 1), 2: (2, 2), 3: (3, 3)}
        mapped = sr._map_coordinates(args)

        assert not np.array_equal(mapped["loc"], arr)
        assert mapped["loc"].shape == arr.shape


def test_render_calls():
    """Test that the render method calls the appropriate drawing methods."""
    model = CustomModel()
    sr = SpaceRenderer(model)

    sr.draw_structure = MagicMock()
    sr.draw_agents = MagicMock()
    sr.draw_propertylayer = MagicMock()

    sr.setup_agents(agent_portrayal=lambda _: {}).setup_propertylayer(
        propertylayer_portrayal=lambda _: PropertyLayerStyle(color="red")
    ).render()

    sr.draw_structure.assert_called_once()
    sr.draw_agents.assert_called_once()
    sr.draw_propertylayer.assert_called_once()


def test_no_property_layers():
    """Test to confirm the SpaceRenderer raises an exception when no property layers are found."""
    model = CustomModel()
    sr = SpaceRenderer(model)

    # Simulate missing property layer in the grid
    with (
        patch.object(model.grid, "_mesa_property_layers", new={}),
        pytest.raises(
            Exception, match=re.escape("No property layers were found on the space.")
        ),
    ):
        sr.setup_propertylayer(
            lambda _: PropertyLayerStyle(color="red")
        ).draw_propertylayer()


def test_post_process():
    """Test the post-processing step of the SpaceRenderer."""
    model = CustomModel()
    sr = SpaceRenderer(model)

    def post_process_ax(ax):
        ax.set_xlim(0, 400)
        ax.set_ylim(0, 400)
        return ax

    ax = MagicMock()
    sr.post_process_ax = post_process_ax
    processed = sr.post_process_ax(ax)

    # Assert that the axis limits were set correctly
    ax.set_xlim.assert_called_once_with(0, 400)
    ax.set_ylim.assert_called_once_with(0, 400)
    assert processed == ax

    def post_process_chart(chart):
        chart = chart.properties(width=400, height=400)
        return chart

    # Simulate a chart object
    chart = MagicMock()
    chart.properties.return_value = chart

    # Call the post_process method
    sr.post_process = post_process_chart
    processed = sr.post_process(chart)

    # Assert that the chart properties were set correctly
    chart.properties.assert_called_once_with(width=400, height=400)
    assert processed == chart


def test_property_layer_style_instance():
    """Test that draw_propertylayer accepts a PropertyLayerStyle instance."""
    model = CustomModel()
    sr = SpaceRenderer(model)
    sr.backend_renderer = MagicMock()

    style = PropertyLayerStyle(color="blue")
    sr.draw_propertylayer(style)

    # Verify that the backend renderer's draw_propertylayer was called
    sr.backend_renderer.draw_propertylayer.assert_called_once()

    # Verify that the portrayal passed to the backend is a callable that returns the style
    call_args = sr.backend_renderer.draw_propertylayer.call_args
    portrayal_arg = call_args[0][2]
    assert callable(portrayal_arg)
    assert portrayal_arg("any_layer") == style


def test_network_non_contiguous_nodes():
    """Test network visualization with non-contiguous node IDs."""
    # Create a network with non-contiguous node IDs
    mock_graph = MagicMock()
    mock_graph.nodes = [0, 1, 5, 10, 15]

    model = CustomModel()
    network = Network(G=mock_graph, random=random.Random(42))

    with patch.object(model, "grid", new=network):
        sr = SpaceRenderer(model)
        sr.space_drawer.pos = {
            0: np.array([0.1, 0.2]),
            1: np.array([0.3, 0.4]),
            5: np.array([0.5, 0.6]),
            10: np.array([0.7, 0.8]),
            15: np.array([0.9, 1.0]),
        }

        # Create arguments with agent positions at non-contiguous nodes
        args = {
            "loc": np.array([[0, 0], [1, 1], [5, 5], [10, 10], [15, 15]], dtype=float)
        }

        # Map coordinates
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mapped = sr._map_coordinates(args)

            # Should not have any warnings since all nodes are present
            assert len(w) == 0

            # All agents should be mapped correctly
            assert mapped["loc"].shape == (5, 2)
            expected_positions = np.array(
                [
                    [0.1, 0.2],  # Node 0
                    [0.3, 0.4],  # Node 1
                    [0.5, 0.6],  # Node 5
                    [0.7, 0.8],  # Node 10
                    [0.9, 1.0],  # Node 15
                ]
            )
            np.testing.assert_array_equal(mapped["loc"], expected_positions)


def test_network_missing_nodes_warning():
    """Test that warning is issued for missing node positions."""
    # Create a network
    mock_graph = MagicMock()
    mock_graph.nodes = [0, 1, 5, 10, 15]

    model = CustomModel()
    network = Network(G=mock_graph, random=random.Random(42))

    with patch.object(model, "grid", new=network):
        sr = SpaceRenderer(model)
        # Position dictionary missing some nodes
        sr.space_drawer.pos = {
            0: np.array([0.1, 0.2]),
            1: np.array([0.3, 0.4]),
            5: np.array([0.5, 0.6]),
            # Missing nodes 10 and 15
        }

        # Create arguments with agent positions including missing nodes
        args = {
            "loc": np.array([[0, 0], [1, 1], [5, 5], [10, 10], [15, 15]], dtype=float)
        }

        # Map coordinates and capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mapped = sr._map_coordinates(args)

            # Should have one warning about missing nodes
            assert len(w) == 1
            assert "not found in position mapping" in str(w[0].message)
            assert issubclass(w[0].category, UserWarning)

            # Should still map all agents (missing ones at origin)
            assert mapped["loc"].shape == (5, 2)
            expected_positions = np.array(
                [
                    [0.1, 0.2],  # Node 0
                    [0.3, 0.4],  # Node 1
                    [0.5, 0.6],  # Node 5
                    [0.0, 0.0],  # Node 10 (missing, default to origin)
                    [0.0, 0.0],  # Node 15 (missing, default to origin)
                ]
            )
            np.testing.assert_array_equal(mapped["loc"], expected_positions)


def test_network_single_node():
    """Test network with single node."""
    mock_graph = MagicMock()
    mock_graph.nodes = [42]  # Single non-zero node ID

    model = CustomModel()
    network = Network(G=mock_graph, random=random.Random(42))

    with patch.object(model, "grid", new=network):
        sr = SpaceRenderer(model)
        sr.space_drawer.pos = {42: np.array([1.0, 2.0])}

        args = {"loc": np.array([[42, 42]], dtype=float)}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mapped = sr._map_coordinates(args)

            assert len(w) == 0
            assert mapped["loc"].shape == (1, 2)
            np.testing.assert_array_equal(mapped["loc"], [[1.0, 2.0]])


def test_network_empty_locations():
    """Test network with empty location array."""
    mock_graph = MagicMock()
    mock_graph.nodes = []

    model = CustomModel()
    network = Network(G=mock_graph, random=random.Random(42))

    with patch.object(model, "grid", new=network):
        sr = SpaceRenderer(model)
        sr.space_drawer.pos = {}

        args = {"loc": np.array([], dtype=float).reshape(0, 2)}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mapped = sr._map_coordinates(args)

            assert len(w) == 0
            assert mapped["loc"].shape == (0, 2)


def test_network_large_node_ids():
    """Test network with very large node IDs."""
    mock_graph = MagicMock()
    mock_graph.nodes = [1000, 5000, 10000]  # Large node IDs

    model = CustomModel()
    network = Network(G=mock_graph, random=random.Random(42))

    with patch.object(model, "grid", new=network):
        sr = SpaceRenderer(model)
        sr.space_drawer.pos = {
            1000: np.array([1.0, 1.0]),
            5000: np.array([2.0, 2.0]),
            10000: np.array([3.0, 3.0]),
        }

        args = {
            "loc": np.array([[1000, 1000], [5000, 5000], [10000, 10000]], dtype=float)
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mapped = sr._map_coordinates(args)

            assert len(w) == 0
            assert mapped["loc"].shape == (3, 2)
            expected = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
            np.testing.assert_array_equal(mapped["loc"], expected)


def test_network_regression_indexerror_bug():
    """Regression test for the IndexError suppression bug."""
    # This test reproduces the exact bug scenario
    mock_graph = MagicMock()
    mock_graph.nodes = [0, 1, 5, 10, 15]

    model = CustomModel()
    network = Network(G=mock_graph, random=random.Random(42))

    with patch.object(model, "grid", new=network):
        sr = SpaceRenderer(model)
        sr.space_drawer.pos = {
            0: np.array([0.1, 0.2]),
            1: np.array([0.3, 0.4]),
            5: np.array([0.5, 0.6]),
            10: np.array([0.7, 0.8]),
            15: np.array([0.9, 1.0]),
        }

        # This would have caused IndexError in the old code
        args = {
            "loc": np.array([[0, 0], [1, 1], [5, 5], [10, 10], [15, 15]], dtype=float)
        }

        # Before fix: this would return None or raise suppressed IndexError
        # After fix: this should return proper mapping
        mapped = sr._map_coordinates(args)

        # Verify all agents are mapped (no silent failures)
        assert mapped["loc"] is not None
        assert mapped["loc"].shape == (5, 2)
        assert not np.any(np.isnan(mapped["loc"]))  # No NaN values

        # Verify specific positions
        expected = np.array(
            [
                [0.1, 0.2],  # Node 0
                [0.3, 0.4],  # Node 1
                [0.5, 0.6],  # Node 5
                [0.7, 0.8],  # Node 10
                [0.9, 1.0],  # Node 15
            ]
        )
        np.testing.assert_array_equal(mapped["loc"], expected)
