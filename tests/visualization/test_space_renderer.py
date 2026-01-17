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

    def __init__(self, rng=None):  # noqa: D107
        super().__init__(rng=rng)
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
    sr.setup_propertylayer(style).draw_propertylayer()

    # Verify that the backend renderer's draw_propertylayer was called
    sr.backend_renderer.draw_propertylayer.assert_called_once()

    # Verify that the portrayal passed to the backend is a callable that returns the style
    call_args = sr.backend_renderer.draw_propertylayer.call_args
    portrayal_arg = call_args[0][2]
    assert callable(portrayal_arg)
    assert portrayal_arg("any_layer") == style


def test_network_non_contiguous_nodes():
    """Test network with non-contiguous node IDs (Issue #3023).

    Verifies dictionary lookup correctly maps agents to positions
    regardless of node ID values.
    """
    mock_graph = MagicMock()
    mock_graph.nodes = [0, 1, 5, 10, 15]  # Non-contiguous node IDs

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

        args = {
            "loc": np.array([[0, 0], [1, 1], [5, 5], [10, 10], [15, 15]], dtype=float)
        }

        mapped = sr._map_coordinates(args)

        assert mapped["loc"].shape == (5, 2)
        assert not np.any(np.isnan(mapped["loc"]))
        expected = np.array(
            [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]
        )
        np.testing.assert_array_equal(mapped["loc"], expected)


def test_network_missing_nodes_warning():
    """Test warning when many nodes missing from layout (Issue #3064).

    Verifies NaN masking for missing nodes and warning threshold (>10%).
    """
    mock_graph = MagicMock()
    mock_graph.nodes = list(range(10))

    model = CustomModel()
    network = Network(G=mock_graph, random=random.Random(42))

    with patch.object(model, "grid", new=network):
        sr = SpaceRenderer(model)
        # Only 5 of 10 nodes in layout (50% missing, triggers warning)
        sr.space_drawer.pos = {i: np.array([float(i), 0.0]) for i in range(5)}

        args = {"loc": np.array([[i, i] for i in range(10)], dtype=float)}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mapped = sr._map_coordinates(args)

            assert len(w) == 1
            assert "5/10 agents" in str(w[0].message)
            # First 5 mapped, last 5 are NaN
            for i in range(5):
                np.testing.assert_array_equal(mapped["loc"][i], [float(i), 0.0])
            for i in range(5, 10):
                assert np.all(np.isnan(mapped["loc"][i]))


def test_network_race_condition_graceful():
    """Test graceful handling when layout lags behind simulation.

    Combines both fixes: dictionary lookup + NaN masking for resilience.
    """
    mock_graph = MagicMock()
    mock_graph.nodes = [0, 1, 2, 50, 100]

    model = CustomModel()
    network = Network(G=mock_graph, random=random.Random(42))

    with patch.object(model, "grid", new=network):
        sr = SpaceRenderer(model)
        # Layout only has nodes 0-2 (stale)
        sr.space_drawer.pos = {
            0: np.array([0.0, 0.0]),
            1: np.array([1.0, 0.0]),
            2: np.array([2.0, 0.0]),
        }

        args = {
            "loc": np.array([[0, 0], [1, 1], [2, 2], [50, 50], [100, 100]], dtype=float)
        }

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mapped = sr._map_coordinates(args)

            # Should warn since 2/5 = 40% > 10% threshold
            assert len(w) == 1
            assert "2/5 agents" in str(w[0].message)

        assert mapped["loc"].shape == (5, 2)
        # Known nodes mapped correctly
        np.testing.assert_array_equal(mapped["loc"][0], [0.0, 0.0])
        np.testing.assert_array_equal(mapped["loc"][1], [1.0, 0.0])
        np.testing.assert_array_equal(mapped["loc"][2], [2.0, 0.0])
        # Missing nodes become NaN (hidden, not crashed)
        assert np.all(np.isnan(mapped["loc"][3]))
        assert np.all(np.isnan(mapped["loc"][4]))
