"""Test cases for the SpaceRenderer class in Mesa."""

import random
import re
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
from mesa.visualization.backends import altair_backend, matplotlib_backend
from mesa.visualization.components import PropertyLayerStyle
from mesa.visualization.space_drawers import (
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
        mapped = sr._map_coordinates(args)

        assert np.array_equal(mapped["loc"], arr)
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
