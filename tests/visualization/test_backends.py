"""Test the backends of the visualization package."""

import random
import types
from typing import ClassVar
from unittest.mock import MagicMock

import numpy as np
import pytest

from mesa.discrete_space.grid import OrthogonalMooreGrid
from mesa.discrete_space.property_layer import PropertyLayer
from mesa.visualization.backends import AltairBackend, MatplotlibBackend
from mesa.visualization.components import AgentPortrayalStyle, PropertyLayerStyle


def test_matplotlib_initialize_canvas():
    """Test that MatplotlibBackend initializes canvas with ax and fig."""
    mb = MatplotlibBackend(space_drawer=MagicMock())
    mb.initialize_canvas()
    assert mb.ax is not None
    assert mb.fig is not None


def test_matplotlib_initialize_canvas_with_custom_ax():
    """Test initializing canvas with a provided ax skips creating fig."""
    mb = MatplotlibBackend(space_drawer=MagicMock())
    ax = MagicMock()
    mb.initialize_canvas(ax=ax)
    assert mb.ax == ax
    assert not hasattr(mb, "fig")


def test_matplotlib_backend_draw_structure(monkeypatch):
    """Test draw_structure returns ax from draw_matplotlib."""
    mb = MatplotlibBackend(space_drawer=MagicMock())
    mb.initialize_canvas()
    ax = MagicMock()
    monkeypatch.setattr(mb, "ax", ax)
    mb.space_drawer.draw_matplotlib = MagicMock(return_value=ax)
    assert mb.draw_structure() == ax


def test_matplotlib_backend_collects_agent_data():
    """Test collect_agent_data."""
    mb = MatplotlibBackend(space_drawer=MagicMock())

    class DummyAgent:
        pos = (0, 0)
        cell = types.SimpleNamespace(coordinate=(0, 0))

    class DummySpace:
        agents: ClassVar[list] = [DummyAgent()]

    # Test with AgentPortrayalStyle
    def agent_portrayal_style(agent):
        return AgentPortrayalStyle(
            x=0,
            y=0,
            size=5,
            color="red",
            marker="o",
            zorder=1,
            alpha=1.0,
            edgecolors="black",
            linewidths=1,
        )

    data = mb.collect_agent_data(DummySpace(), agent_portrayal_style)
    assert "loc" in data and data["loc"].shape[0] == 1

    # Test with dict-based portrayal (deprecated, emits FutureWarning)
    def agent_portrayal_dict(agent):
        return {"size": 5, "color": "red", "marker": "o"}

    with pytest.warns(FutureWarning):
        data = mb.collect_agent_data(DummySpace(), agent_portrayal_dict)

    assert "loc" in data and data["loc"].shape[0] == 1


def test_matplotlib_backend_draw_agents():
    """Test drawing agents."""
    mb = MatplotlibBackend(space_drawer=MagicMock())
    mb.initialize_canvas()

    # Test with empty data
    arguments = {"loc": np.array([]), "marker": np.array([]), "zorder": np.array([])}
    result = mb.draw_agents(arguments)
    assert result is None

    # Test with data
    arguments = {
        "loc": np.array([[0, 0], [1, 1]]),
        "marker": np.array(["o", "s"]),
        "zorder": np.array([1, 1]),
        "s": np.array([5, 5]),
        "c": np.array(["red", "blue"]),
        "alpha": np.array([1.0, 1.0]),
        "edgecolors": np.array(["black", "black"]),
        "linewidths": np.array([1, 1]),
    }
    result = mb.draw_agents(arguments)
    assert result == mb.ax


def test_matplotlib_backend_draw_agents_bad_marker(monkeypatch):
    """Test drawing agents with nonexistent marker file raises ValueError."""
    mb = MatplotlibBackend(space_drawer=MagicMock())
    mb.initialize_canvas()
    monkeypatch.setattr("os.path.isfile", lambda path: False)
    arguments = {
        "loc": np.array([[0, 0]]),
        "marker": np.array(["notafile.png"], dtype=object),
        "zorder": np.array([1]),
        "s": np.array([1]),
        "c": np.array(["red"]),
        "alpha": np.array([1.0]),
        "edgecolors": np.array(["black"]),
        "linewidths": np.array([1]),
    }
    with pytest.raises(ValueError):
        mb.draw_agents(arguments.copy())


def test_matplotlib_backend_draw_propertylayer():
    """Test drawing property layer."""
    # Test with color
    mb = MatplotlibBackend(space_drawer=MagicMock())
    mb.initialize_canvas()

    # set up space and layer
    space = OrthogonalMooreGrid([2, 2], random=random.Random(42))
    layer = PropertyLayer("test", [2, 2], default_value=0.0)
    space.add_property_layer(layer)

    # Test with color
    def propertylayer_portrayal_color(layer):
        return PropertyLayerStyle(
            color="red", alpha=0.5, vmin=0, vmax=1, colorbar=False
        )

    result = mb.draw_propertylayer(
        space, space._mesa_property_layers, propertylayer_portrayal_color
    )
    assert result[0] == mb.ax
    assert result[1] is None

    # Test with colormap
    def propertylayer_portrayal_colormap(layer):
        return PropertyLayerStyle(
            colormap="viridis", alpha=0.5, vmin=0, vmax=1, colorbar=True
        )

    result = mb.draw_propertylayer(
        space, space._mesa_property_layers, propertylayer_portrayal_colormap
    )
    assert result[0] == mb.ax
    assert result[1] is not None

    # Test with no color or colormap
    def propertylayer_portrayal_no_color_colormap(layer):
        return PropertyLayerStyle(
            color=None, colormap=None, alpha=1.0, vmin=0, vmax=1, colorbar=False
        )

    with pytest.raises(ValueError, match="Specify one of 'color' or 'colormap'"):
        mb.draw_propertylayer(
            space,
            space._mesa_property_layers,
            propertylayer_portrayal_no_color_colormap,
        )


def test_altair_backend_draw_structure():
    """Test AltairBackend draw_structure returns chart."""
    ab = AltairBackend(space_drawer=MagicMock())
    ab.space_drawer.draw_altair = MagicMock(return_value="chart")
    assert ab.draw_structure() == "chart"


def test_altair_backend_collects_agent_data():
    """Test collect_agent_data."""
    ab = AltairBackend(space_drawer=MagicMock())

    class DummyAgent:
        pos = (0, 0)
        cell = types.SimpleNamespace(coordinate=(0, 0))

    class DummySpace:
        agents: ClassVar[list] = [DummyAgent()]

    # Test with AgentPortrayalStyle
    def agent_portrayal_style(agent):
        return AgentPortrayalStyle(
            x=0,
            y=0,
            size=5,
            color="red",
            marker="o",
            zorder=1,
            alpha=1.0,
            edgecolors="black",
            linewidths=1,
        )

    data = ab.collect_agent_data(DummySpace(), agent_portrayal_style)
    assert "loc" in data and data["loc"].shape[0] == 1

    # Test with dict-based portrayal (deprecated, emits FutureWarning)
    def agent_portrayal_dict(agent):
        return {"size": 5, "color": "red", "marker": "o"}

    with pytest.warns(FutureWarning):
        data = ab.collect_agent_data(DummySpace(), agent_portrayal_dict)

    assert "loc" in data and data["loc"].shape[0] == 1


def test_altair_backend_collects_agent_data_marker_mapping():
    """Test collect_agent_data maps markers to Altair shapes."""
    ab = AltairBackend(space_drawer=MagicMock())

    class DummyAgent:
        pos = (0, 0)
        cell = types.SimpleNamespace(coordinate=(0, 0))

    class DummySpace:
        agents: ClassVar[list] = [DummyAgent()]

    def agent_portrayal(agent):
        return AgentPortrayalStyle(
            x=0, y=0, size=5, color="red", marker="s", zorder=1, alpha=1.0
        )

    data = ab.collect_agent_data(DummySpace(), agent_portrayal)
    assert data["shape"][0] == "square"


def test_altair_backend_draw_agents():
    """Test draw_agents."""
    # Test with empty data
    ab = AltairBackend(space_drawer=MagicMock())
    result = ab.draw_agents({"loc": np.array([])})
    assert result is None

    # Test with data
    arguments = {
        "loc": np.array([[0, 0], [1, 1]]),
        "size": np.array([5, 5]),
        "shape": np.array(["circle", "square"]),
        "opacity": np.array([1.0, 1.0]),
        "strokeWidth": np.array([1, 1]),
        "color": np.array(["red", "blue"]),
        "filled": np.array([True, True]),
        "stroke": np.array(["black", "black"]),
    }
    ab.space_drawer.get_viz_limits = MagicMock(return_value=(0, 10, 0, 10))
    assert ab.draw_agents(arguments) is not None


def test_altair_backend_draw_propertylayer():
    """Test drawing propertylayer."""
    ab = AltairBackend(space_drawer=MagicMock())

    # set up space and layer
    space = OrthogonalMooreGrid([2, 2], random=random.Random(42))
    layer = PropertyLayer("test", [2, 2], default_value=0.0)
    space.add_property_layer(layer)

    # Test with color
    def propertylayer_portrayal_color(layer):
        return PropertyLayerStyle(
            color="red", alpha=0.5, vmin=0, vmax=1, colorbar=False
        )

    result = ab.draw_propertylayer(
        space, space._mesa_property_layers, propertylayer_portrayal_color
    )
    assert result is not None

    # Test with colormap
    def propertylayer_portrayal_colormap(layer):
        return PropertyLayerStyle(
            colormap="viridis", alpha=0.5, vmin=0, vmax=1, colorbar=True
        )

    result = ab.draw_propertylayer(
        space, space._mesa_property_layers, propertylayer_portrayal_colormap
    )
    assert result is not None

    # Test with no color or colormap
    def propertylayer_portrayal(layer):
        return PropertyLayerStyle(
            color=None, colormap=None, alpha=1.0, vmin=0, vmax=1, colorbar=False
        )

    with pytest.raises(ValueError, match="Specify one of 'color' or 'colormap'"):
        ab.draw_propertylayer(
            space, space._mesa_property_layers, propertylayer_portrayal
        )


def test_backend_get_agent_pos():
    """Test extracting agent position from pos and cell.coordinate attributes."""
    mb = MatplotlibBackend(space_drawer=MagicMock())

    class AgentWithPos:
        pos = (1, 2)

    x, y = mb._get_agent_pos(AgentWithPos(), None)
    assert (x, y) == (1, 2)

    class AgentWithCell:
        pos = None
        cell = types.SimpleNamespace(coordinate=(3, 4))

    x, y = mb._get_agent_pos(AgentWithCell(), None)
    assert (x, y) == (3, 4)


def test_backends_handle_errors():
    """Test error handling scenarios for invalid agent/propertylayer data."""
    mb = MatplotlibBackend(space_drawer=MagicMock())
    mb.initialize_canvas()
    arguments = {
        "loc": np.array([[0, 0]]),
        "marker": np.array(["o"]),
        "zorder": np.array([1]),
        "s": np.array([5]),
        "c": np.array(["red"]),
        "alpha": np.array([1.0]),
        "edgecolors": np.array(["black"]),
        "linewidths": np.array([1]),
    }
    with pytest.raises(ValueError):
        mb.draw_agents(arguments, edgecolors="blue")
