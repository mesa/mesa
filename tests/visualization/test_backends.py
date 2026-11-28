"""Test the backends of the visualization package."""

import random
import types
from typing import ClassVar
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from mesa import Model
from mesa.discrete_space import HexGrid
from mesa.discrete_space.grid import OrthogonalMooreGrid
from mesa.experimental.continuous_space import ContinuousSpace, ContinuousSpaceAgent
from mesa.visualization.backends import AltairBackend, MatplotlibBackend
from mesa.visualization.backends.matplotlib_backend import _to_numpy_argument_array
from mesa.visualization.components import AgentPortrayalStyle, PropertyLayerStyle
from mesa.visualization.space_drawers import HexSpaceDrawer


@pytest.mark.parametrize("backend_cls", [MatplotlibBackend, AltairBackend])
def test_backend_get_agent_pos_raises_when_position_too_short(backend_cls):
    """Backends should reject agent positions with fewer than 2 coordinates."""
    backend = backend_cls(space_drawer=MagicMock())

    class DummySpace:
        viz_dims = (0, 1)

    class DummyAgent:
        position = (0.1,)

    with pytest.raises(ValueError, match="at least 2 dimensions"):
        backend._get_agent_pos(DummyAgent(), DummySpace())


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
        position = (0, 0)
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


@pytest.mark.parametrize("edgecolor", ["black", (1.0, 0.0, 0.0, 1.0)])
def test_matplotlib_backend_collects_partial_edgecolors(edgecolor):
    """Collecting agent data preserves edgecolor alignment across agents."""
    mb = MatplotlibBackend(space_drawer=MagicMock())

    class DummyAgent:
        position = (0, 0)
        cell = types.SimpleNamespace(coordinate=(0, 0))

        def __init__(self, kind):
            self.kind = kind

    class DummySpace:
        agents: ClassVar[list] = [DummyAgent(0), DummyAgent(1)]

    def agent_portrayal(agent):
        portrayal = {"size": 5, "color": "red", "marker": "o"}
        if agent.kind == 0:
            portrayal["edgecolors"] = edgecolor
        return portrayal

    with pytest.warns(FutureWarning):
        data = mb.collect_agent_data(DummySpace(), agent_portrayal)

    assert data["edgecolors"].tolist() == [edgecolor, "none"]
    assert len(data["edgecolors"]) == data["loc"].shape[0]


def test_matplotlib_backend_collects_tuple_markers_without_flattening():
    """Equal-length tuple marker specs must stay 1D, not collapse into a 2D array.

    Regression guard: a naive ``np.asarray(value, dtype=object)`` silently turns a
    list of same-length tuples (e.g. Matplotlib's ``(numsides, style, angle)``
    polygon marker spec) into a 2D array of scalars instead of a 1D array of
    tuple objects.
    """
    mb = MatplotlibBackend(space_drawer=MagicMock())

    class DummyAgent:
        position = (0, 0)
        cell = types.SimpleNamespace(coordinate=(0, 0))

    class DummySpace:
        agents: ClassVar[list] = [DummyAgent(), DummyAgent()]

    def agent_portrayal(agent):
        return AgentPortrayalStyle(size=5, color="red", marker=(3, 0, 0))

    data = mb.collect_agent_data(DummySpace(), agent_portrayal)

    assert data["marker"].shape == (2,)
    assert list(data["marker"]) == [(3, 0, 0), (3, 0, 0)]


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


def test_matplotlib_backend_draw_property():
    """Test drawing property layer."""
    # Test with color
    mb = MatplotlibBackend(space_drawer=MagicMock())
    mb.initialize_canvas()

    # set up space and layer
    space = OrthogonalMooreGrid([2, 2], random=random.Random(42))
    space.create_property_layer("test", default_value=0.0)

    result = mb.draw_property_layer(
        space,
        space.property_layers,
        lambda l: PropertyLayerStyle(  # noqa: E741
            color="red", alpha=0.5, vmin=0, vmax=1, colorbar=False
        ),
    )
    assert result[0] == mb.ax
    assert result[1] is None

    result = mb.draw_property_layer(
        space,
        space.property_layers,
        lambda l: PropertyLayerStyle(  # noqa: E741
            colormap="viridis", alpha=0.5, vmin=0, vmax=1, colorbar=True
        ),
    )
    assert result[0] == mb.ax
    assert result[1] is not None

    with pytest.raises(ValueError, match="Specify one of 'color' or 'colormap'"):
        mb.draw_property_layer(
            space,
            space.property_layers,
            lambda l: PropertyLayerStyle(  # noqa: E741
                color=None, colormap=None, alpha=1.0, vmin=0, vmax=1, colorbar=False
            ),
        )


def test_matplotlib_backend_preserves_colorbar_from_previous_layer(monkeypatch):
    """A layer without a colorbar does not clear the previous layer's colorbar."""
    mb = MatplotlibBackend(space_drawer=MagicMock())
    mb.initialize_canvas()

    space = OrthogonalMooreGrid([2, 2], random=random.Random(42))
    space.create_property_layer("layer_a", default_value=0.0)
    space.create_property_layer("layer_b", default_value=1.0)

    created_colorbars = []
    original_colorbar = plt.colorbar

    def track_colorbar(*args, **kwargs):
        colorbar = original_colorbar(*args, **kwargs)
        created_colorbars.append(colorbar)
        return colorbar

    monkeypatch.setattr(plt, "colorbar", track_colorbar)

    result = mb.draw_property_layer(
        space,
        space.property_layers,
        lambda layer_name: PropertyLayerStyle(
            colormap="viridis",
            alpha=0.5,
            vmin=0,
            vmax=1,
            colorbar=layer_name == "layer_a",
        ),
    )

    assert len(mb.ax.images) == 2
    assert len(created_colorbars) == 1
    assert result[0] is mb.ax
    assert result[1] is created_colorbars[0]


def test_altair_backend_draw_structure():
    """Test AltairBackend draw_structure returns chart."""
    ab = AltairBackend(space_drawer=MagicMock())
    ab.space_drawer.draw_altair = MagicMock(return_value="chart")
    assert ab.draw_structure() == "chart"


def test_altair_backend_draw_property_hexgrid_fills_hexagons():
    """A HexGrid property layer renders filled hexagons, not a square rect grid."""
    space = HexGrid([3, 3], random=random.Random(42))
    space.create_property_layer("test", default_value=0.0)
    ab = AltairBackend(space_drawer=HexSpaceDrawer(space))

    chart = ab.draw_property_layer(
        space,
        space.property_layers,
        lambda l: PropertyLayerStyle(  # noqa: E741
            colormap="viridis", alpha=0.5, vmin=0, vmax=1, colorbar=True
        ),
    )

    layer_spec = chart.to_dict()["layer"][0]
    # Hex cells are drawn as filled closed polygons (a line mark with a fill),
    # not the square rect grid used for orthogonal spaces.
    assert layer_spec["mark"]["type"] == "line"
    assert "fill" in layer_spec["encoding"]


def test_altair_backend_draw_property_orthogonal_uses_rect():
    """An orthogonal grid property layer still uses the square rect path."""
    space = OrthogonalMooreGrid([3, 3], random=random.Random(42))
    space.create_property_layer("test", default_value=0.0)
    ab = AltairBackend(space_drawer=MagicMock())

    chart = ab.draw_property_layer(
        space,
        space.property_layers,
        lambda l: PropertyLayerStyle(  # noqa: E741
            colormap="viridis", alpha=0.5, vmin=0, vmax=1, colorbar=True
        ),
    )

    layer_spec = chart.to_dict()["layer"][0]
    assert layer_spec["mark"]["type"] == "rect"


def test_altair_backend_collects_agent_data():
    """Test collect_agent_data."""
    ab = AltairBackend(space_drawer=MagicMock())

    class DummyAgent:
        position = (0, 0)
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
        "tooltip": np.array([None, None]),
    }
    ab.space_drawer.get_viz_limits = MagicMock(return_value=(0, 10, 0, 10))
    assert ab.draw_agents(arguments) is not None


def test_altair_backend_draw_property_layer():
    """Test drawing property_layer."""
    ab = AltairBackend(space_drawer=MagicMock())

    space = OrthogonalMooreGrid([2, 2], random=random.Random(42))
    space.create_property_layer("test", default_value=0.0)

    assert (
        ab.draw_property_layer(
            space,
            space.property_layers,
            lambda l: PropertyLayerStyle(  # noqa: E741
                color="red", alpha=0.5, vmin=0, vmax=1, colorbar=False
            ),
        )
        is not None
    )

    assert (
        ab.draw_property_layer(
            space,
            space.property_layers,
            lambda l: PropertyLayerStyle(  # noqa: E741
                colormap="viridis", alpha=0.5, vmin=0, vmax=1, colorbar=True
            ),
        )
        is not None
    )

    with pytest.raises(ValueError, match="Specify one of 'color' or 'colormap'"):
        ab.draw_property_layer(
            space,
            space.property_layers,
            lambda l: PropertyLayerStyle(  # noqa: E741
                color=None, colormap=None, alpha=1.0, vmin=0, vmax=1, colorbar=False
            ),
        )


def test_backend_get_agent_pos():
    """Test extracting agent position from pos and cell.coordinate attributes."""
    mb = MatplotlibBackend(space_drawer=MagicMock())

    class AgentWithPos:
        position = (1, 2)

    x, y = mb._get_agent_pos(AgentWithPos(), None)
    assert (x, y) == (1, 2)

    class AgentWithCell:
        position = (3, 4)
        cell = types.SimpleNamespace(coordinate=(3, 4))

    x, y = mb._get_agent_pos(AgentWithCell(), None)
    assert (x, y) == (3, 4)


def test_backend_get_agent_pos_uses_space_drawer_viz_dims():
    """Backends should project continuous positions using the space drawer's viz_dims."""
    mb = MatplotlibBackend(space_drawer=types.SimpleNamespace(viz_dims=(0, 2)))

    class DummyAgent:
        position = (0.1, 0.2, 0.3)

    x, y = mb._get_agent_pos(DummyAgent(), None)
    assert (x, y) == (0.1, 0.3)


def test_backend_get_agent_pos_raises_when_viz_dims_out_of_range():
    """Backends should raise a helpful error when viz_dims do not match the position."""
    mb = MatplotlibBackend(space_drawer=types.SimpleNamespace(viz_dims=(0, 2)))

    class DummyAgent:
        position = (0.1, 0.2)

    with pytest.raises(ValueError, match="not have enough dimensions"):
        mb._get_agent_pos(DummyAgent(), None)


@pytest.mark.parametrize("backend_cls", [MatplotlibBackend, AltairBackend])
def test_backend_collect_agent_data_projects_3d_continuous_positions(backend_cls):
    """Test collect_agent_data projects nD continuous positions onto viz_dims."""
    backend = backend_cls(space_drawer=MagicMock())
    model = Model(rng=42)
    space = ContinuousSpace(
        dimensions=np.array([[0, 1], [0, 1], [0, 1]]),
        torus=False,
        random=model.random,
    )

    agent_1 = ContinuousSpaceAgent(space, model)
    agent_1.position = [0.1, 0.2, 0.3]
    agent_2 = ContinuousSpaceAgent(space, model)
    agent_2.position = [0.4, 0.5, 0.6]

    def agent_portrayal(agent):
        return AgentPortrayalStyle(
            x=None,
            y=None,
            size=5,
            color="red",
            marker="o",
            zorder=1,
            alpha=1.0,
            edgecolors="black",
            linewidths=1,
        )

    data = backend.collect_agent_data(space, agent_portrayal)

    assert data["loc"].shape == (2, 2)
    np.testing.assert_allclose(data["loc"][:, 0], [0.1, 0.4])
    np.testing.assert_allclose(data["loc"][:, 1], [0.2, 0.5])


@pytest.mark.parametrize("backend_cls", [MatplotlibBackend, AltairBackend])
def test_backend_collect_agent_data_warns_once_for_dict_portrayal(backend_cls):
    """The dict portrayal FutureWarning is emitted once per call, not per agent."""
    backend = backend_cls(space_drawer=MagicMock())

    class DummyAgent:
        position = (0, 0)
        cell = types.SimpleNamespace(coordinate=(0, 0))

    class DummySpace:
        agents: ClassVar[list] = [DummyAgent() for _ in range(5)]

    def agent_portrayal_dict(agent):
        return {"size": 5, "color": "red", "marker": "o"}

    with pytest.warns(FutureWarning) as record:
        data = backend.collect_agent_data(DummySpace(), agent_portrayal_dict)

    assert len([w for w in record if issubclass(w.category, FutureWarning)]) == 1
    assert data["loc"].shape[0] == 5


def test_backends_handle_errors():
    """Test error handling scenarios for invalid agent/property_layer data."""
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


@pytest.mark.parametrize("backend_cls", [MatplotlibBackend, AltairBackend])
def test_collect_agent_data_does_not_mutate_shared_portrayal(backend_cls):
    """Shared AgentPortrayalStyle instances should not be mutated during data collection."""

    class DummyAgent:
        def __init__(self, position):
            self.position = position

    class DummySpace:
        pass

    portrayal = AgentPortrayalStyle(x=None, y=None)
    space = DummySpace()
    space.agents = [DummyAgent((0, 1)), DummyAgent((2, 3))]

    backend = backend_cls(space_drawer=MagicMock())
    data = backend.collect_agent_data(space, lambda agent: portrayal)

    assert data["loc"].tolist() == [[0, 1], [2, 3]]
    assert (portrayal.x, portrayal.y) == (None, None)


@pytest.mark.parametrize("backend_cls", [MatplotlibBackend, AltairBackend])
def test_collect_agent_data_preserves_explicit_coordinates(backend_cls):
    """Explicit x, y coordinates should be used and not mutated."""

    class DummyAgent:
        def __init__(self, position):
            self.position = position

    class DummySpace:
        pass

    portrayal = AgentPortrayalStyle(x=10, y=20)
    space = DummySpace()
    space.agents = [DummyAgent((0, 1)), DummyAgent((2, 3))]

    backend = backend_cls(space_drawer=MagicMock())
    data = backend.collect_agent_data(space, lambda agent: portrayal)

    assert data["loc"].tolist() == [[10, 20], [10, 20]]
    assert (portrayal.x, portrayal.y) == (10, 20)


@pytest.mark.parametrize(
    ("edgecolors", "expected"),
    [
        ([None, None], []),
        (["black", None], ["black", "none"]),
        ([(1.0, 0.0, 0.0, 1.0), None], [(1.0, 0.0, 0.0, 1.0), "none"]),
    ],
)
def test_to_numpy_argument_array_edgecolors(edgecolors, expected):
    """Edgecolors normalize None to "none", or drop entirely if unused by anyone."""
    result = _to_numpy_argument_array("edgecolors", edgecolors)
    assert result.tolist() == expected


def test_to_numpy_argument_array_preserves_tuple_markers():
    """Equal-length tuple marker specs must not collapse into a 2D array."""
    result = _to_numpy_argument_array("marker", [(3, 0, 0), (3, 0, 0)])
    assert result.shape == (2,)
    assert list(result) == [(3, 0, 0), (3, 0, 0)]


def test_to_numpy_argument_array_mixed_face_colors():
    """Face colors can be a mix of named and RGBA colors."""
    mixed = ["blue", (1.0, 0.0, 0.0, 1.0)]
    result = _to_numpy_argument_array("c", mixed)
    assert list(result) == mixed

    uniform = [(1.0, 0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 1.0)]
    result = _to_numpy_argument_array("c", uniform)
    assert result.shape == (2, 4)


@pytest.mark.parametrize(
    "key",
    [
        "x",
        "y",
        "size",
        "shape",
        "opacity",
        "strokeWidth",
        "original_color",
        "is_filled",
        "original_stroke",
        "viz_fill_color",
        "viz_stroke_color",
    ],
)
@pytest.mark.parametrize("colors", [["red", "blue"], [2, 4]])
def test_altair_tooltip_does_not_overwrite_rendering_columns(key, colors):
    """Tooltip fields must not change position, styling, or color inference."""
    backend = AltairBackend(space_drawer=MagicMock())
    backend.space_drawer.get_viz_limits.return_value = (0, 10, 0, 10)
    arguments = {
        "loc": np.array([[1, 2], [3, 4]]),
        "size": np.array([5, 10]),
        "shape": np.array(["circle", "square"]),
        "opacity": np.array([0.5, 1.0]),
        "strokeWidth": np.array([1, 2]),
        "color": np.array(colors),
        "filled": np.array([True, False]),
        "stroke": np.array(["black", "green"]),
        "tooltip": np.array([None, None]),
    }
    control = backend.draw_agents(arguments)
    arguments["tooltip"] = np.array([{key: 99}, {key: None}], dtype=object)
    chart = backend.draw_agents(arguments)
    for column in control.data:
        assert chart.data[column].equals(control.data[column])
    encoding = chart.to_dict()["encoding"]
    control_encoding = control.to_dict()["encoding"]
    assert {k: v for k, v in encoding.items() if k != "tooltip"} == {
        k: v for k, v in control_encoding.items() if k != "tooltip"
    }
    tooltip = encoding["tooltip"][0]
    assert tooltip["title"] == key
    assert tooltip["field"] not in control.data
    assert chart.data[tooltip["field"]].iloc[0] == 99
    assert np.isnan(chart.data[tooltip["field"]].iloc[1])


def test_altair_tooltip_heterogeneous_keys_and_labels():
    """Preserve labels and missing values without interpreting field syntax."""
    backend = AltairBackend(space_drawer=MagicMock())
    backend.space_drawer.get_viz_limits.return_value = (0, 10, 0, 10)
    arguments = {
        "loc": np.array([[1, 2], [3, 4], [5, 6]]),
        "size": np.array([5, 5, 5]),
        "shape": np.array(["circle"] * 3),
        "opacity": np.ones(3),
        "strokeWidth": np.ones(3),
        "color": np.array(["red"] * 3),
        "filled": np.ones(3, dtype=bool),
        "stroke": np.array(["black"] * 3),
        "tooltip": np.array(
            [
                {"x": 99, "rate:Q": 1, "a.b": "first"},
                {"_mesa_tooltip_0": "user", "empty": None, "a[b]": "second"},
                None,
            ],
            dtype=object,
        ),
    }
    chart = backend.draw_agents(arguments)
    tooltips = chart.to_dict()["encoding"]["tooltip"]
    assert [t["title"] for t in tooltips] == sorted(
        ["x", "rate:Q", "a.b", "_mesa_tooltip_0", "empty", "a[b]"]
    )
    fields = {t["title"]: f"tooltip_{t['title']}" for t in tooltips}
    assert len(set(fields.values())) == len(fields)
    assert chart.data[fields["x"]].iloc[0] == 99
    assert chart.data[fields["_mesa_tooltip_0"]].iloc[1] == "user"
    assert chart.data[fields["a.b"]].iloc[0] == "first"
    assert chart.data[fields["a[b]"]].iloc[1] == "second"
    assert chart.data[fields["empty"]].isna().all()
    assert chart.data[list(fields.values())].iloc[2].isna().all()
    assert chart.data["x"].tolist() == [1, 3, 5]


def _tooltip_chart(tooltips):
    backend = AltairBackend(space_drawer=MagicMock())
    backend.space_drawer.get_viz_limits.return_value = (0, 10, 0, 10)
    count = len(tooltips)
    return backend.draw_agents(
        {
            "loc": np.ones((count, 2)),
            "size": np.ones(count),
            "shape": np.array(["circle"] * count),
            "opacity": np.ones(count),
            "strokeWidth": np.ones(count),
            "color": np.array(["red"] * count),
            "filled": np.ones(count, dtype=bool),
            "stroke": np.array(["black"] * count),
            "tooltip": np.array(tooltips, dtype=object),
        }
    )


def test_altair_tooltip_names_are_stable_for_post_processing():
    """Adding another tooltip does not rename existing data columns."""
    first = _tooltip_chart([{"wealth": 3}])
    second = _tooltip_chart([{"age": 4, "wealth": 3}])
    assert first.data["tooltip_wealth"].equals(second.data["tooltip_wealth"])
    from mesa.visualization import SpaceRenderer  # noqa: PLC0415

    renderer = SpaceRenderer.__new__(SpaceRenderer)
    renderer.post_process = lambda chart: chart.transform_filter(
        "datum.tooltip_wealth > 2"
    )
    assert renderer.post_process(second).to_dict()["transform"] == [
        {"filter": "datum.tooltip_wealth > 2"}
    ]


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1, 2], "quantitative"),
        (["a", "b"], "nominal"),
        ([pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")], "temporal"),
        ([None, None], "nominal"),
    ],
)
def test_altair_tooltip_types(values, expected):
    """Tooltip specifications retain the inferred measurement type."""
    chart = _tooltip_chart([{"value": value} for value in values])
    assert chart.to_dict()["encoding"]["tooltip"][0]["type"] == expected


@pytest.mark.parametrize("key", [7, None, (1, 2)])
def test_altair_tooltip_titles_are_strings(key):
    """Non-string dictionary keys become valid tooltip titles."""
    chart = _tooltip_chart([{key: 3, "other": 4}])
    tooltip = next(
        t for t in chart.to_dict()["encoding"]["tooltip"] if t["title"] == str(key)
    )
    assert tooltip["field"] == f"tooltip_{key}"


def test_altair_tooltip_duplicate_string_labels():
    """Ambiguous labels cannot silently overwrite another tooltip column."""
    with pytest.raises(ValueError, match="distinct string labels"):
        _tooltip_chart([{1: 2, "1": 3}])


@pytest.mark.parametrize("key", ["a.b", "a[b]", "a\\b", "rate:Q"])
def test_altair_tooltip_special_fields_are_literal(key):
    """Encoding field paths escape punctuation while dataframe names stay literal."""
    chart = _tooltip_chart([{key: 3}])
    tooltip = chart.to_dict()["encoding"]["tooltip"][0]
    field = f"tooltip_{key}"
    escaped = field.replace("\\", "\\\\")
    for character in ".[]":
        escaped = escaped.replace(character, f"\\{character}")
    assert tooltip["field"] == escaped
    assert chart.data[field].iloc[0] == 3
