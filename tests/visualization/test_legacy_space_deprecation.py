"""The legacy space drawing API warns once per call and is replaced by SpaceRenderer (#4053)."""

import warnings

import pytest
import solara

import mesa
from mesa.discrete_space import CellAgent, OrthogonalMooreGrid
from mesa.visualization import SolaraViz, draw_space, make_space_component
from mesa.visualization.components import (
    AgentPortrayalStyle,
    make_altair_space,
    make_mpl_space_component,
)
from mesa.visualization.components.altair_components import make_space_altair
from mesa.visualization.components.matplotlib_components import (
    SpaceMatplotlib,
    make_space_matplotlib,
)


class GridModel(mesa.Model):
    """Small grid model with one agent."""

    def __init__(self):  # noqa: D107
        super().__init__(rng=42)
        self.grid = OrthogonalMooreGrid((4, 4), random=self.random)
        agent = CellAgent(self)
        agent.cell = self.grid[1, 1]


def portrayal(_):
    """Portray every agent with the default style."""
    return AgentPortrayalStyle()


@pytest.mark.parametrize(
    ("name", "call"),
    [
        ("draw_space", lambda: draw_space(GridModel().grid, portrayal)),
        ("make_space_component", lambda: make_space_component(portrayal)),
        (
            "make_space_component",
            lambda: make_space_component(portrayal, backend="altair"),
        ),
        ("make_mpl_space_component", lambda: make_mpl_space_component(portrayal)),
        ("make_altair_space", lambda: make_altair_space(portrayal)),
        ("make_space_matplotlib", lambda: make_space_matplotlib(portrayal)),
        ("make_space_altair", lambda: make_space_altair(portrayal)),
    ],
)
def test_legacy_space_api_warns_once(name, call):
    """Each legacy entry point warns exactly once and points to SpaceRenderer."""
    with pytest.warns(FutureWarning) as record:
        call()

    assert len(record) == 1
    message = str(record[0].message)
    assert message.startswith(f"{name}() is deprecated")
    assert "SpaceRenderer" in message
    assert record[0].filename == __file__


@pytest.mark.parametrize(
    "page",
    [
        lambda: SolaraViz(GridModel()),
        lambda: SpaceMatplotlib(GridModel(), portrayal, None),
    ],
)
def test_internal_paths_do_not_warn(page):
    """SolaraViz's default space and SpaceMatplotlib use the legacy code without warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        solara.render(page(), handle_error=False)
