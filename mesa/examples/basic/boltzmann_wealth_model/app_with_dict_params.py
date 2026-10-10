import importlib
import sys

import altair as alt

import mesa.visualization.solara_viz as solara_viz

# Import custom parameter support
from mesa.examples.basic.boltzmann_wealth_model.custom_params import CustomUserInputs
from mesa.examples.basic.boltzmann_wealth_model.model import BoltzmannWealth
from mesa.mesa_logging import INFO, log_to_stderr
from mesa.visualization import (
    SolaraViz,
    SpaceRenderer,
    make_plot_component,
)
from mesa.visualization.components import AgentPortrayalStyle

# Override UserInputs with CustomUserInputs
solara_viz.UserInputs = CustomUserInputs

log_to_stderr(INFO)


def agent_portrayal(agent):
    return AgentPortrayalStyle(color=agent.wealth)


# Modified model_params to use Dict type for grid dimensions
model_params = {
    "rng": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "n": {
        "type": "SliderInt",
        "value": 50,
        "label": "Number of agents:",
        "min": 10,
        "max": 100,
        "step": 1,
    },
    "grid_dimensions": {
        "type": "Dict",
        "label": "Grid Dimensions",
        "entries": {
            "width": {
                "value": 10,
                "label": "Width",
                "type": "SliderInt",
                "min": 5,
                "max": 50,
                "step": 1,
            },
            "height": {
                "value": 10,
                "label": "Height",
                "type": "SliderInt",
                "min": 5,
                "max": 50,
                "step": 1,
            },
        },
    },
}


# Wrapper model to accept grid_dimensions dict
class BoltzmannWealthWithDictParams(BoltzmannWealth):
    def __init__(self, n=100, grid_dimensions=None, rng=None):
        if grid_dimensions is None:
            grid_dimensions = {"width": 10, "height": 10}
        width = grid_dimensions["width"]
        height = grid_dimensions["height"]
        super().__init__(n=n, width=width, height=height, rng=rng)


def post_process(chart):
    """Post-process the Altair chart to add a colorbar legend."""
    chart = chart.encode(
        color=alt.Color(
            "color:N",
            scale=alt.Scale(scheme="viridis", domain=[0, 10]),
            legend=alt.Legend(
                title="Wealth",
                orient="right",
                type="gradient",
                gradientLength=200,
            ),
        ),
    )
    return chart


model = BoltzmannWealthWithDictParams(
    n=50, grid_dimensions={"width": 10, "height": 10}, rng=42
)

renderer = (
    SpaceRenderer(model, backend="altair")
    .setup_structure(grid_color="black", grid_dash=[6, 2], grid_opacity=0.3)
    .setup_agents(agent_portrayal, cmap="viridis", vmin=0, vmax=10)
)
renderer.render()

renderer.post_process = post_process

GiniPlot = make_plot_component("Gini")

page = SolaraViz(
    model,
    renderer,
    components=[GiniPlot],
    model_params=model_params,
    name="Boltzmann Wealth Model (Custom Dict Parameters)",
)

page  # noqa

APP = "mesa.examples.basic.boltzmann_wealth_model.app_with_dict_params"


def test_app_with_dict_params(monkeypatch):
    # The app overrides solara_viz.UserInputs at import time. Registering the
    # current value with monkeypatch means it is restored after this test.
    monkeypatch.setattr(solara_viz, "UserInputs", solara_viz.UserInputs)
    sys.modules.pop(APP, None)
    app = importlib.import_module(APP)

    assert app.page is not None
    assert solara_viz.UserInputs is custom_params.CustomUserInputs

    # Default grid_dimensions branch
    default_model = app.BoltzmannWealthWithDictParams()
    assert len(default_model.agents) == 100

    # Explicit grid_dimensions branch
    model = app.BoltzmannWealthWithDictParams(
        n=5, grid_dimensions={"width": 6, "height": 7}, rng=1
    )
    assert len(model.agents) == 5

    # agent_portrayal and post_process
    agent = next(iter(model.agents))
    assert app.agent_portrayal(agent) is not None
    assert app.post_process(alt.Chart().mark_point()) is not None
