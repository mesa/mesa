import altair as alt

from mesa.examples.basic.boltzmann_wealth_model.model import BoltzmannWealth
from mesa.mesa_logging import INFO, log_to_stderr
from mesa.visualization import (
    SolaraViz,
    SpaceRenderer,
    make_plot_component,
)
from mesa.visualization.components import AgentPortrayalStyle

log_to_stderr(INFO)


def agent_portrayal(agent):
    return AgentPortrayalStyle(
        color=agent.wealth
    )  # we are using a colormap to translate wealth to color


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
    "width": 10,
    "height": 10,
}


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


model = BoltzmannWealth(50, 10, 10)

# The SpaceRenderer is responsible for drawing the model's space and agents.
# It builds the visualization in layers, first drawing the grid structure,
# and then drawing the agents on top. It uses a specified backend
# (like "altair" or "matplotlib") for creating the plots.

renderer = (
    SpaceRenderer(model, backend="altair")
    .setup_structure(  # To customize the grid appearance.
        grid_color="black", grid_dash=[6, 2], grid_opacity=0.3
    )
    .setup_agents(agent_portrayal, cmap="viridis", vmin=0, vmax=10)
)
renderer.render()

# The post_process function is used to modify the Altair chart after it has been created.
# It can be used to add legends, colorbars, or other visual elements.
renderer.post_process = post_process

# Creates a line plot component from the model's "Gini" datacollector.
GiniPlot = make_plot_component("Gini")

# The SolaraViz page combines the model, renderer, and components into a web interface.
# To run the visualization, save this code as app.py and run `solara run app.py`
page = SolaraViz(
    model,
    renderer,
    components=[GiniPlot],
    model_params=model_params,
    name="Boltzmann Wealth Model",
)
page  # noqa
