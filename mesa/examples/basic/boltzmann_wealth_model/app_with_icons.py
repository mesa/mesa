"""Example demonstrating icon-based agent visualization in Mesa.
This example shows how to use bundled icons (smiley, sad_face, neutral_face)
to visualize agents. Bundled icons work with no extra dependencies.
To run this example:
    solara run app_with_icons.py
"""

import altair as alt

from mesa.examples.basic.boltzmann_wealth_model.model import BoltzmannWealth
from mesa.visualization import (
    SolaraViz,
    SpaceRenderer,
    make_plot_component,
)


def agent_portrayal(agent):
    """Portray agent with icon based on wealth level.
    Demonstrates icon usage: agents with wealth > 5 get a smiley icon,
    others get a neutral or sad face. Icons work with no extra dependencies.
    Args:
        agent: The agent to portray
    Returns:
        dict: Portrayal dictionary with icon and styling information
    """
    # Base style with color mapping for wealth
    portrayal = {
        "color": agent.wealth,
    }

    # Add icon based on wealth (bundled icons work with no extra dependencies)
    if agent.wealth > 5:
        portrayal["icon"] = "smiley"
        portrayal["icon_size"] = 24
    elif agent.wealth > 2:
        portrayal["icon"] = "neutral_face"
        portrayal["icon_size"] = 24
    else:
        portrayal["icon"] = "sad_face"
        portrayal["icon_size"] = 24

    if agent.wealth > 5:
        portrayal["icon"] = "smiley"
        portrayal["icon_size"] = 24
    elif agent.wealth > 2:
        portrayal["icon"] = "neutral_face"
        portrayal["icon_size"] = 24
    else:
        portrayal["icon"] = "sad_face"
        portrayal["icon_size"] = 24

    # Debug print
    print(f"Portrayal for agent {agent.unique_id}: {portrayal}")

    return portrayal


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


model_params = {
    "seed": {
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
# Create model instance
model = BoltzmannWealth(50, 10, 10)
# Create renderer with icon support enabled
renderer = SpaceRenderer(model, backend="altair", icon_mode="force")
# Customize the grid appearance
renderer.draw_structure(grid_color="black", grid_dash=[6, 2], grid_opacity=0.3)
# Draw agents with icon portrayal
renderer.draw_agents(agent_portrayal=agent_portrayal, cmap="viridis", vmin=0, vmax=10)
# Apply post-processing for colorbar
renderer.post_process = post_process
# Create a line plot component from the model's "Gini" datacollector
GiniPlot = make_plot_component("Gini")
# Create the SolaraViz page
page = SolaraViz(
    model,
    renderer,
    components=[GiniPlot],
    model_params=model_params,
    name="Boltzmann Wealth Model with Icons",
)
