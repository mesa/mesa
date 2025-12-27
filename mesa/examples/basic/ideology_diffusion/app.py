import solara
from model import IdeologyModel

from mesa.experimental.solara_viz import (
    SolaraViz,
    make_plot_component,
    make_space_component,
)


def agent_portrayal(agent):
    color = {
        "neutral": "gray",
        "moderate": "orange",
        "radical": "red",
    }[agent.opinion]

    return {
        "color": color,
        "size": 40,
    }


model_params = {
    "N": solara.SliderInt(10, 300, value=120, label="Population"),
    "economic_crisis": solara.SliderFloat(0.0, 1.0, value=0.5, label="Economic Crisis"),
    "propaganda": solara.SliderFloat(0.0, 1.0, value=0.2, label="Propaganda"),
}

space = make_space_component(agent_portrayal)
plot = make_plot_component(
    {
        "Neutrals": "gray",
        "Moderates": "orange",
        "Radicals": "red",
    }
)


@solara.component
def App():
    SolaraViz(
        IdeologyModel,
        components=[space, plot],
        model_params=model_params,
        name="Ideology Diffusion Model",
    )
