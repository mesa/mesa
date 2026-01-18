import math

import solara

from mesa.examples.basic.virus_on_network.model import (
    State,
    VirusOnNetwork,
    number_infected,
)
from mesa.visualization import (
    Slider,
    SolaraViz,
    SpaceRenderer,
    make_plot_component,
)
from mesa.visualization.components import AgentPortrayalStyle


def agent_portrayal(agent):
    node_color_dict = {
        State.INFECTED: "red",
        State.SUSCEPTIBLE: "green",
        State.RESISTANT: "gray",
    }
    return AgentPortrayalStyle(color=node_color_dict[agent.state], size=20)


def get_resistant_susceptible_ratio(model):
    ratio = model.resistant_susceptible_ratio()
    ratio_text = r"$\infty$" if ratio is math.inf else f"{ratio:.2f}"
    infected_text = str(number_infected(model))

    return solara.Markdown(
        f"Resistant/Susceptible Ratio: {ratio_text}<br>Infected Remaining: {infected_text}"
    )


model_params = {
    "rng": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "num_nodes": Slider(
        label="Number of agents",
        value=10,
        min=10,
        max=100,
        step=1,
    ),
    "avg_node_degree": Slider(
        label="Avg Node Degree",
        value=3,
        min=3,
        max=8,
        step=1,
    ),
    "initial_outbreak_size": Slider(
        label="Initial Outbreak Size",
        value=1,
        min=1,
        max=10,
        step=1,
    ),
    "virus_spread_chance": Slider(
        label="Virus Spread Chance",
        value=0.4,
        min=0.0,
        max=1.0,
        step=0.1,
    ),
    "virus_check_frequency": Slider(
        label="Virus Check Frequency",
        value=0.4,
        min=0.0,
        max=1.0,
        step=0.1,
    ),
    "recovery_chance": Slider(
        label="Recovery Chance",
        value=0.3,
        min=0.0,
        max=1.0,
        step=0.1,
    ),
    "gain_resistance_chance": Slider(
        label="Gain Resistance Chance",
        value=0.5,
        min=0.0,
        max=1.0,
        step=0.1,
    ),
}


def post_process_lineplot(chart):
    chart = chart.properties(
        width=400,
        height=400,
    ).configure_legend(
        strokeColor="black",
        fillColor="#ECE9E9",
        orient="right",
        cornerRadius=5,
        padding=10,
        strokeWidth=1,
    )
    return chart


model1 = VirusOnNetwork()
renderer = (
    SpaceRenderer(model1, backend="altair")
    .setup_structure(  # Do this to draw the underlying network and customize it
        node_kwargs={"color": "black", "filled": False, "strokeWidth": 5},
        edge_kwargs={"strokeDash": [6, 1]},
    )
    .setup_agents(agent_portrayal)
)

renderer.render()

# Plot components can also be in altair and support post_process
StatePlot = make_plot_component(
    {"Infected": "red", "Susceptible": "green", "Resistant": "gray"},
    backend="altair",
    post_process=post_process_lineplot,
)

page = SolaraViz(
    model1,
    renderer,
    components=[
        StatePlot,
        get_resistant_susceptible_ratio,
    ],
    model_params=model_params,
    name="Virus Model",
)
page  # noqa
