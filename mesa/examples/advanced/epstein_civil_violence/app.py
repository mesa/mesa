from mesa.examples.advanced.epstein_civil_violence.agents import (
    Citizen,
    CitizenState,
    Cop,
)
from mesa.examples.advanced.epstein_civil_violence.model import EpsteinCivilViolence
from mesa.visualization import (
    Slider,
    SolaraViz,
    SpaceRenderer,
    make_plot_component,
)
from mesa.visualization.components import AgentPortrayalStyle

COP_COLOR = "#000000"

agent_colors = {
    CitizenState.ACTIVE: "#FE6100",
    CitizenState.QUIET: "#648FFF",
    CitizenState.ARRESTED: "#808080",
}


def citizen_cop_portrayal(agent):
    if agent is None:
        return

    portrayal = AgentPortrayalStyle(size=200)

    if isinstance(agent, Citizen):
        portrayal.update(("color", agent_colors[agent.state]))
    elif isinstance(agent, Cop):
        portrayal.update(("color", COP_COLOR))

    return portrayal


def post_process(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.get_figure().set_size_inches(10, 10)


model_params = {
    "rng": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "height": 40,
    "width": 40,
    "citizen_density": Slider("Initial Agent Density", 0.7, 0.1, 0.9, 0.1),
    "cop_density": Slider("Initial Cop Density", 0.04, 0.0, 0.1, 0.01),
    "citizen_vision": Slider("Citizen Vision", 7, 1, 10, 1),
    "cop_vision": Slider("Cop Vision", 7, 1, 10, 1),
    "legitimacy": Slider("Government Legitimacy", 0.82, 0.0, 1, 0.01),
    "max_jail_term": Slider("Max Jail Term", 30, 0, 50, 1),
    "activation_order": {
        "type": "Select",
        "value": "Random",
        "values": ["Random", "Sequential"],
        "label": "Activation Order",
    },
    "grid_type": {
        "type": "Select",
        "value": "Von Neumann",
        "values": ["Von Neumann", "Moore"],
        "label": "Grid Type",
    },
}


chart_component = make_plot_component(
    {state.name.lower(): agent_colors[state] for state in CitizenState}
)

epstein_model = EpsteinCivilViolence()
renderer = SpaceRenderer(epstein_model, backend="matplotlib").setup_agents(
    citizen_cop_portrayal
)
# Specifically, avoid drawing the grid to hide the grid lines.
renderer.draw_agents()
renderer.post_process = post_process

page = SolaraViz(
    epstein_model,
    renderer,
    components=[chart_component],
    model_params=model_params,
    name="Epstein Civil Violence",
)
page  # noqa
