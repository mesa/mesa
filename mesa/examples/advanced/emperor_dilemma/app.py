from model import EmperorModel

from mesa.experimental.devs.simulator import DEVSimulator
from mesa.visualization import (
    Slider,
    SolaraViz,
    SpaceRenderer,
    make_plot_component,
)
from mesa.visualization.components.portrayal_components import AgentPortrayalStyle

# Colors matching Figure 2
COLOR_COMPLY_QUIET = "#F0F8FF"  # AliceBlue
COLOR_DEVIATE_QUIET = "lightgray"  # Light Gray
COLOR_COMPLY_ENFORCE = "dimgray"  # Dark Gray
COLOR_DEVIATE_ENFORCE = "black"  # Black


def emperor_portrayal(agent):
    if agent is None:
        return
    portrayal = AgentPortrayalStyle(
        size=170,
        marker="s",
        zorder=1,
    )

    if agent.compliance == 1:
        if agent.enforcement == 1:
            portrayal.update(("color", COLOR_COMPLY_ENFORCE))
        else:
            portrayal.update(("color", COLOR_COMPLY_QUIET))

    else:
        if agent.enforcement == -1:
            portrayal.update(("color", COLOR_DEVIATE_ENFORCE))
        else:
            portrayal.update(("color", COLOR_DEVIATE_QUIET))

    return portrayal


def post_process_lines(ax):
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.9))
    ax.set_ylabel("Rate")


lineplot_component = make_plot_component(
    {
        "Compliance": "tab:green",
        "Enforcement": "tab:red",
        "False Enforcement": "tab:blue",
    },
    post_process=post_process_lines,
)


def post_process_space(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.5, 24.5)
    ax.set_ylim(-0.5, 24.5)
    ax.get_figure().set_size_inches(8, 8)


model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "fraction_true_believers": Slider("Fraction True Believers", 0.05, 0.0, 1.0, 0.01),
    "k": Slider("Enforcement Cost (K)", 0.125, 0.0, 0.5, 0.025),
    "homophily": {
        "type": "Select",
        "value": False,
        "values": [True, False],
        "label": "Cluster Believers (Homophily)?",
    },
    "width": 25,
    "height": 25,
}

simulator = DEVSimulator()

simulator = DEVSimulator()
model = EmperorModel(
    simulator=simulator, fraction_true_believers=0.05, k=0.125, width=40, height=25
)


renderer = SpaceRenderer(
    model,
    backend="matplotlib",
).setup_agents(emperor_portrayal)
renderer.post_process = post_process_space
renderer.draw_agents()


page = SolaraViz(
    model,
    renderer,
    components=[lineplot_component],
    model_params=model_params,
    name="The Emperor's Dilemma",
    simulator=simulator,
)

page #noqa
