from mesa.experimental.devs.simulator import ABMSimulator
from mesa.visualization import (
    CommandConsole,
    Slider,
    SolaraViz,
    SpaceRenderer,
    make_plot_component,
)
from model import EmperorModel


COLOR_COMPLY_ENFORCE = "#0C0707" 
COLOR_COMPLY_QUIET = "aqua"
COLOR_DEVIATE_ENFORCE = "black"   
COLOR_DEVIATE_QUIET = "lightgray"

def emperor_portrayal(agent):
    if agent is None:
        return

    # Use a simple dictionary for the portrayal to ensure Solara renders it correctly
    portrayal = {
        "size": 180,
        "marker": "s",
        "zorder": 1,
    }

    """ Logic: Map (Compliance, Enforcement) state to color
    Compliance: 1 (True), -1 (False)
    Enforcement: 1 (Enforce Norm), -1 (Enforce Deviance), 0 (None)
    """
    
    if agent.compliance == 1:
        if agent.enforcement == 1:
            portrayal["color"] = COLOR_COMPLY_ENFORCE
        else:
            portrayal["color"] = COLOR_COMPLY_QUIET
    else:
        # Compliance is -1 (Deviate)
        if agent.enforcement == -1:
            portrayal["color"] = COLOR_DEVIATE_ENFORCE
        else:
            portrayal["color"] = COLOR_DEVIATE_QUIET

    return portrayal

def post_process_lines(ax):
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.9))
    ax.set_ylabel("Rate")

lineplot_component = make_plot_component(
    {
        "Compliance": "tab:green",
        "Enforcement": "tab:red",
        "False Enforcement": "tab:blue"
    },
    post_process=post_process_lines,
)

def post_process_space(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.get_figure().set_size_inches(10, 10) 


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

# 1. Create the Simulator
simulator = ABMSimulator()

# 2. Create the Model
model = EmperorModel(
    simulator=simulator,
    fraction_true_believers=0.05, 
    k=0.125,
    width=25,
    height=25
)


# 3. Configure Renderer
renderer = SpaceRenderer(
    model,
    backend="matplotlib",
).setup_agents(emperor_portrayal)
renderer.draw_agents()
renderer.post_process = post_process_space

# 4. Create Visualization Page
page = SolaraViz(
    model,
    renderer,
    components=[lineplot_component, CommandConsole],
    model_params=model_params,
    name="The Emperor's Dilemma",
    simulator=simulator,
)
page # noqa