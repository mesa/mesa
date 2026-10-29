from mesa.examples.advanced.birds_and_trees.agents import Bird, Tree
from mesa.examples.advanced.birds_and_trees.model import (
    BirdsAndTrees,
    BirdsAndTreesScenario,
)
from mesa.visualization import Slider, SolaraViz, SpaceRenderer, make_plot_component
from mesa.visualization.components import AgentPortrayalStyle


def portrayal(agent):
    if isinstance(agent, Bird):
        return AgentPortrayalStyle(color="tab:blue", marker="^", size=30, zorder=2)
    if isinstance(agent, Tree):
        color = "tab:red" if agent.has_fruit else "tab:green"
        return AgentPortrayalStyle(color=color, marker="s", size=60, zorder=1)


def post_process_space(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


model_params = {
    "rng": {"type": "InputText", "value": 42, "label": "Random Seed"},
    "tree_density": Slider("Tree density", 0.2, 0.01, 0.5, 0.01),
    "initial_birds": Slider("Initial birds", 50, 1, 200),
    "speed": Slider("Bird speed", 3.0, 0.5, 10.0, 0.5),
    "vision": Slider("Vision (ground cells)", 1, 0, 5),
    "fruit_regrowth": Slider("Fruit regrowth chance", 0.05, 0.01, 0.5, 0.01),
}

model = BirdsAndTrees(scenario=BirdsAndTreesScenario())
renderer = SpaceRenderer(model, backend="matplotlib").setup_agents(portrayal)
renderer.render()
renderer.post_process = post_process_space

lineplot = make_plot_component({"Birds": "tab:blue", "Fruit": "tab:red"})

page = SolaraViz(
    model,
    renderer,
    components=[lineplot],
    model_params=model_params,
    name="Birds and Trees",
)
page  # noqa
