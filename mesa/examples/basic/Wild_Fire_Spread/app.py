from examples.basic.Wild_Fire_Spread.agent import AgentState, FuelAgent
from examples.basic.Wild_Fire_Spread.model import ForestFireModel

from mesa.visualization import (
    SolaraViz,
    make_plot_component,
    make_space_component,
)

# 🎨 Colors
fuel_colors = {
    AgentState.HEALTHY: "#00FF00",  # green
    AgentState.BURNING: "#FFD700",  # yellow
    AgentState.BURNED: "#FF0000",  # red
}


# 🎯 Portrayal
def fuel_agent_portrayal(agent):
    if agent is None:
        return

    portrayal = {"size": 80}

    if isinstance(agent, FuelAgent):
        portrayal["color"] = fuel_colors[agent.state]

    return portrayal


# 🧼 Clean plot
def post_process(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.get_figure().set_size_inches(8, 8)


# ⚙️ Model params
model_params = {
    "width": 60,
    "height": 60,
}


# 📊 Chart
chart_component = make_plot_component(
    {
        "Healthy": "green",
        "Burning": "orange",
        "Burned": "red",
    }
)


# 🚀 Create model
model = ForestFireModel(
    width=model_params["width"],
    height=model_params["height"],
)


# 🎥 Renderer
space_component = make_space_component(
    fuel_agent_portrayal, post_process=post_process, draw_grid=False
)


# 🌐 App
page = SolaraViz(
    model,
    components=[space_component, chart_component],
    model_params=model_params,
    name="🔥 Forest Fire Model",
)
