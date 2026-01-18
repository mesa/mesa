from mesa.examples.advanced.sugarscape_g1mt.model import SugarscapeG1mt
from mesa.visualization import Slider, SolaraViz, SpaceRenderer, make_plot_component
from mesa.visualization.components import AgentPortrayalStyle, PropertyLayerStyle


def agent_portrayal(agent):
    return AgentPortrayalStyle(
        x=agent.cell.coordinate[0],
        y=agent.cell.coordinate[1],
        color="red",
        marker="o",
        size=10,
        zorder=1,
    )


def propertylayer_portrayal(layer):
    if layer.name == "sugar":
        return PropertyLayerStyle(
            color="blue", alpha=0.8, colorbar=True, vmin=0, vmax=10
        )
    return PropertyLayerStyle(color="red", alpha=0.8, colorbar=True, vmin=0, vmax=10)


def post_process(chart):
    chart = chart.properties(width=400, height=400)
    return chart


model_params = {
    "rng": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "width": 50,
    "height": 50,
    # Population parameters
    "initial_population": Slider(
        "Initial Population", value=200, min=50, max=500, step=10
    ),
    # Agent endowment parameters
    "endowment_min": Slider("Min Initial Endowment", value=25, min=5, max=30, step=1),
    "endowment_max": Slider("Max Initial Endowment", value=50, min=30, max=100, step=1),
    # Metabolism parameters
    "metabolism_min": Slider("Min Metabolism", value=1, min=1, max=3, step=1),
    "metabolism_max": Slider("Max Metabolism", value=5, min=3, max=8, step=1),
    # Vision parameters
    "vision_min": Slider("Min Vision", value=1, min=1, max=3, step=1),
    "vision_max": Slider("Max Vision", value=5, min=3, max=8, step=1),
    # Trade parameter
    "enable_trade": {"type": "Checkbox", "value": True, "label": "Enable Trading"},
}

model = SugarscapeG1mt()

# Here, the renderer uses the Altair backend, while the plot components
# use the Matplotlib backend.
# Both can be mixed and matched to enhance the visuals of your model.
renderer = (
    SpaceRenderer(model, backend="altair")
    .setup_agents(agent_portrayal)
    .setup_propertylayer(propertylayer_portrayal)
)
# Specifically, avoid drawing the grid to hide the grid lines.
renderer.draw_agents()
renderer.draw_propertylayer()

renderer.post_process = post_process

# Note: It is advised to switch the pages after pausing the model
# on the Solara dashboard.
page = SolaraViz(
    model,
    renderer,
    components=[
        make_plot_component("#Traders", page=1),
        make_plot_component("Price", page=1),
    ],
    model_params=model_params,
    name="Sugarscape {G1, M, T}",
    play_interval=150,
)
page  # noqa
