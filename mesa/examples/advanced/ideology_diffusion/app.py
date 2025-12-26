import solara
from mesa.visualization import SolaraViz, make_plot_component
from model import IdeologyDiffusionModel

# Define interactive parameters for the Solara dashboard
model_params = {
    "economic_crisis": solara.reactive(True),
    "media_influence": solara.reactive(True),
    "government_repression": solara.reactive(False),
    "unemployment_increase": solara.reactive(0.25),
    "width": 20,
    "height": 20,
}

def agent_portrayal(agent):
    # Hex colors for Neutral (Blue), Moderate (Orange), and Radical (Red)
    colors = {0: "#3498db", 1: "#e67e22", 2: "#e74c3c"}
    return {
        "color": colors.get(agent.political_ideology, "gray"),
        "size": 10,
    }

page = SolaraViz(
    IdeologyDiffusionModel,
    model_params,
    prop_names=[
        "economic_crisis", 
        "media_influence", 
        "government_repression", 
        "unemployment_increase"
    ],
    agent_portrayal=agent_portrayal,
    name="Ideological Diffusion Model",
    components=[
        make_plot_component({
            "Neutral": "#3498db", 
            "Moderate": "#e67e22", 
            "Radical": "#e74c3c"
        })
    ]
)