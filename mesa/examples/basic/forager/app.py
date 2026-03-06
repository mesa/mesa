"""Interactive Solara visualization for the Forager model."""

from mesa.examples.basic.forager.model import ForagerModel
from mesa.visualization import Slider, SolaraViz, make_plot_component

model_params = {
    "n_agents": Slider("Number of agents", 20, 5, 100, 5),
    "threat_prob": Slider("Threat probability", 0.1, 0.0, 0.5, 0.05),
    "forage_duration": Slider("Forage duration", 5.0, 1.0, 10.0, 1.0),
    "flee_duration": Slider("Flee duration", 1.0, 0.5, 3.0, 0.5),
    "food_per_forage": Slider("Food per forage", 1.0, 0.5, 5.0, 0.5),
}

FoodPlot = make_plot_component({"total_food": "tab:green"})
ActivityPlot = make_plot_component(
    {
        "foraging_agents": "tab:blue",
        "fleeing_agents": "tab:red",
        "idle_agents": "tab:gray",
    }
)

model = ForagerModel()

page = SolaraViz(
    model,
    components=[FoodPlot, ActivityPlot],
    model_params=model_params,
    name="Forager Model",
)
page  # noqa
