from mesa.visualization import SolaraViz, make_plot_component, SpaceRenderer
from model import HumanitarianModel
from agents import Beneficiary, Truck
from mesa.visualization.components import AgentPortrayalStyle


def agent_portrayal(agent):
    """
    Defines how agents look in the browser visualization.

    Color Coding for Beneficiaries (Needs-Based status):
    - RED -> Desperate
    - ORANGE -> Seeking
    - LIME -> Opportunistic
    - GREEN -> Wandering

    Trucks:
    - BLUE Squares: Delivery agents.

    Args:
        agent: The agent to portray.

    Returns:
        AgentPortrayalStyle: A class instance with style properties.
    """
    if agent is None:
        return

    portrayal_color = "grey"
    portrayal_size = 50
    portrayal_marker = "o"
    portrayal_zorder = 0

    if isinstance(agent, Beneficiary):
        # Color coding based on behavioral regime
        if agent.state == "desperate":
            portrayal_color = "red"
            portrayal_size = 80
        elif agent.state == "seeking":
            portrayal_color = "orange"
        elif agent.state == "opportunistic":
            portrayal_color = "lime"
        else:
            # Wandering / Comfort
            portrayal_color = "green"

    elif isinstance(agent, Truck):
        portrayal_color = "blue"
        portrayal_marker = "s"  # Square
        portrayal_size = 70
        portrayal_size = 70
        portrayal_zorder = (
            1  # Draw on top of beneficiaries so trucks are always visible
        )

    return AgentPortrayalStyle(
        color=portrayal_color,
        marker=portrayal_marker,
        size=portrayal_size,
        zorder=portrayal_zorder,
    )


model_params = {
    "num_beneficiaries": {
        "type": "SliderInt",
        "value": 30,
        "label": "Number of Beneficiaries",
        "min": 10,
        "max": 100,
        "step": 5,
    },
    "num_trucks": {
        "type": "SliderInt",
        "value": 3,
        "label": "Number of Trucks",
        "min": 1,
        "max": 20,
        "step": 1,
    },
    "critical_days_threshold": {
        "type": "SliderInt",
        "value": 5,
        "label": "Critical Days Threshold",
        "min": 1,
        "max": 20,
        "step": 1,
    },
    "width": 20,
    "height": 20,
}

# Create the Initial Model Instance
initial_model = HumanitarianModel(
    num_beneficiaries=30, num_trucks=3, width=20, height=20
)


renderer = SpaceRenderer(initial_model, backend="matplotlib")
renderer.setup_agents(agent_portrayal)
renderer.render()

page = SolaraViz(
    model=initial_model,
    renderer=renderer,
    components=[
        make_plot_component("Avg Urgency", page=1),
        make_plot_component("Deaths", page=1),
        make_plot_component("Critical Count", page=1),
    ],
    model_params=model_params,
    name="Needs-Based Aid Distribution",
)
