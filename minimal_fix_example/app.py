"""
Minimal fix for Hotelling's Law visualization crash.
Fixed to use Mesa 3.2.0+ visualization API with proper agent rendering.
"""

import solara
from mesa.visualization import SolaraViz, SpaceRenderer
from mesa.visualization.components import AgentPortrayalStyle
from model import HotellingModel, Firm


def agent_portrayal(agent):
    """Define how agents are portrayed in the visualization."""
    if isinstance(agent, Firm):
        # FIXED: Safe access to position attribute
        pos = getattr(agent, 'position', None)
        if pos is None:
            return AgentPortrayalStyle()  # Return empty style for agents without position
            
        # FIXED: Use Mesa 3.2.0+ AgentPortrayalStyle instead of dict
        color = "blue" if agent.firm_id == 0 else "red"
        return AgentPortrayalStyle(
            x=pos[0],
            y=pos[1], 
            color=color,
            size=50,
            tooltip={
                "Firm ID": agent.firm_id,
                "Position": pos,
                "Profit": agent.profit
            }
        )
    return AgentPortrayalStyle()


# Model parameters for the interface
model_params = {
    "num_firms": {
        "type": "SliderInt",
        "value": 2,
        "label": "Number of firms:",
        "min": 2,
        "max": 5,
        "step": 1,
    },
    "grid_width": {
        "type": "SliderInt", 
        "value": 10,
        "label": "Grid width:",
        "min": 5,
        "max": 20,
        "step": 1,
    },
}

# Create the model
model = HotellingModel(num_firms=2, grid_width=10)

# FIXED: Use SpaceRenderer with proper setup for Mesa 3.2.0+
try:
    renderer = SpaceRenderer(model, backend="altair")
    renderer.setup_agents(agent_portrayal)
    renderer.render()
    
    # Create the visualization page
    page = SolaraViz(
        model,
        renderer,
        components=[],  # No additional components for now
        model_params=model_params,
        name="Hotelling's Law Model",
    )
except ImportError as e:
    # Fallback if visualization dependencies are missing
    print(f"Visualization dependencies missing: {e}")
    print("Install with: pip install mesa[viz]")
    
    @solara.component
    def Page():
        solara.Markdown("# Hotelling's Law Model")
        solara.Markdown("**Visualization dependencies missing. Install with:** `pip install mesa[viz]`")
        
        model = HotellingModel(num_firms=2, grid_width=10)
        model.step()
        
        with solara.Column():
            solara.Markdown("## Agent Status:")
            for agent in model.agents:
                pos = getattr(agent, 'position', None)
                if pos:
                    solara.Markdown(f"- **Firm {agent.firm_id}**: Position {pos}, Profit {agent.profit}")
                else:
                    solara.Markdown(f"- **Firm {agent.firm_id}**: No position!")
    
    page = Page()

if __name__ == "__main__":
    page  # This will be picked up by solara run