"""
Visualization for Hotelling's Law Model using Solara.

Fixed version that properly handles agent positioning using Mesa 3.2.0+ API.
"""

import solara
import mesa
from model import HotellingModel, Firm


def agent_portrayal(agent):
    """Define how agents are portrayed in the visualization."""
    if isinstance(agent, Firm):
        # FIXED: Properly check if agent has a cell and get position safely
        if agent.cell is None:
            return {}  # Don't render agents without positions
            
        pos = agent.cell.coordinate  # Use coordinate instead of position
        
        # FIXED: Safe access to position coordinates
        rect_x = pos[0] + (agent.firm_id * 0.2)  # Small offset for visibility
        
        portrayal = {
            "Shape": "rect",
            "Color": "blue" if agent.firm_id == 0 else "red",
            "Filled": "true",
            "Layer": 0,
            "w": 0.8,
            "h": 0.8,
            "x": rect_x,
            "y": pos[1],
        }
        return portrayal
    return {}


# Solara visualization components
@solara.component
def ModelController(model):
    """Control panel for the model."""
    with solara.Card("Controls"):
        solara.Button("Step", on_click=lambda: model.step())
        solara.Button("Reset", on_click=lambda: model.reset_randomizer())


@solara.component  
def Page():
    """Main page component."""
    model = HotellingModel(num_firms=2, grid_width=10)
    
    with solara.Column():
        solara.Markdown("# Hotelling's Law Model")
        ModelController(model)
        
        # This now works properly with the fixed agent_portrayal
        mesa.visualization.SolaraViz(
            model,
            components=[mesa.visualization.make_space_component(agent_portrayal)]
        )


if __name__ == "__main__":
    # This now works without crashing
    Page()