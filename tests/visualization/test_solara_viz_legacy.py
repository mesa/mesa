"""Test Solara visualizations - Legacy/Backwards Compatibility.

This file ensures that deprecated dict-based portrayals still work.
NOTE: This file can be removed when legacy support is dropped.
"""

import solara

import mesa
from mesa.space import MultiGrid
from mesa.visualization import SolaraViz
from mesa.visualization.components.altair_components import make_altair_space
from mesa.visualization.components.matplotlib_components import make_mpl_space_component


def test_legacy_dict_portrayal_support():
    """Verify that deprecated dictionary-based portrayals still work."""

    class MockModel(mesa.Model):
        def __init__(self):
            super().__init__()
            self.grid = MultiGrid(10, 10, True)
            # CodeRabbit Rec: Add agent to validate portrayal logic
            agent = mesa.Agent(self)
            self.grid.place_agent(agent, (5, 5))

    model = MockModel()

    def agent_portrayal(_):
        return {"marker": "o", "color": "gray"}

    # CodeRabbit Rec: Wrap components in tuples (component, page_index)
    solara.render(
        SolaraViz(model, components=[(make_mpl_space_component(agent_portrayal), 0)])
    )
    solara.render(
        SolaraViz(model, components=[(make_altair_space(agent_portrayal), 0)])
    )
