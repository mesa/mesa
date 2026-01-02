"""Test Solara visualizations - Legacy/Backwards Compatibility.

This file ensures that deprecated dict-based portrayals still work.
NOTE: This file can be removed when legacy support is dropped.
"""

import random

import solara

import mesa
from mesa.discrete_space import VoronoiGrid
from mesa.space import MultiGrid, PropertyLayer
from mesa.visualization import SolaraViz
from mesa.visualization.components import AgentPortrayalStyle, PropertyLayerStyle
from mesa.visualization.components.altair_components import make_altair_space
from mesa.visualization.components.matplotlib_components import make_mpl_space_component


def test_legacy_dict_portrayal_support(mocker):
    """Verify that deprecated dictionary-based portrayals still work for agents."""
    mock_mpl = mocker.spy(
        mesa.visualization.components.matplotlib_components, "SpaceMatplotlib"
    )
    mock_altair = mocker.spy(
        mesa.visualization.components.altair_components, "SpaceAltair"
    )

    class MockModel(mesa.Model):
        def __init__(self):
            super().__init__()
            self.grid = MultiGrid(10, 10, True)
            agent = mesa.Agent(self)
            self.grid.place_agent(agent, (5, 5))

    model = MockModel()

    def agent_portrayal(_):
        return {"marker": "o", "color": "gray"}

    # Test with matplotlib component
    solara.render(
        SolaraViz(model, components=[(make_mpl_space_component(agent_portrayal), 0)])
    )
    mock_mpl.assert_called()

    # Test with altair component
    solara.render(
        SolaraViz(model, components=[(make_altair_space(agent_portrayal), 0)])
    )
    mock_altair.assert_called()


def test_legacy_property_layer_portrayal(mocker):
    """Include test for older property layer portrayal here, as requested."""
    mock_mpl = mocker.spy(
        mesa.visualization.components.matplotlib_components, "SpaceMatplotlib"
    )

    class MockModel(mesa.Model):
        def __init__(self):
            super().__init__()
            layer = PropertyLayer("sugar", width=10, height=10, default_value=0)
            self.grid = MultiGrid(10, 10, True, property_layers=layer)
            agent = mesa.Agent(self)
            self.grid.place_agent(agent, (5, 5))

    model = MockModel()
    property_portrayal = {"sugar": {"colormap": "viridis"}}

    solara.render(
        SolaraViz(
            model,
            components=[
                (
                    make_mpl_space_component(
                        agent_portrayal=None,
                        propertylayer_portrayal=property_portrayal,
                    ),
                    0,
                )
            ],
        )
    )

    args, _ = mock_mpl.call_args
    # args are (model, agent_portrayal, propertylayer_portrayal)
    assert args[2] == property_portrayal


def test_call_space_drawer_full(mocker):
    """Test the legacy space drawer component APIs in full.

    This was moved from test_solara_viz.py to preserve coverage
    of the legacy make_mpl_space_component and make_altair_space APIs.
    """
    mock_space_matplotlib = mocker.spy(
        mesa.visualization.components.matplotlib_components, "SpaceMatplotlib"
    )
    mock_space_altair = mocker.spy(
        mesa.visualization.components.altair_components, "SpaceAltair"
    )
    mock_chart_property_layer = mocker.spy(
        mesa.visualization.components.altair_components, "chart_property_layers"
    )

    class MockAgent(mesa.Agent):
        def __init__(self, model):
            super().__init__(model)

    class MockModel(mesa.Model):
        def __init__(self, seed=None):
            super().__init__(seed=seed)
            layer1 = PropertyLayer(
                name="sugar", width=10, height=10, default_value=10.0, dtype=float
            )
            self.grid = MultiGrid(
                width=10, height=10, torus=True, property_layers=layer1
            )
            a = MockAgent(self)
            self.grid.place_agent(a, (5, 5))

    model = MockModel()

    def agent_portrayal(_):
        return AgentPortrayalStyle(marker="o", color="gray")

    # FIX: Define a property_portrayal that returns the new PropertyLayerStyle
    def property_portrayal(_):
        return PropertyLayerStyle(colormap="viridis")

    # Test compatibility of new style objects with the legacy Matplotlib component
    solara.render(
        SolaraViz(
            model,
            components=[
                make_mpl_space_component(
                    agent_portrayal=agent_portrayal,
                    propertylayer_portrayal=property_portrayal,
                )
            ],
        )
    )
    # Assert that the old component was called correctly with the NEW style objects
    mock_space_matplotlib.assert_called_with(
        model, agent_portrayal, property_portrayal, post_process=None
    )

    # specify no space should be drawn
    mock_space_matplotlib.reset_mock()
    solara.render(SolaraViz(model, components="default"))
    assert mock_space_matplotlib.call_count == 0
    assert mock_space_altair.call_count == 1  # altair is the default method

    # checking if SpaceAltair is working as intended with a dict-based portrayal
    propertylayer_portrayal_dict = {
        "sugar": {
            "colormap": "pastel1",
            "alpha": 0.75,
            "colorbar": True,
            "vmin": 0,
            "vmax": 10,
        }
    }
    mock_post_process = mocker.MagicMock()
    solara.render(
        SolaraViz(
            model,
            components=[
                make_altair_space(
                    agent_portrayal,
                    post_process=mock_post_process,
                    propertylayer_portrayal=propertylayer_portrayal_dict,
                )
            ],
        )
    )

    args, kwargs = mock_space_altair.call_args
    assert args == (model, agent_portrayal)
    assert kwargs["propertylayer_portrayal"] == propertylayer_portrayal_dict
    mock_post_process.assert_called_once()
    assert mock_chart_property_layer.call_count == 1
    assert mock_space_matplotlib.call_count == 0

    mock_space_altair.reset_mock()
    mock_space_matplotlib.reset_mock()
    mock_post_process.reset_mock()
    mock_chart_property_layer.reset_mock()

    # specify a custom space method
    class AltSpace:
        @staticmethod
        def drawer(model):
            return

    altspace_drawer = mocker.spy(AltSpace, "drawer")
    solara.render(SolaraViz(model, components=[AltSpace.drawer]))
    altspace_drawer.assert_called_with(model)

    # check voronoi space drawer
    voronoi_model = mesa.Model()
    voronoi_model.grid = VoronoiGrid(
        centroids_coordinates=[(0, 1), (0, 0), (1, 0)],
        random=random.Random(42),
    )
    solara.render(
        SolaraViz(voronoi_model, components=[make_mpl_space_component(agent_portrayal)])
    )
