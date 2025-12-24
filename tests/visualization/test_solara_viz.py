"""Test Solara visualizations - Modern API."""

import random
import unittest

import ipyvuetify as vw
import pytest
import solara

import mesa
import mesa.visualization.backends
from mesa.discrete_space import VoronoiGrid
from mesa.space import MultiGrid, PropertyLayer
from mesa.visualization.components import AgentPortrayalStyle, PropertyLayerStyle
from mesa.visualization.solara_viz import (
    ModelCreator,
    Slider,
    SolaraViz,
    UserInputs,
    _check_model_params,
)
from mesa.visualization.space_renderer import SpaceRenderer


class TestMakeUserInput(unittest.TestCase):
    """Test the UserInputs component and its field generation."""

    def test_unsupported_type(self):
        """Verify that an unsupported input type raises a ValueError."""

        @solara.component
        def Test(user_params):
            UserInputs(user_params)

        with self.assertRaisesRegex(ValueError, "not a supported input type"):
            solara.render(Test({"mock": {"type": "bogus"}}), handle_error=False)

    def test_slider_int(self):
        """Test that a SliderInt type correctly creates a Vuetify slider."""

        @solara.component
        def Test(user_params):
            UserInputs(user_params)

        options = {
            "type": "SliderInt",
            "value": 10,
            "label": "agents",
            "min": 10,
            "max": 20,
            "step": 1,
        }
        _, rc = solara.render(Test({"num_agents": options}), handle_error=False)
        slider = rc.find(vw.Slider).widget
        assert slider.v_model == 10

    def test_checkbox(self):
        """Test that a Checkbox type correctly creates a Vuetify checkbox."""

        @solara.component
        def Test(user_params):
            UserInputs(user_params)

        options = {"type": "Checkbox", "value": True, "label": "On"}
        _, rc = solara.render(Test({"on": options}), handle_error=False)
        assert rc.find(vw.Checkbox).widget.v_model is True

    def test_label_fallback(self):
        """Name should be used as fallback label if label is missing."""

        @solara.component
        def Test(user_params):
            UserInputs(user_params)

        options = {"type": "SliderInt", "value": 10}
        _, rc = solara.render(Test({"num_agents": options}), handle_error=False)
        assert rc.find(vw.Slider).widget.label == "num_agents"


@pytest.mark.parametrize("backend", ["matplotlib", "altair"])
def test_solara_viz_backends(mocker, backend):
    """Validates BOTH backends using the modern API.

    This resolves Issue #2993 by ensuring Altair coverage parity with
    Matplotlib for AgentPortrayalStyle and PropertyLayerStyle.
    """
    spy_structure = mocker.spy(SpaceRenderer, "draw_structure")
    spy_agents = mocker.spy(SpaceRenderer, "draw_agents")
    spy_properties = mocker.spy(SpaceRenderer, "draw_propertylayer")

    class MockModel(mesa.Model):
        def __init__(self):
            super().__init__()
            layer = PropertyLayer("sugar", 10, 10, 10.0)
            self.grid = MultiGrid(10, 10, True, property_layers=layer)
            self.grid.place_agent(mesa.Agent(self), (5, 5))

    model = MockModel()

    def agent_portrayal(agent):
        return AgentPortrayalStyle(marker="o", color="gray")

    def property_portrayal(_):
        return PropertyLayerStyle(colormap="viridis", alpha=0.5)

    renderer = (
        SpaceRenderer(model, backend=backend)
        .setup_agents(agent_portrayal)
        .setup_propertylayer(property_portrayal)
        .render()
    )

    solara.render(SolaraViz(model, renderer, components=[]))

    assert renderer.backend == backend
    spy_structure.assert_called_once()
    spy_agents.assert_called_once()
    spy_properties.assert_called_once()


def test_voronoi_grid_renderer():
    """Test SpaceRenderer with VoronoiGrid using modern API."""

    def agent_portrayal(agent):
        return AgentPortrayalStyle(marker="o", color="blue")

    model = mesa.Model()
    model.grid = VoronoiGrid(
        centroids_coordinates=[(0, 1), (0, 0), (1, 0)],
        random=random.Random(42),
    )
    renderer = (
        SpaceRenderer(model, backend="matplotlib")
        .setup_agents(agent_portrayal)
        .render()
    )
    solara.render(SolaraViz(model, renderer))
    assert renderer.backend == "matplotlib"


def test_slider():
    """Test the standalone Slider component properties."""
    slider = Slider("Test", 0.8, 0.1, 1.0, 0.1)
    assert slider.is_float_slider
    assert slider.value == 0.8


def test_model_param_checks():
    """Test that model parameter validation works as expected."""

    def mock_init(self, req, opt=10):
        pass

    _check_model_params(mock_init, {"req": 1})
    with pytest.raises(ValueError, match="Missing required model parameter"):
        _check_model_params(mock_init, {})


def test_model_creator():
    """Test the ModelCreator component initialization."""

    class ModelWithRequiredParam:
        def __init__(self, param1):
            pass

    solara.render(
        ModelCreator(
            solara.reactive(ModelWithRequiredParam(param1="mock")),
            user_params={"param1": 1},
        ),
        handle_error=False,
    )


def test_check_model_params_with_args_only():
    """Test that _check_model_params raises ValueError when *args are present."""

    def mock_init(self, param1, *args):
        pass

    with pytest.raises(ValueError, match="keyword arguments"):
        _check_model_params(mock_init, {"param1": 1})
