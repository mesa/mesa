"""Test Solara visualizations - Modern API.

This is the primary test file for Mesa's visualization components.
Uses AgentPortrayalStyle, PropertyLayerStyle, and SpaceRenderer APIs.

For backwards compatibility tests with dict-based portrayals, see test_solara_viz_legacy.py.
"""

import random
import re
import unittest

import ipyvuetify as vw
import pytest
import solara

import mesa
import mesa.visualization.backends
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


class TestMakeUserInput(unittest.TestCase):  # noqa: D101
    def test_unsupported_type(self):  # noqa: D102
        @solara.component
        def Test(user_params):
            UserInputs(user_params)

        """unsupported input type should raise ValueError"""
        # bogus type
        with self.assertRaisesRegex(ValueError, "not a supported input type"):
            solara.render(Test({"mock": {"type": "bogus"}}), handle_error=False)

        # no type is specified
        with self.assertRaisesRegex(ValueError, "not a supported input type"):
            solara.render(Test({"mock": {}}), handle_error=False)

    def test_slider_int(self):  # noqa: D102
        @solara.component
        def Test(user_params):
            UserInputs(user_params)

        options = {
            "type": "SliderInt",
            "value": 10,
            "label": "number of agents",
            "min": 10,
            "max": 20,
            "step": 1,
        }
        user_params = {"num_agents": options}
        _, rc = solara.render(Test(user_params), handle_error=False)
        slider_int = rc.find(vw.Slider).widget

        assert slider_int.v_model == options["value"]
        assert slider_int.label == options["label"]
        assert slider_int.min == options["min"]
        assert slider_int.max == options["max"]
        assert slider_int.step == options["step"]

    def test_checkbox(self):  # noqa: D102
        @solara.component
        def Test(user_params):
            UserInputs(user_params)

        options = {"type": "Checkbox", "value": True, "label": "On"}
        user_params = {"num_agents": options}
        _, rc = solara.render(Test(user_params), handle_error=False)
        checkbox = rc.find(vw.Checkbox).widget

        assert checkbox.v_model == options["value"]
        assert checkbox.label == options["label"]

    def test_label_fallback(self):
        """Name should be used as fallback label."""

        @solara.component
        def Test(user_params):
            UserInputs(user_params)

        options = {
            "type": "SliderInt",
            "value": 10,
        }

        user_params = {"num_agents": options}
        _, rc = solara.render(Test(user_params), handle_error=False)
        slider_int = rc.find(vw.Slider).widget

        assert slider_int.v_model == options["value"]
        assert slider_int.label == "num_agents"
        assert slider_int.min is None
        assert slider_int.max is None
        assert slider_int.step is None

    def test_input_text_field(self):
        """Test that "InputText" type creates a vw.TextField."""

        @solara.component
        def Test(user_params):
            UserInputs(user_params)

        options = {"type": "InputText", "value": "JohnDoe", "label": "Agent Name"}

        user_params = {"agent_name": options}

        _, rc = solara.render(Test(user_params), handle_error=False)

        textfield = rc.find(vw.TextField).widget

        assert textfield.v_model == options["value"]
        assert textfield.label == options["label"]


def test_call_space_drawer(mocker):
    """Test the call to space drawer."""
    mock_draw_space = mocker.spy(
        mesa.visualization.space_renderer.SpaceRenderer, "draw_structure"
    )
    mock_draw_agents = mocker.spy(
        mesa.visualization.space_renderer.SpaceRenderer, "draw_agents"
    )
    mock_draw_properties = mocker.spy(
        mesa.visualization.space_renderer.SpaceRenderer, "draw_propertylayer"
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

    def agent_portrayal(agent):
        return AgentPortrayalStyle(marker="o", color="gray")

    propertylayer_portrayal = None

    renderer = (
        SpaceRenderer(model, backend="matplotlib")
        .setup_agents(agent_portrayal)
        .setup_propertylayer(propertylayer_portrayal)
        .render()
    )

    # component must be rendered for code to run
    solara.render(
        SolaraViz(
            model,
            renderer,
            components=[],
        )
    )

    assert renderer.backend == "matplotlib"
    assert isinstance(
        renderer.backend_renderer, mesa.visualization.backends.MatplotlibBackend
    )

    mock_draw_space.assert_called_with(renderer)
    mock_draw_agents.assert_called_with(renderer)
    # should not call this method if portrayal is None
    mock_draw_properties.assert_not_called()

    mock_draw_space.reset_mock()
    mock_draw_agents.reset_mock()
    mock_draw_properties.reset_mock()

    solara.render(SolaraViz(model))

    # noting is drawn if renderer is not passed
    assert mock_draw_space.call_count == 0
    assert mock_draw_agents.call_count == 0
    assert mock_draw_properties.call_count == 0

    # checking if SpaceAltair is working as intended with post_process
    def propertylayer_portrayal(_):
        return PropertyLayerStyle(
            colormap="pastel1",
            alpha=0.75,
            colorbar=True,
            vmin=0,
            vmax=10,
        )

    solara.render(SolaraViz(model, renderer, components=[]))

    renderer = (
        SpaceRenderer(model, backend="altair")
        .setup_agents(agent_portrayal)
        .setup_propertylayer(propertylayer_portrayal)
        .render()
    )

    assert renderer.backend == "altair"
    assert isinstance(
        renderer.backend_renderer, mesa.visualization.backends.AltairBackend
    )

    mock_draw_space.assert_called_with(renderer)
    mock_draw_agents.assert_called_with(renderer)
    mock_draw_properties.assert_called_with(renderer)

    mock_draw_space.reset_mock()
    mock_draw_agents.reset_mock()
    mock_draw_properties.reset_mock()

    solara.render(SolaraViz(model))

    # nothing is drawn if renderer is not passed
    assert mock_draw_space.call_count == 0
    assert mock_draw_agents.call_count == 0
    assert mock_draw_properties.call_count == 0

    # specify a custom space method
    class AltSpace:
        @staticmethod
        def drawer(model):
            return

    # check to verify that components are passed with the model instance
    altspace_drawer = mocker.spy(AltSpace, "drawer")
    solara.render(SolaraViz(model, components=[AltSpace.drawer]))
    altspace_drawer.assert_called_with(model)


def test_voronoi_grid_renderer():
    """Test SpaceRenderer with VoronoiGrid using modern API."""

    def agent_portrayal(agent):
        return AgentPortrayalStyle(marker="o", color="blue")

    voronoi_model = mesa.Model()
    voronoi_model.grid = mesa.discrete_space.VoronoiGrid(
        centroids_coordinates=[(0, 1), (0, 0), (1, 0)],
        random=random.Random(42),
    )

    renderer = (
        SpaceRenderer(voronoi_model, backend="matplotlib")
        .setup_agents(agent_portrayal)
        .render()
    )

    # Should not raise
    solara.render(SolaraViz(voronoi_model, renderer, components=[]))


def test_altair_backend_with_propertylayer(mocker):
    """Test Altair backend with PropertyLayerStyle using modern API."""
    mock_draw_space = mocker.spy(
        mesa.visualization.space_renderer.SpaceRenderer, "draw_structure"
    )
    mock_draw_agents = mocker.spy(
        mesa.visualization.space_renderer.SpaceRenderer, "draw_agents"
    )
    mock_draw_properties = mocker.spy(
        mesa.visualization.space_renderer.SpaceRenderer, "draw_propertylayer"
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

    def agent_portrayal(agent):
        return AgentPortrayalStyle(marker="o", color="gray")

    def propertylayer_portrayal(_):
        return PropertyLayerStyle(
            colormap="pastel1",
            alpha=0.75,
            colorbar=True,
            vmin=0,
            vmax=10,
        )

    renderer = (
        SpaceRenderer(model, backend="altair")
        .setup_agents(agent_portrayal)
        .setup_propertylayer(propertylayer_portrayal)
        .render()
    )

    solara.render(SolaraViz(model, renderer, components=[]))

    assert renderer.backend == "altair"
    assert isinstance(
        renderer.backend_renderer, mesa.visualization.backends.AltairBackend
    )

    mock_draw_space.assert_called_with(renderer)
    mock_draw_agents.assert_called_with(renderer)
    mock_draw_properties.assert_called_with(renderer)


def test_matplotlib_backend_with_propertylayer(mocker):
    """Test Matplotlib backend with PropertyLayerStyle using modern API."""
    mock_draw_properties = mocker.spy(
        mesa.visualization.space_renderer.SpaceRenderer, "draw_propertylayer"
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

    def agent_portrayal(agent):
        return AgentPortrayalStyle(marker="o", color="gray")

    def propertylayer_portrayal(_):
        return PropertyLayerStyle(
            colormap="viridis",
            alpha=0.5,
            colorbar=True,
            vmin=0,
            vmax=10,
        )

    renderer = (
        SpaceRenderer(model, backend="matplotlib")
        .setup_agents(agent_portrayal)
        .setup_propertylayer(propertylayer_portrayal)
        .render()
    )

    solara.render(SolaraViz(model, renderer, components=[]))

    assert renderer.backend == "matplotlib"
    assert isinstance(
        renderer.backend_renderer, mesa.visualization.backends.MatplotlibBackend
    )

    mock_draw_properties.assert_called_with(renderer)


def test_slider():
    """Test the Slider component."""
    slider_float = Slider("Agent density", 0.8, 0.1, 1.0, 0.1)
    assert slider_float.is_float_slider
    assert slider_float.value == 0.8
    assert slider_float.get("value") == 0.8
    assert slider_float.min == 0.1
    assert slider_float.max == 1.0
    assert slider_float.step == 0.1
    slider_int = Slider("Homophily", 3, 0, 8, 1)
    assert not slider_int.is_float_slider
    slider_dtype_float = Slider("Homophily", 3, 0, 8, 1, dtype=float)
    assert slider_dtype_float.is_float_slider


def test_model_param_checks():
    """Test the model parameter checks."""

    class ModelWithOptionalParams:
        def __init__(self, required_param, optional_param=10):
            pass

    class ModelWithOnlyRequired:
        def __init__(self, param1, param2):
            pass

    class ModelWithKwargs:
        def __init__(self, **kwargs):
            pass

    # Test that optional params can be omitted
    _check_model_params(ModelWithOptionalParams.__init__, {"required_param": 1})

    # Test that optional params can be provided
    _check_model_params(
        ModelWithOptionalParams.__init__, {"required_param": 1, "optional_param": 5}
    )

    # Test that model_params are accepted if model uses **kwargs
    _check_model_params(ModelWithKwargs.__init__, {"another_kwarg": 6})

    # test hat kwargs are accepted even if no model_params are specified
    _check_model_params(ModelWithKwargs.__init__, {})

    # Test invalid parameter name raises ValueError
    with pytest.raises(
        ValueError, match=re.escape("Invalid model parameter: invalid_param")
    ):
        _check_model_params(
            ModelWithOptionalParams.__init__, {"required_param": 1, "invalid_param": 2}
        )

    # Test missing required parameter raises ValueError
    with pytest.raises(
        ValueError, match=re.escape("Missing required model parameter: param2")
    ):
        _check_model_params(ModelWithOnlyRequired.__init__, {"param1": 1})

    # Test passing extra parameters raises ValueError
    with pytest.raises(ValueError, match=re.escape("Invalid model parameter: extra")):
        _check_model_params(
            ModelWithOnlyRequired.__init__, {"param1": 1, "param2": 2, "extra": 3}
        )

    # Test empty params dict raises ValueError if required params
    with pytest.raises(ValueError, match=re.escape("Missing required model parameter")):
        _check_model_params(ModelWithOnlyRequired.__init__, {})


def test_model_creator():  # noqa: D103
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

    solara.render(
        ModelCreator(
            solara.reactive(ModelWithRequiredParam(param1="mock")),
            user_params={"param1": Slider("Param1", 10, 10, 100, 1)},
        ),
        handle_error=False,
    )

    with pytest.raises(ValueError, match=re.escape("Missing required model parameter")):
        solara.render(
            ModelCreator(
                solara.reactive(ModelWithRequiredParam(param1="mock")), user_params={}
            ),
            handle_error=False,
        )


# test that _check_model_params raises ValueError when *args are present
def test_check_model_params_with_args_only():
    """Test that _check_model_params raises ValueError when *args are present."""

    class ModelWithArgsOnly:
        def __init__(self, param1, *args):
            pass

    model_params = {"param1": 1}

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Mesa's visualization requires the use of keyword arguments to ensure the parameters are passed to Solara correctly. Please ensure all model parameters are of form param=value"
        ),
    ):
        _check_model_params(ModelWithArgsOnly.__init__, model_params)
