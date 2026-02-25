"""Test Solara visualizations - Modern API."""

import random
import re
import unittest

import ipyvuetify as vw
import pytest
import solara

import mesa
from mesa.discrete_space import CellAgent, OrthogonalMooreGrid, PropertyLayer
from mesa.experimental.scenarios import Scenario
from mesa.visualization.backends.altair_backend import AltairBackend
from mesa.visualization.backends.matplotlib_backend import MatplotlibBackend
from mesa.visualization.components import AgentPortrayalStyle, PropertyLayerStyle
from mesa.visualization.solara_viz import (
    ModelCreator,
    Slider,
    SolaraViz,
    UserInputs,
    _build_model_init_kwargs,
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

    def test_input_text_field(self):
        """Test that 'InputText' type correctly creates a vw.TextField."""

        @solara.component
        def Test(user_params):
            UserInputs(user_params)

        options = {"type": "InputText", "value": "JohnDoe", "label": "Agent Name"}
        _, rc = solara.render(Test({"agent_name": options}), handle_error=False)
        textfield = rc.find(vw.TextField).widget
        assert textfield.v_model == "JohnDoe"
        assert textfield.label == "Agent Name"

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

        options = {"type": "SliderInt", "value": 10}
        user_params = {"num_agents": options}
        _, rc = solara.render(Test(user_params), handle_error=False)
        slider_int = rc.find(vw.Slider).widget

        assert slider_int.v_model == options["value"]
        assert slider_int.label == "num_agents"
        assert slider_int.min is None
        assert slider_int.max is None
        assert slider_int.step is None


@pytest.mark.parametrize("backend", ["matplotlib", "altair"])
def test_solara_viz_backends(mocker, backend):
    """Validates BOTH backends using the modern API."""
    spy_structure = mocker.spy(SpaceRenderer, "draw_structure")
    spy_agents = mocker.spy(SpaceRenderer, "draw_agents")
    spy_properties = mocker.spy(SpaceRenderer, "draw_propertylayer")

    class MockModel(mesa.Model):
        def __init__(self):
            super().__init__()
            # Include property layer to verify it gets drawn
            layer = PropertyLayer("sugar", (10, 10), default_value=10.0, dtype=float)

            self.grid = OrthogonalMooreGrid(
                (10, 10), torus=True, random=random.Random(42)
            )
            self.grid.add_property_layer(layer)

            agent = CellAgent(self)
            agent.cell = self.grid[
                (
                    5,
                    5,
                )
            ]

    model = MockModel()

    def agent_portrayal(_):
        return AgentPortrayalStyle(marker="o", color="gray")

    def property_portrayal(_):
        return PropertyLayerStyle(colormap="viridis")

    renderer = (
        SpaceRenderer(model, backend=backend)
        .setup_agents(agent_portrayal)
        .setup_propertylayer(property_portrayal)
        .render()
    )

    solara.render(SolaraViz(model, renderer, components=[]))

    assert renderer.backend == backend

    if backend == "matplotlib":
        assert isinstance(renderer.backend_renderer, MatplotlibBackend)
    elif backend == "altair":
        assert isinstance(renderer.backend_renderer, AltairBackend)

    spy_structure.assert_called_with(renderer)
    spy_agents.assert_called_with(renderer)
    spy_properties.assert_called_with(renderer)

    # Test that nothing is drawn if the renderer is not passed
    spy_structure.reset_mock()
    spy_agents.reset_mock()
    spy_properties.reset_mock()
    solara.render(SolaraViz(model))
    assert spy_structure.call_count == 0
    assert spy_agents.call_count == 0
    assert spy_properties.call_count == 0


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


def test_solara_viz_with_scenario():
    """Test SolaraViz with scenario-enabled models."""

    class TestScenario(Scenario):
        density: float = 0.8
        vision: int = 7

    class TestModel(mesa.Model):
        def __init__(self, height=20, width=20, scenario: TestScenario | None = None):
            super().__init__(scenario=scenario)
            self.height = height
            self.width = width

    scenario = TestScenario(density=0.5, vision=10)
    model = TestModel(scenario=scenario)

    model_params = {
        "height": Slider("Height", 30, 10, 50, 5),
        "width": Slider("Width", 25, 10, 50, 5),
        "density": Slider("Density", 0.7, 0.0, 1.0, 0.1),  # Scenario param
        "vision": Slider("Vision", 5, 1, 15, 1),  # Scenario param
    }

    # Should render without error
    solara.render(SolaraViz(model, model_params=model_params), handle_error=False)


def test_model_creator_with_scenario():
    """Test ModelCreator component with scenario parameters."""

    class TestScenario(Scenario):
        param1: float = 0.5
        param2: int = 10

    class TestModel(mesa.Model):
        def __init__(self, model_param=5, scenario: TestScenario | None = None):
            super().__init__(scenario=scenario)
            self.model_param = model_param

    scenario = TestScenario(param1=0.8, param2=15)
    model = TestModel(model_param=10, scenario=scenario)

    # Only include model parameters in user_params for ModelCreator
    # Scenario parameters are handled internally by the parameter splitting logic
    user_params = {
        "model_param": 20,
    }

    # Should render without error
    solara.render(
        ModelCreator(
            solara.reactive(model),
            user_params=user_params,
        ),
        handle_error=False,
    )


def test_parameter_splitting_logic():
    """Test the core parameter splitting logic used in ModelController."""

    class TestScenario(Scenario):
        scenario_param1: float = 0.5
        scenario_param2: int = 10

    class TestModel(mesa.Model):
        def __init__(
            self, model_param1=5, model_param2=15, scenario: TestScenario | None = None
        ):
            super().__init__(scenario=scenario)
            self.model_param1 = model_param1
            self.model_param2 = model_param2

    # Test the splitting logic
    scenario = TestScenario(scenario_param1=0.8, scenario_param2=20)
    model = TestModel(model_param1=10, model_param2=25, scenario=scenario)

    # Mock model parameters (mixed model and scenario params)
    model_parameters = {
        "model_param1": 15,
        "model_param2": 30,
        "scenario_param1": 0.9,  # Should go to scenario
        "scenario_param2": 25,  # Should go to scenario
    }

    # Test the splitting logic using the actual helper
    kwargs = _build_model_init_kwargs(
        model,
        model_parameters,
        add_scenario_when_empty=True,
        require_model_accepts_scenario=True,
    )

    # Verify the split
    assert "model_param1" in kwargs
    assert "model_param2" in kwargs
    assert "scenario" in kwargs
    assert isinstance(kwargs["scenario"], TestScenario)
    assert kwargs["scenario"].scenario_param1 == 0.9
    assert kwargs["scenario"].scenario_param2 == 25
    assert kwargs["model_param1"] == 15
    assert kwargs["model_param2"] == 30
