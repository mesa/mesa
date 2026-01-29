"""Test Solara visualizations - Modern API."""

import re
import unittest

import ipyvuetify as vw
import pytest
import solara

import mesa
import mesa.visualization.solara_viz as solara_viz_module
from mesa.space import MultiGrid, PropertyLayer
from mesa.visualization.backends.altair_backend import AltairBackend
from mesa.visualization.backends.matplotlib_backend import MatplotlibBackend
from mesa.visualization.components import AgentPortrayalStyle, PropertyLayerStyle
from mesa.visualization.solara_viz import (
    ModelCreator,
    Slider,
    SolaraViz,
    UserInputs,
    _check_model_params,
    extract_default_value,
    register_param_extractor,
)
from mesa.visualization.space_renderer import SpaceRenderer


@pytest.fixture(autouse=True)
def reset_param_extractors():
    """Reset the parameter extractors registry before each test."""
    original_extractors = solara_viz_module._param_extractors.copy()
    solara_viz_module._param_extractors.clear()
    yield
    solara_viz_module._param_extractors.clear()
    solara_viz_module._param_extractors.update(original_extractors)


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
            layer = PropertyLayer("sugar", 10, 10, 10.0, dtype=float)
            self.grid = MultiGrid(10, 10, True, property_layers=layer)
            self.grid.place_agent(mesa.Agent(self), (5, 5))

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


@pytest.fixture(autouse=True)
def clear_extractors():
    """Clear the extractor registry before and after each test."""
    solara_viz_module._param_extractors.clear()
    yield
    solara_viz_module._param_extractors.clear()


class TestExtractDefaultValue:
    """Test the extract_default_value function."""

    def test_extract_simple_value(self):
        """Test extracting a simple value from a parameter spec."""
        param_spec = {"type": "slider", "value": 10, "min": 0, "max": 100}
        assert extract_default_value(param_spec) == 10

    def test_extract_direct_value(self):
        """Test extracting a direct value (not a dict)."""
        assert extract_default_value(42) == 42
        assert extract_default_value("test") == "test"
        assert extract_default_value([1, 2, 3]) == [1, 2, 3]

    def test_extract_dict_without_value_key(self):
        """Test extracting from a dict without a 'value' key returns None."""
        param_spec = {"type": "custom", "data": "something"}
        assert extract_default_value(param_spec) is None

    def test_extract_with_custom_extractor(self):
        """Test that custom extractors are used when registered."""

        def custom_extractor(spec):
            return spec.get("custom_field", "default")

        register_param_extractor("CustomType", custom_extractor)

        param_spec = {"type": "CustomType", "custom_field": "custom_value"}
        assert extract_default_value(param_spec) == "custom_value"

    def test_fallback_to_value_key(self):
        """Test that unregistered types fall back to 'value' key."""
        param_spec = {"type": "UnregisteredType", "value": 123}
        assert extract_default_value(param_spec) == 123


class TestRegisterParamExtractor:
    """Test the register_param_extractor function."""

    def test_register_simple_extractor(self):
        """Test registering a simple extractor."""

        def my_extractor(spec):
            return spec.get("my_value")

        register_param_extractor("MyType", my_extractor)
        assert "MyType" in _param_extractors
        assert _param_extractors["MyType"] == my_extractor

    def test_register_multiple_extractors(self):
        """Test registering multiple extractors."""

        def extractor1(spec):
            return spec.get("value1")

        def extractor2(spec):
            return spec.get("value2")

        register_param_extractor("Type1", extractor1)
        register_param_extractor("Type2", extractor2)

        assert len(_param_extractors) == 2
        assert _param_extractors["Type1"] == extractor1
        assert _param_extractors["Type2"] == extractor2

    def test_overwrite_extractor(self):
        """Test that registering the same type overwrites the previous extractor."""

        def extractor1(spec):
            return "first"

        def extractor2(spec):
            return "second"

        register_param_extractor("MyType", extractor1)
        register_param_extractor("MyType", extractor2)

        assert _param_extractors["MyType"] == extractor2


class TestNestedDictExtractor:
    """Test a realistic nested dictionary extractor."""

    @pytest.fixture
    def dict_extractor(self):
        """Fixture providing a nested dict extractor."""

        def extract_dict_param(spec):
            entries = spec.get("entries", {})
            result = {}
            for key, value in entries.items():
                if isinstance(value, dict) and "value" in value:
                    result[key] = value["value"]
                elif isinstance(value, dict):
                    result[key] = extract_dict_param({"entries": value})
                else:
                    result[key] = value
            return result

        register_param_extractor("Dict", extract_dict_param)
        return extract_dict_param

    def test_flat_dict(self, dict_extractor):
        """Test extracting a flat dictionary."""
        param_spec = {
            "type": "Dict",
            "entries": {
                "param1": {"value": 10},
                "param2": {"value": "test"},
                "param3": {"value": True},
            },
        }

        result = extract_default_value(param_spec)
        assert result == {"param1": 10, "param2": "test", "param3": True}

    def test_nested_dict(self, dict_extractor):
        """Test extracting a nested dictionary."""
        param_spec = {
            "type": "Dict",
            "entries": {
                "learning_rate": {"value": 0.01},
                "optimizer": {
                    "name": {"value": "adam"},
                    "beta1": {"value": 0.9},
                    "beta2": {"value": 0.999},
                },
            },
        }

        result = extract_default_value(param_spec)
        assert result == {
            "learning_rate": 0.01,
            "optimizer": {"name": "adam", "beta1": 0.9, "beta2": 0.999},
        }

    def test_deeply_nested_dict(self, dict_extractor):
        """Test extracting a deeply nested dictionary."""
        param_spec = {
            "type": "Dict",
            "entries": {"level1": {"level2": {"level3": {"value": "deep"}}}},
        }

        result = extract_default_value(param_spec)
        assert result == {"level1": {"level2": {"level3": "deep"}}}

    def test_empty_dict(self, dict_extractor):
        """Test extracting an empty dictionary."""
        param_spec = {"type": "Dict", "entries": {}}
        result = extract_default_value(param_spec)
        assert result == {}


class TestMatrixExtractor:
    """Test a realistic matrix extractor."""

    @pytest.fixture
    def matrix_extractor(self):
        """Fixture providing a matrix extractor."""

        def extract_matrix_param(spec):
            rows = spec.get("rows", 3)
            cols = spec.get("cols", 3)
            default_value = spec.get("default_value", 0)
            return [[default_value for _ in range(cols)] for _ in range(rows)]

        register_param_extractor("Matrix", extract_matrix_param)
        return extract_matrix_param

    def test_default_matrix(self, matrix_extractor):
        """Test creating a matrix with default dimensions."""
        param_spec = {"type": "Matrix"}
        result = extract_default_value(param_spec)
        assert len(result) == 3
        assert all(len(row) == 3 for row in result)
        assert all(all(cell == 0 for cell in row) for row in result)

    def test_custom_dimensions(self, matrix_extractor):
        """Test creating a matrix with custom dimensions."""
        param_spec = {"type": "Matrix", "rows": 5, "cols": 4, "default_value": 1}
        result = extract_default_value(param_spec)
        assert len(result) == 5
        assert all(len(row) == 4 for row in result)
        assert all(all(cell == 1 for cell in row) for row in result)


class TestBackwardCompatibility:
    """Test that existing parameter types still work."""

    def test_slider_param(self):
        """Test that Slider parameters work unchanged."""
        param_spec = {"type": "SliderInt", "value": 50, "min": 0, "max": 100}
        assert extract_default_value(param_spec) == 50

    def test_checkbox_param(self):
        """Test that Checkbox parameters work unchanged."""
        param_spec = {"type": "Checkbox", "value": True}
        assert extract_default_value(param_spec) == True

    def test_choice_param(self):
        """Test that Choice parameters work unchanged."""
        param_spec = {
            "type": "Choice",
            "value": "option1",
            "choices": ["option1", "option2"],
        }
        assert extract_default_value(param_spec) == "option1"

    def test_fixed_param(self):
        """Test that fixed parameters (direct values) work unchanged."""
        assert extract_default_value(100) == 100
        assert extract_default_value("fixed_string") == "fixed_string"
        assert extract_default_value({"key": "value"}) is None  # No "value" key


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_none_param_spec(self):
        """Test handling None as parameter spec."""
        assert extract_default_value(None) is None

    def test_extractor_raises_exception(self):
        """Test that exceptions in extractors propagate."""

        def bad_extractor(spec):
            raise ValueError("Intentional error")

        register_param_extractor("BadType", bad_extractor)

        param_spec = {"type": "BadType"}
        with pytest.raises(ValueError, match="Intentional error"):
            extract_default_value(param_spec)

    def test_extractor_returns_none(self):
        """Test that extractors can return None."""

        def none_extractor(spec):
            return None

        register_param_extractor("NoneType", none_extractor)

        param_spec = {"type": "NoneType"}
        assert extract_default_value(param_spec) is None

    def test_complex_nested_structure(self):
        """Test a complex nested structure with mixed types."""

        def complex_extractor(spec):
            data = spec.get("data", {})
            result = {}
            for key, value in data.items():
                if isinstance(value, dict) and "value" in value:
                    result[key] = value["value"]
                elif isinstance(value, list):
                    result[key] = [extract_default_value(item) for item in value]
                else:
                    result[key] = value
            return result

        register_param_extractor("Complex", complex_extractor)

        param_spec = {
            "type": "Complex",
            "data": {
                "simple": {"value": 42},
                "list": [{"value": 1}, {"value": 2}, {"value": 3}],
                "direct": "string",
            },
        }

        result = extract_default_value(param_spec)
        assert result == {"simple": 42, "list": [1, 2, 3], "direct": "string"}
