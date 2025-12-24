"""Test Solara visualizations - Modern API.

This is the primary test file for Mesa's visualization components.
Uses AgentPortrayalStyle, PropertyLayerStyle, and SpaceRenderer APIs.

For backwards compatibility tests with dict-based portrayals, see test_solara_viz_legacy.py.

Test Coverage:
- SpaceRenderer with both matplotlib and altair backends
- AgentPortrayalStyle attribute handling
- PropertyLayerStyle visualization
- Multiple space types (Orthogonal, Hex, Voronoi, Continuous, Network)
- User input components (Slider, Checkbox, InputText)
- Model parameter validation
- Custom space components
- Error handling for invalid configurations
"""

import random
import re
import unittest

import ipyvuetify as vw
import pytest
import solara

import mesa
import mesa.visualization.backends
from mesa.space import (
    ContinuousSpace,
    HexMultiGrid,
    MultiGrid,
    PropertyLayer,
)
from mesa.visualization.components import AgentPortrayalStyle, PropertyLayerStyle
from mesa.visualization.solara_viz import (
    ModelCreator,
    Slider,
    SolaraViz,
    UserInputs,
    _check_model_params,
)
from mesa.visualization.space_renderer import SpaceRenderer

# --- Fixtures and Helpers ---


class MockAgent(mesa.Agent):
    """A minimal mock agent for testing visualization."""

    def __init__(self, model):
        """Initialize the mock agent."""
        super().__init__(model)


class MockModel(mesa.Model):
    """A minimal mock model for testing visualization."""

    def __init__(self, seed=None):
        """Initialize the mock model with a grid and property layer."""
        super().__init__(seed=seed)
        layer1 = PropertyLayer(
            name="sugar", width=10, height=10, default_value=10.0, dtype=float
        )
        self.grid = MultiGrid(width=10, height=10, torus=True, property_layers=layer1)
        a = MockAgent(self)
        self.grid.place_agent(a, (5, 5))


@pytest.fixture
def mock_model():
    """Fixture that provides a MockModel instance."""
    return MockModel()


def agent_portrayal(agent):
    """Standard agent portrayal using modern AgentPortrayalStyle."""
    return AgentPortrayalStyle(marker="o", color="gray", size=50)


def propertylayer_portrayal(_):
    """Standard property layer portrayal using modern PropertyLayerStyle."""
    return PropertyLayerStyle(
        colormap="viridis",
        alpha=0.5,
        colorbar=True,
        vmin=0,
        vmax=10,
    )


# --- Additional Model Fixtures for Different Space Types ---


class HexMultiGridModel(mesa.Model):
    """Mock model with legacy HexMultiGrid for testing."""

    def __init__(self, seed=None):
        """Initialize model with HexMultiGrid."""
        super().__init__(seed=seed)
        self.grid = HexMultiGrid(width=10, height=10, torus=True)
        a = MockAgent(self)
        self.grid.place_agent(a, (5, 5))


class ContinuousSpaceModel(mesa.Model):
    """Mock model with ContinuousSpace for testing."""

    def __init__(self, seed=None):
        """Initialize model with ContinuousSpace."""
        super().__init__(seed=seed)
        self.space = ContinuousSpace(x_max=100, y_max=100, torus=True)
        a = MockAgent(self)
        self.space.place_agent(a, (50.0, 50.0))


class VoronoiModel(mesa.Model):
    """Mock model with VoronoiGrid for testing."""

    def __init__(self, seed=None):
        """Initialize model with VoronoiGrid."""
        super().__init__(seed=seed)
        self.grid = mesa.discrete_space.VoronoiGrid(
            centroids_coordinates=[(0, 1), (0, 0), (1, 0), (1, 1)],
            random=random.Random(42),
        )


@pytest.fixture
def continuous_model():
    """Fixture that provides a ContinuousSpaceModel instance."""
    return ContinuousSpaceModel()


@pytest.fixture
def voronoi_model():
    """Fixture that provides a VoronoiModel instance."""
    return VoronoiModel()


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


# --- Parametrized Backend Tests ---


@pytest.mark.parametrize("backend", ["matplotlib", "altair"])
def test_space_renderer_with_backend(mocker, mock_model, backend):
    """Test SpaceRenderer works correctly with both matplotlib and altair backends."""
    mock_draw_space = mocker.spy(SpaceRenderer, "draw_structure")
    mock_draw_agents = mocker.spy(SpaceRenderer, "draw_agents")

    renderer = (
        SpaceRenderer(mock_model, backend=backend)
        .setup_agents(agent_portrayal)
        .render()
    )

    solara.render(SolaraViz(mock_model, renderer, components=[]))

    assert renderer.backend == backend
    mock_draw_space.assert_called_with(renderer)
    mock_draw_agents.assert_called_with(renderer)


@pytest.mark.parametrize("backend", ["matplotlib", "altair"])
def test_space_renderer_with_propertylayer(mocker, mock_model, backend):
    """Test SpaceRenderer with PropertyLayerStyle for both backends."""
    mock_draw_properties = mocker.spy(SpaceRenderer, "draw_propertylayer")

    renderer = (
        SpaceRenderer(mock_model, backend=backend)
        .setup_agents(agent_portrayal)
        .setup_propertylayer(propertylayer_portrayal)
        .render()
    )

    solara.render(SolaraViz(mock_model, renderer, components=[]))

    assert renderer.backend == backend
    mock_draw_properties.assert_called_with(renderer)


@pytest.mark.parametrize("backend", ["matplotlib", "altair"])
def test_backend_instance_type(mock_model, backend):
    """Test that the correct backend instance is created."""
    renderer = (
        SpaceRenderer(mock_model, backend=backend)
        .setup_agents(agent_portrayal)
        .render()
    )

    if backend == "matplotlib":
        assert isinstance(
            renderer.backend_renderer, mesa.visualization.backends.MatplotlibBackend
        )
    else:
        assert isinstance(
            renderer.backend_renderer, mesa.visualization.backends.AltairBackend
        )


@pytest.mark.parametrize("backend", ["matplotlib", "altair"])
def test_no_propertylayer_portrayal(mocker, mock_model, backend):
    """Test that draw_propertylayer is not called when portrayal is None."""
    mock_draw_properties = mocker.spy(SpaceRenderer, "draw_propertylayer")

    renderer = (
        SpaceRenderer(mock_model, backend=backend)
        .setup_agents(agent_portrayal)
        .setup_propertylayer(None)
        .render()
    )

    solara.render(SolaraViz(mock_model, renderer, components=[]))

    mock_draw_properties.assert_not_called()


# --- Non-parametrized Tests ---


def test_no_renderer_passed(mocker):
    """Test that nothing is drawn if renderer is not passed."""
    mock_draw_space = mocker.spy(SpaceRenderer, "draw_structure")
    mock_draw_agents = mocker.spy(SpaceRenderer, "draw_agents")
    mock_draw_properties = mocker.spy(SpaceRenderer, "draw_propertylayer")

    model = MockModel()
    solara.render(SolaraViz(model))

    assert mock_draw_space.call_count == 0
    assert mock_draw_agents.call_count == 0
    assert mock_draw_properties.call_count == 0


def test_custom_space_component(mocker):
    """Test that custom space drawer components are called correctly."""
    model = MockModel()

    class AltSpace:
        @staticmethod
        def drawer(model):
            return

    altspace_drawer = mocker.spy(AltSpace, "drawer")
    solara.render(SolaraViz(model, components=[AltSpace.drawer]))
    altspace_drawer.assert_called_with(model)


def test_voronoi_grid_renderer():
    """Test SpaceRenderer with VoronoiGrid using modern API."""

    def voronoi_agent_portrayal(agent):
        return AgentPortrayalStyle(marker="o", color="blue")

    voronoi_model = mesa.Model()
    voronoi_model.grid = mesa.discrete_space.VoronoiGrid(
        centroids_coordinates=[(0, 1), (0, 0), (1, 0)],
        random=random.Random(42),
    )

    renderer = (
        SpaceRenderer(voronoi_model, backend="matplotlib")
        .setup_agents(voronoi_agent_portrayal)
        .render()
    )

    # Should not raise
    solara.render(SolaraViz(voronoi_model, renderer, components=[]))


@pytest.mark.parametrize("backend", ["matplotlib", "altair"])
def test_voronoi_grid_with_backend(backend):
    """Test VoronoiGrid works with both backends."""

    def voronoi_agent_portrayal(agent):
        return AgentPortrayalStyle(marker="o", color="blue")

    voronoi_model = mesa.Model()
    voronoi_model.grid = mesa.discrete_space.VoronoiGrid(
        centroids_coordinates=[(0, 1), (0, 0), (1, 0)],
        random=random.Random(42),
    )

    renderer = (
        SpaceRenderer(voronoi_model, backend=backend)
        .setup_agents(voronoi_agent_portrayal)
        .render()
    )

    # Should not raise
    solara.render(SolaraViz(voronoi_model, renderer, components=[]))
    assert renderer.backend == backend


# --- AgentPortrayalStyle Tests ---


@pytest.mark.parametrize("backend", ["matplotlib", "altair"])
def test_agent_portrayal_style_attributes(mock_model, backend):
    """Test that AgentPortrayalStyle attributes are correctly used."""

    def custom_portrayal(agent):
        return AgentPortrayalStyle(
            marker="s",  # square
            color="red",
            size=100,
            alpha=0.8,
            zorder=10,
        )

    renderer = (
        SpaceRenderer(mock_model, backend=backend)
        .setup_agents(custom_portrayal)
        .render()
    )

    solara.render(SolaraViz(mock_model, renderer, components=[]))
    assert renderer.backend == backend


@pytest.mark.parametrize("backend", ["matplotlib", "altair"])
def test_conditional_agent_portrayal(mock_model, backend):
    """Test conditional agent portrayal based on agent attributes."""

    def conditional_portrayal(agent):
        # Conditional portrayal based on agent unique_id
        color = "red" if agent.unique_id % 2 == 0 else "blue"
        return AgentPortrayalStyle(marker="o", color=color, size=50)

    renderer = (
        SpaceRenderer(mock_model, backend=backend)
        .setup_agents(conditional_portrayal)
        .render()
    )

    solara.render(SolaraViz(mock_model, renderer, components=[]))
    assert renderer.backend == backend


# --- Different Space Types Tests ---


@pytest.mark.parametrize("backend", ["matplotlib", "altair"])
def test_hex_multigrid_with_backend(backend):
    """Test HexMultiGrid (legacy) rendering works with both backends."""
    model = HexMultiGridModel()
    renderer = (
        SpaceRenderer(model, backend=backend)
        .setup_agents(agent_portrayal)
        .render()
    )

    # Should not raise
    solara.render(SolaraViz(model, renderer, components=[]))
    assert renderer.backend == backend


@pytest.mark.parametrize("backend", ["matplotlib", "altair"])
def test_continuous_space_with_backend(continuous_model, backend):
    """Test ContinuousSpace rendering works with both backends."""
    renderer = (
        SpaceRenderer(continuous_model, backend=backend)
        .setup_agents(agent_portrayal)
        .render()
    )

    # Should not raise
    solara.render(SolaraViz(continuous_model, renderer, components=[]))
    assert renderer.backend == backend


@pytest.mark.parametrize("backend", ["matplotlib", "altair"])
def test_voronoi_with_backend_fixture(voronoi_model, backend):
    """Test VoronoiGrid rendering using fixture with both backends."""
    renderer = (
        SpaceRenderer(voronoi_model, backend=backend)
        .setup_agents(agent_portrayal)
        .render()
    )

    # Should not raise
    solara.render(SolaraViz(voronoi_model, renderer, components=[]))
    assert renderer.backend == backend


# --- Error Handling Tests ---


def test_invalid_backend_raises():
    """Test that invalid backend raises ValueError."""
    model = MockModel()
    with pytest.raises(ValueError, match="Unsupported backend"):
        SpaceRenderer(model, backend="invalid_backend")


def test_no_space_raises():
    """Test that model without space raises ValueError."""

    class NoSpaceModel(mesa.Model):
        def __init__(self):
            super().__init__()
            # No grid or space attribute

    model = NoSpaceModel()
    with pytest.raises(ValueError, match="Unsupported space type"):
        SpaceRenderer(model)


# --- Multiple Agents Tests ---


@pytest.mark.parametrize("backend", ["matplotlib", "altair"])
def test_multiple_agents_rendering(backend):
    """Test rendering model with multiple agents."""

    class MultiAgentModel(mesa.Model):
        def __init__(self):
            super().__init__()
            self.grid = MultiGrid(width=10, height=10, torus=True)
            # Add multiple agents
            for i in range(5):
                a = MockAgent(self)
                self.grid.place_agent(a, (i, i))

    model = MultiAgentModel()

    def multi_agent_portrayal(agent):
        return AgentPortrayalStyle(
            marker="o", color="blue", size=30 + agent.unique_id * 10
        )

    renderer = (
        SpaceRenderer(model, backend=backend)
        .setup_agents(multi_agent_portrayal)
        .render()
    )

    solara.render(SolaraViz(model, renderer, components=[]))
    assert renderer.backend == backend


# --- Slider and Model Param Tests ---


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
