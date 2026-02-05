"""Test Solara visualizations with Scenarios."""

import unittest

import solara

import mesa
from mesa.examples.basic.boltzmann_wealth_model.model import (
    BoltzmannScenario,
    BoltzmannWealth,
)
from mesa.experimental.scenarios import Scenario
from mesa.visualization.solara_viz import (
    Slider,
    SolaraViz,
)


class MyScenario(Scenario):
    """A mock scenario for testing."""

    density: float = 0.7
    vision: int = 7


class MyModel(mesa.Model):
    """A mock model for testing."""

    def __init__(self, height=40, width=40, scenario: MyScenario | None = None):
        """Initialize the mock model."""
        super().__init__(scenario=scenario)
        self.height = height
        self.width = width


class TestSolaraVizScenarios(unittest.TestCase):
    """Test suite for SolaraViz Scenario support."""

    def test_mixed_params_rendering(self):
        """Test that mixing model and scenario parameters renders correctly."""
        model = MyModel()
        model_params = {
            "height": 50,
            "density": Slider("Density", 0.8, 0.1, 1.0, 0.1),  # Scenario param
            "width": Slider("Width", 40, 10, 100, 10),  # Model param
        }

        # Check if it renders without error
        solara.render(SolaraViz(model, model_params=model_params), handle_error=False)

    def test_scenario_subclass_with_type_hints(self):
        """Test that scenario subclasses with type hints work correctly."""

        class TypedScenario(Scenario):
            agent_density: float = 0.8
            agent_vision: int = 7
            movement_enabled: bool = True
            speed: float = 1.0

        class TypedModel(mesa.Model):
            def __init__(self, grid_size=10, scenario: TypedScenario | None = None):
                super().__init__(scenario=scenario)
                self.grid_size = grid_size

        scenario = TypedScenario(agent_density=0.5, agent_vision=10)
        model = TypedModel(scenario=scenario)

        model_params = {
            "grid_size": 20,
            "agent_density": Slider("Density", 0.7, 0.0, 1.0, 0.1),
            "movement_enabled": Slider("Movement", False, True, True, True),
        }

        # Should render without error
        solara.render(SolaraViz(model, model_params=model_params), handle_error=False)

        # Verify type hints are preserved
        self.assertIsInstance(scenario.agent_density, float)
        self.assertIsInstance(scenario.agent_vision, int)
        self.assertIsInstance(scenario.movement_enabled, bool)

    def test_empty_scenario_params(self):
        """Test handling of empty scenario parameters."""

        class EmptyScenario(Scenario):
            pass

        class ModelWithEmptyScenario(mesa.Model):
            def __init__(self, height=20, scenario: EmptyScenario | None = None):
                super().__init__(scenario=scenario)
                self.height = height

        scenario = EmptyScenario()
        model = ModelWithEmptyScenario(scenario=scenario)

        model_params = {
            "height": 25,
        }

        # Should work without errors
        solara.render(SolaraViz(model, model_params=model_params), handle_error=False)

    def test_scenario_with_defaults(self):
        """Test scenario with default values."""

        class ScenarioWithDefaults(Scenario):
            density: float = 0.5
            vision: int = 5
            speed: float = 1.0

        class ModelWithDefaults(mesa.Model):
            def __init__(self, width=10, scenario: ScenarioWithDefaults | None = None):
                super().__init__(scenario=scenario)
                self.width = width

        # Test with default scenario values
        scenario = ScenarioWithDefaults()
        ModelWithDefaults(scenario=scenario)

        self.assertEqual(scenario.density, 0.5)
        self.assertEqual(scenario.vision, 5)
        self.assertEqual(scenario.speed, 1.0)

        # Test with overridden values
        scenario = ScenarioWithDefaults(density=0.8, vision=10)
        ModelWithDefaults(scenario=scenario)

        self.assertEqual(scenario.density, 0.8)
        self.assertEqual(scenario.vision, 10)
        self.assertEqual(scenario.speed, 1.0)  # Still default

    def test_reset_with_scenario(self):
        """Test that resetting the model correctly reconstructs the scenario."""
        # This test would require more complex setup with actual Solara interaction
        # For now, we test the core logic separately in test_parameter_splitting_logic

    def test_boltzmann_scenario_integration(self):
        """Integration test for Boltzmann Wealth model with scenario params in SolaraViz."""
        scenario = BoltzmannScenario(n=50, width=10, height=10)
        model = BoltzmannWealth(scenario=scenario)

        model_params = {
            "n": Slider("Agents", 60, 10, 100, 1),
            "width": Slider("Width", 12, 5, 20, 1),
            "height": Slider("Height", 11, 5, 20, 1),
            "rng": 42,
        }

        solara.render(SolaraViz(model, model_params=model_params), handle_error=False)

    def test_boltzmann_scenario_integration(self):
        """Integration test for Boltzmann Wealth model with scenario params in SolaraViz."""
        scenario = BoltzmannScenario(n=50, width=10, height=10)
        model = BoltzmannWealth(scenario=scenario)

        model_params = {
            "n": Slider("Agents", 60, 10, 100, 1),
            "width": Slider("Width", 12, 5, 20, 1),
            "height": Slider("Height", 11, 5, 20, 1),
            "rng": 42,
        }

        solara.render(SolaraViz(model, model_params=model_params), handle_error=False)


if __name__ == "__main__":
    unittest.main()
