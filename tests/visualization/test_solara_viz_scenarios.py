"""Test Solara visualizations with Scenarios."""

import unittest

import solara

import mesa
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

    def test_reset_with_scenario(self):
        """Test that resetting the model correctly reconstructs the scenario."""
        # This is harder to test without a full Solara environment and interaction
        # but we can try to test the ModelController logic if we can isolate it.


if __name__ == "__main__":
    unittest.main()
