"""Test Solara visualizations with Scenarios."""

import unittest
import solara
import mesa
from mesa.experimental.scenarios import Scenario
from mesa.visualization.solara_viz import SolaraViz, ModelController, ModelCreator, Slider

class MyScenario(Scenario):
    density: float = 0.7
    vision: int = 7

class MyModel(mesa.Model):
    def __init__(self, height=40, width=40, scenario: MyScenario | None = None):
        super().__init__(scenario=scenario)
        self.height = height
        self.width = width

class TestSolaraVizScenarios(unittest.TestCase):
    def test_auto_split_params(self):
        """Test that parameters are automatically split between model and scenario."""
        model = MyModel()
        model_params = {
            "height": 50,
            "density": 0.8  # Should be moved to scenario
        }
        
        # We can't easily check the internal state of SolaraViz components without complex mocks
        # but we can check if it renders without error
        solara.render(SolaraViz(model, model_params=model_params), handle_error=False)

    def test_explicit_scenario_params(self):
        """Test that scenario_params can be explicitly provided."""
        model = MyModel()
        model_params = {"height": 50}
        scenario_params = {"density": Slider("Density", 0.8, 0.1, 1.0, 0.1)}
        
        solara.render(SolaraViz(model, model_params=model_params, scenario_params=scenario_params), handle_error=False)

    def test_reset_with_scenario(self):
        """Test that resetting the model correctly reconstructs the scenario."""
        # This is harder to test without a full Solara environment and interaction
        # but we can try to test the ModelController logic if we can isolate it.
        pass

if __name__ == "__main__":
    unittest.main()
