
"""Tests for DataCollector index alignment regression."""

import os
import sys

sys.path.insert(0, os.getcwd())

import unittest

import mesa


class TestDataCollectorIndex(unittest.TestCase):
    """Regression tests for DataCollector index consistency."""
    def test_model_and_agent_dataframe_alignment(self):
        """Regression test for inconsistent DataCollector indexes.

        Verifies that when collect() is called after stepping, the model DataFrame
        index matches the agent DataFrame 'Step' index.
        """

        class MockAgent(mesa.Agent):
            def __init__(self, model):
                super().__init__(model)
                self.wealth = 1

            def step(self):
                self.wealth += 1

        class MockModel(mesa.Model):
            def __init__(self):
                super().__init__()
                # Auto-registration handles agents list
                for _ in range(5):
                    MockAgent(self)

                self.datacollector = mesa.DataCollector(
                    model_reporters={"ModelStep": lambda m: m.steps},
                    agent_reporters={"Wealth": "wealth"}
                )

            def step(self):
                # Crucial: Collect AFTER step to trigger the misalignment bug
                # Model.step() is wrapped and handles time/steps increment automatically
                self.datacollector.collect(self)
                self.agents.do("step")

        model = MockModel()

        # Step the model a few times
        for _ in range(3):
            model.step()

        model_df = model.datacollector.get_model_vars_dataframe()
        agent_df = model.datacollector.get_agent_vars_dataframe()

        # Get unique steps from agent DataFrame
        if "Step" in agent_df.index.names:
            agent_steps = agent_df.index.get_level_values("Step").unique()
        else:
            agent_steps = agent_df["Step"].unique()

        # Assertion: Model DF index should match Agent DF step index
        # Using list comparison for strict equality of values
        self.assertEqual(
            list(model_df.index),
            list(agent_steps),
            "Model DataFrame index does not match Agent DataFrame Step index"
        )

if __name__ == "__main__":
    unittest.main()
