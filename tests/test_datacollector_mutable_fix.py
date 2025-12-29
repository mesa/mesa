"""Test DataCollector correctly deep-copies mutable agent data."""

import unittest

from mesa import Agent, Model
from mesa.datacollection import DataCollector


class MutableAgent(Agent):
    """Agent with mutable list attribute."""

    def __init__(self, model):
        """Initialize agent with empty data list."""
        super().__init__(model)
        self.data = []


class MutableModel(Model):
    """Model that modifies agent data after collection."""

    def __init__(self):
        """Initialize model with agent and datacollector."""
        super().__init__()
        self.agent = MutableAgent(self)
        self.datacollector = DataCollector(agent_reporters={"Data": "data"})

    def step(self):
        """Collect data then modify agent data."""
        self.datacollector.collect(self)
        self.agent.data.append(self.steps)  # Modify after collection


class TestDataCollectorMutableFix(unittest.TestCase):
    """Test that mutable data is deep-copied."""

    def test_mutable_data_independence(self):
        """Historical records should not change when agent modifies data."""
        model = MutableModel()

        model.step()  # Step 1: collect [], then add 1
        model.step()  # Step 2: collect [1], then add 2
        model.step()  # Step 3: collect [1, 2], then add 3

        df = model.datacollector.get_agent_vars_dataframe()

        # Each step should preserve its historical state
        self.assertEqual(df.loc[(1, 1), "Data"], [])
        self.assertEqual(df.loc[(2, 1), "Data"], [1])
        self.assertEqual(df.loc[(3, 1), "Data"], [1, 2])

