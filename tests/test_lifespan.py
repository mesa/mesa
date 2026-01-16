"""Test removal of agents."""

import unittest

import numpy as np

from mesa import Agent, Model
from mesa.datacollection import DataCollector


class LifeTimeModel(Model):
    """Simple model for running models with a finite life."""

    def __init__(self, agent_lifetime=1, n_agents=10, seed=None):  # noqa: D107
        super().__init__(seed=seed)

        self.agent_lifetime = agent_lifetime
        self.n_agents = n_agents

        # keep track of the the remaining life of an agent and
        # how many ticks it has seen
        self.datacollector = DataCollector(
            agent_reporters={
                "remaining_life": lambda a: a.remaining_life,
                "time": lambda a: a.time,
            }
        )

        self.current_ID = 0

        for _ in range(n_agents):
            FiniteLifeAgent(self.agent_lifetime, self)

        self.datacollector.collect(self)

    def step(self):
        """Add agents back to n_agents in each step."""
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)

        if len(self.agents) < self.n_agents:
            for _ in range(self.n_agents - len(self.agents)):
                FiniteLifeAgent(self.agent_lifetime, self)

    def run_model(self, step_count=100):  # noqa: D102
        for _ in range(step_count):
            self.step()


class FiniteLifeAgent(Agent):
    """An agent that is supposed to live for a finite number of ticks.

    Also has a 10% chance of dying in each tick.
    """

    def __init__(self, lifetime, model):  # noqa: D107
        super().__init__(model)
        self.remaining_life = lifetime
        self.time = 0
        self.model = model

    def step(self):  # noqa: D102
        deactivated = self.deactivate()
        if not deactivated:
            self.time += 1  # keep track of how many ticks are seen
            if np.random.binomial(1, 0.1) != 0:  # 10% chance of dying
                self.remove()

    def deactivate(self):  # noqa: D102
        self.remaining_life -= 1
        if self.remaining_life < 0:
            self.remove()
            return True
        return False


class TestAgentLifespan(unittest.TestCase):  # noqa: D101
    def setUp(self):  # noqa: D102
        self.model = LifeTimeModel()
        self.model.run_model()
        self.df = self.model.datacollector.get_agent_vars_dataframe()
        self.df = self.df.reset_index()

    def test_ticks_seen(self):
        """Each agent should be activated no more than one time."""
        assert self.df.time.max() == 1

    def test_agent_lifetime(self):  # noqa: D102
        lifetimes = self.df.groupby(["AgentID"]).agg({"Step": len})
        assert lifetimes.Step.max() == 2


if __name__ == "__main__":
    unittest.main()
