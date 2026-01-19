"""
Boltzmann Wealth Model
=====================

A simple model of wealth distribution based on the Boltzmann-Gibbs distribution.
Agents move randomly on a grid, giving one unit of wealth to a random neighbor
when they occupy the same cell.
"""

import numpy as np

from mesa import Model
from mesa.discrete_space import OrthogonalMooreGrid
from mesa.examples.basic.boltzmann_wealth_model.agents import MoneyAgent
from mesa.experimental.statistics import DataRegistry, NumpyAgentDataSet


class BoltzmannWealth(Model):
    """A simple model of an economy where agents exchange currency at random.

    All agents begin with one unit of currency, and each time step agents can give
    a unit of currency to another agent in the same cell. Over time, this produces
    a highly skewed distribution of wealth.

    Attributes:
        num_agents (int): Number of agents in the model
        grid (MultiGrid): The space in which agents move
        running (bool): Whether the model should continue running
        datacollector (DataCollector): Collects and stores model data
    """

    def __init__(self, n=100, width=10, height=10, rng=None):
        """Initialize the model.

        Args:
            n (int, optional): Number of agents. Defaults to 100.
            width (int, optional): Grid width. Defaults to 10.
            height (int, optional): Grid height. Defaults to 10.
            rng (int, optional): Random rng. Defaults to None.
        """
        super().__init__(rng=rng)

        self.data_registry = DataRegistry()
        self.data_registry.create_dataset(NumpyAgentDataSet, "wealth", MoneyAgent,"wealth", n=n)

        self.num_agents = n
        self.grid = OrthogonalMooreGrid((width, height), random=self.random)

        # Set up data collection
        MoneyAgent.create_agents(
            self,
            self.num_agents,
            self.random.choices(self.grid.all_cells.cells, k=self.num_agents),
        )

        self.running = True

    def step(self):
        self.agents.shuffle_do("step")  # Activate all agents in random order
        self.compute_gini()

    def compute_gini(self):
        """Calculate the Gini coefficient for the model's current wealth distribution.

        The Gini coefficient is a measure of inequality in distributions.
        - A Gini of 0 represents complete equality, where all agents have equal wealth.
        - A Gini of 1 represents maximal inequality, where one agent has all wealth.
        """
        agent_wealths = self.data_registry["wealth"].data  # fixme agent_id is currently not included
        sorted_x = np.sort(agent_wealths)
        n = len(agent_wealths)
        cumx = np.cumsum(sorted_x, dtype=float)
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


if __name__ == "__main__":
    model = BoltzmannWealth(125)
    for _ in range(125):
        model.step()
