"""
Boltzmann Wealth Model
=====================

A simple model of wealth distribution based on the Boltzmann-Gibbs distribution.
Agents move randomly on a grid, giving one unit of wealth to a random neighbor
when they occupy the same cell.
"""

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.discrete_space import OrthogonalMooreGrid
from mesa.examples.basic.boltzmann_wealth_model.agents import MoneyAgent


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

        self.num_agents = n
        self.grid = OrthogonalMooreGrid((width, height), random=self.random)

        # Set up data collection
        self.datacollector = DataCollector(
            model_reporters={"Gini": self.compute_gini},
            agent_reporters={"Wealth": "wealth"},
        )
        MoneyAgent.create_agents(
            self,
            self.num_agents,
            self.random.choices(self.grid.all_cells.cells, k=self.num_agents),
        )

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        self.agents.shuffle_do("step")  # Activate all agents in random order
        self.datacollector.collect(self)  # Collect data

    def compute_gini(self):
        """Calculate the Gini coefficient for the model's current wealth distribution.

        The Gini coefficient is a measure of inequality in distributions.
        - A Gini of 0 represents complete equality, where all agents have equal wealth.
        - A Gini of 1 represents maximal inequality, where one agent has all wealth.
        """
        agent_wealths = [agent.wealth for agent in self.agents]
        x = sorted(agent_wealths)
        n = self.num_agents
        # Calculate using the standard formula for Gini coefficient
        b = sum(xi * (n - i) for i, xi in enumerate(x)) / (n * sum(x))
        return 1 + (1 / n) - 2 * b
