from mesa import Model
from mesa.datacollection import DataCollector
from mesa.discrete_space import OrthogonalMooreGrid
from mesa.examples.basic.schelling.agents import SchellingAgent
from mesa.experimental.data_collection.dataset import DataRegistry, NumpyAgentDataSet

import numpy as np

class Schelling(Model):
    """Model class for the Schelling segregation model."""

    def __init__(
        self,
        height: int = 20,
        width: int = 20,
        density: float = 0.8,
        minority_pc: float = 0.5,
        homophily: float = 0.4,
        radius: int = 1,
        rng=None,
    ):
        """Create a new Schelling model.

        Args:
            width: Width of the grid
            height: Height of the grid
            density: Initial chance for a cell to be populated (0-1)
            minority_pc: Chance for an agent to be in minority class (0-1)
            homophily: Minimum number of similar neighbors needed for happiness
            radius: Search radius for checking neighbor similarity
            rng: Seed for reproducibility
        """
        super().__init__(rng=rng)

        # Model parameters
        self.density = density
        self.minority_pc = minority_pc

        # Initialize grid
        self.grid = OrthogonalMooreGrid((width, height), random=self.random, capacity=1)

        # Set up data collection
        self.data_registry = DataRegistry()
        self.data_registry.track_model(self, "model_data", "happy", "pct_happy")
        self.agents_happy = self.data_registry.create_dataset(NumpyAgentDataSet, "happy", SchellingAgent, "happy", dtype=bool)
        self.agents_type= self.data_registry.create_dataset(NumpyAgentDataSet, "agent_data", SchellingAgent, "agent_type", dtype=int)

        # Create agents and place them on the grid
        for cell in self.grid.all_cells:
            if self.random.random() < self.density:
                agent_type = 1 if self.random.random() < minority_pc else 0
                SchellingAgent(
                    self, cell, agent_type, homophily=homophily, radius=radius
                )

        # Collect initial state
        self.agents.do("assign_state")

    @property
    def happy(self):
        return np.sum(self.agents_happy.data)

    @property
    def pct_happy(self):
        data = self.agents_happy.data
        return np.sum(data)/data.shape[0]

    def step(self):
        """Run one step of the model."""
        self.agents.shuffle_do("step")  # Activate all agents in random order
        self.agents.do("assign_state")
        self.running = self.happy < len(self.agents)  # Continue until everyone is happy
