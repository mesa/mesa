from mesa import Model
from mesa.discrete_space import OrthogonalMooreGrid
from mesa.examples.basic.schelling.agents import SchellingAgent
from mesa.experimental.data_collection import DataRecorder, DatasetConfig


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

        # Track happiness
        self.happy = 0

        self.data_registry.track_model(
            self,
            "model_data",
            fields=["happy", "pct_happy", "population", "minority_pct"],
        )

        # Track Agent Variables
        self.data_registry.track_agents(
            self.agents,
            "agent_data",
            fields=["type"],
        )

        # Create agents and place them on the grid
        for cell in self.grid.all_cells:
            if self.random.random() < self.density:
                agent_type = 1 if self.random.random() < minority_pc else 0
                SchellingAgent(
                    self, cell, agent_type, homophily=homophily, radius=radius
                )

        self.agents.do("assign_state")
        self.datarecorder = DataRecorder(
            self,
            config={
                "model_data": DatasetConfig(interval=1),
                "agent_data": DatasetConfig(interval=1),
            },
        )

    @property
    def population(self):
        return len(self.agents)

    @property
    def pct_happy(self):
        if self.population > 0:
            return (self.happy / self.population) * 100
        return 0

    @property
    def minority_pct(self):
        if self.population > 0:
            count = sum(1 for agent in self.agents if agent.type == 1)
            return (count / self.population) * 100
        return 0

    def step(self):
        """Run one step of the model."""
        self.happy = 0  # Reset counter of happy agents
        self.agents.shuffle_do("step")  # Activate all agents in random order
        self.agents.do("assign_state")
        self.running = self.happy < len(self.agents)  # Continue until everyone is happy
