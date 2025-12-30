from agents import MultiSpaceAgent

from mesa import Model
from mesa.discrete_space import HexGrid, OrthogonalMooreGrid
from mesa.experimental.continuous_space import ContinuousSpace


class MultiSpaceModel(Model):
    def __init__(self, n_agents=5):
        super().__init__()

        # Create aligned spaces (all use same coordinate system)
        self.continuous = ContinuousSpace([[0, 10], [0, 10]], random=self.random)
        self.grid = OrthogonalMooreGrid([10, 10], random=self.random)
        self.hex_grid = HexGrid([10, 10], random=self.random)

        # Create agents
        for _ in range(n_agents):
            pos = [self.random.uniform(0, 10), self.random.uniform(0, 10)]
            MultiSpaceAgent(self, pos)

    def step(self):
        self.agents.shuffle_do("step")


model = MultiSpaceModel(n_agents=2)
for i in range(5):
    print(f"\n=== Step {i} ===")
    model.step()
