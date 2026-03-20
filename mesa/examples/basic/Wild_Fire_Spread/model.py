from mesa.model import Model
from mesa.datacollection import DataCollector
import random
from mesa.space import MultiGrid

from examples.basic.Wild_Fire_Spread.agent import FuelAgent,AgentState


class ForestFireModel(Model):
    def __init__(self, width=50, height=50):
        super().__init__()

        self.grid = MultiGrid(width, height, torus=False)

        self.wind_speed = random.uniform(0, 1)
        self.humidity = random.uniform(0, 100)

        # Create agents
        for _, (x, y) in self.grid.coord_iter():
            agent = FuelAgent(self)
            self.grid.place_agent(agent, (x, y))

        # Ignite random cells
        for agent in random.sample(list(self.agents), min(3, len(self.agents))):
            agent.state = AgentState.BURNING
            agent.burn_time = random.randint(3, 6)

        self.datacollector = DataCollector(

            {
                "Healthy": lambda m: sum(
                    a.state == AgentState.HEALTHY for a in m.agents
                ),
                "Burning": lambda m: sum(
                    a.state == AgentState.BURNING for a in m.agents
                ),
                "Burned": lambda m: sum(a.state == AgentState.BURNED for a in m.agents),
            }
        )

        self.running = True

    def step(self):
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)
