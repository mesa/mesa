
import mesa
from mesa import Agent
from mesa.model import Model

from mesa.discrete_space import OrthogonalMooreGrid
from mesa.datacollection import DataCollector
import random

from agent import FuelAgent, AgentState


class ForestFireModel(Model):

    def __init__(self, width=50, height=50):

        super().__init__()

        self.grid = OrthogonalMooreGrid(width, height, torus=False)

        self.wind_speed = random.uniform(0, 1)
        self.humidity = random.uniform(0, 100)

        # Create agents
        for (x, y) in self.grid.coord_iter():
            agent = FuelAgent(self)
            self.grid.place_agent(agent, (x, y))
            self.schedule.add(agent)

        # Ignite random cells
        for agent in random.sample(self.schedule.agents, 3):
            agent.state = AgentState.BURNING
            agent.burn_time = random.randint(3, 6)

        self.datacollector = DataCollector({
            "Burning": lambda m: sum(a.state == AgentState.BURNING for a in m.schedule.agents),
            "Burned": lambda m: sum(a.state == AgentState.BURNED for a in m.schedule.agents),
        })

        self.running = True

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)