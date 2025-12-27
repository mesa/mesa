from mesa import Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from agents import Person


class IdeologyModel(Model):
    def __init__(
        self,
        n=120,
        width=15,
        height=15,
        economic_crisis=0.5,
        propaganda=0.2,
        seed=None,
    ):
        super().__init__(seed=seed)

        self.grid = MultiGrid(width, height, torus=True)
        self.schedule = RandomActivation(self)

        self.economic_crisis = economic_crisis
        self.propaganda = propaganda

        for _ in range(n):
            agent = Person(self)
            self.schedule.add(agent)

            x = self.random.randrange(width)
            y = self.random.randrange(height)
            self.grid.place_agent(agent, (x, y))

        self.datacollector = DataCollector(
            {
                "Neutrals": lambda m: sum(
                    a.opinion == "neutral" for a in m.schedule.agents
                ),
                "Moderates": lambda m: sum(
                    a.opinion == "moderate" for a in m.schedule.agents
                ),
                "Radicals": lambda m: sum(
                    a.opinion == "radical" for a in m.schedule.agents
                ),
            }
        )

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
