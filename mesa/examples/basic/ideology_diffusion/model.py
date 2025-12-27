from mesa import Model
from mesa.datacollection import DataCollector
from mesa.discrete_space import OrthogonalMooreGrid
from agents import IndividualAgent

class IdeologyDiffusionModel(Model):
    def __init__(
        self, 
        width: int = 20, 
        height: int = 20, 
        economic_crisis: bool = False,
        unemployment_increase: float = 0.2,
        media_influence: bool = False,
        government_repression: bool = False
    ):
        super().__init__()
        self.grid = OrthogonalMooreGrid(width, height, torus=True)
        self.economic_crisis = economic_crisis
        self.unemployment_increase = unemployment_increase
        self.media_influence = media_influence
        self.government_repression = government_repression

        # Setup agents with randomized initial traits
        for cell in self.grid.coord_iter():
            agent = IndividualAgent(
                model=self,
                cell=cell[0],
                economic_dissatisfaction=self.random.random(),
                propaganda_susceptibility=self.random.random(),
                resistance_to_change=self.random.random(),
                political_ideology=0
            )
            self.grid.place_agent(agent, cell[1])

        self.datacollector = DataCollector(
            model_reporters={
                "Neutral": lambda m: self.count_ideology(m, 0),
                "Moderate": lambda m: self.count_ideology(m, 1),
                "Radical": lambda m: self.count_ideology(m, 2),
            }
        )

    @staticmethod
    def count_ideology(model, level):
        return sum(1 for a in model.agents if a.political_ideology == level)

    def step(self):
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)