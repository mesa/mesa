from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import MultiGrid
from mesa.discrete_space import CellAgent, OrthogonalMooreGrid

class IdeologyDiffusionModel(Model):
    """A model for simulating the diffusion of political ideologies among individuals."""

    def __init__(
            self, 
            width: int, 
            height: int, 
            initial_population: int,
            economic_crisis: bool = False,
            unemployment_increase: float = 0.0,
            media_influence: bool = False,
            government_repression: bool = False,
            ):
        """
        Args:
            width: Width of the grid.
            height: Height of the grid.
            initial_population: Initial number of individual agents.
            economic_crisis: Whether an economic crisis is occurring.
            unemployment_increase: Increase in unemployment rate.
            media_influence: Whether media influence is active.
            government_repression: Whether government repression is active.
        """

        super().__init__()
        self.grid = OrthogonalMooreGrid(width, height, torus=True)
        self.initial_population = initial_population
        self.economic_crisis = economic_crisis
        self.unemployment_increase = unemployment_increase
        self.media_influence = media_influence
        self.government_repression = government_repression
        
      