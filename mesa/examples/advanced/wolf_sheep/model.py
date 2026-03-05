"""
Wolf-Sheep Predation Model
================================

Replication of the model found in NetLogo:
    Wilensky, U. (1997). NetLogo Wolf Sheep Predation model.
    http://ccl.northwestern.edu/netlogo/models/WolfSheepPredation.
    Center for Connected Learning and Computer-Based Modeling,
    Northwestern University, Evanston, IL.
"""

import math

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.discrete_space import OrthogonalVonNeumannGrid
from mesa.examples.advanced.wolf_sheep.agents import GrassPatch, Sheep, Wolf
from mesa.experimental.scenarios import Scenario


class WolfSheepScenario(Scenario):
    """Scenario parameters for the Wolf-Sheep model.

    Args:
        height: Height of the grid
        width: Width of the grid
        initial_sheep: Number of sheep to start with
        initial_wolves: Number of wolves to start with
        sheep_reproduce: Probability of each sheep reproducing each step
        wolf_reproduce: Probability of each wolf reproducing each step
        wolf_gain_from_food: Energy a wolf gains from eating a sheep
        grass: Whether to have the sheep eat grass for energy
        grass_regrowth_time: How long it takes for a grass patch to regrow
                            once it is eaten
        sheep_gain_from_food: Energy sheep gain from grass, if enabled
        rng: Random rng
    """

    width: int = 20
    height: int = 20
    initial_sheep: int = 100
    initial_wolves: int = 50
    sheep_reproduce: float = 0.04
    wolf_reproduce: float = 0.05
    wolf_gain_from_food: float = 20.0
    grass: bool = True
    grass_regrowth_time: int = 30
    sheep_gain_from_food: float = 4.0


class WolfSheep(Model):
    """Wolf-Sheep Predation Model.

    A model for simulating wolf and sheep (predator-prey) ecosystem modelling.
    """

    description = (
        "A model for simulating wolf and sheep (predator-prey) ecosystem modelling."
    )

    def __init__(self, scenario=None):
        """Create a new Wolf-Sheep model with the given parameters.

        Args:
            scenario: WolfSheepScenario containing model parameters.
        """
        if scenario is None:
            scenario = WolfSheepScenario()

        super().__init__(scenario=scenario)

        # Initialize model parameters
        self.height = scenario.height
        self.width = scenario.width
        self.grass = scenario.grass

        # Create grid using experimental cell space
        self.grid = OrthogonalVonNeumannGrid(
            [self.height, self.width],
            torus=True,
            capacity=math.inf,
            random=self.random,
        )

        # Set up data collection
        model_reporters = {
            "Wolves": lambda m: len(m.agents_by_type[Wolf]),
            "Sheep": lambda m: len(m.agents_by_type[Sheep]),
        }
        if self.grass:
            model_reporters["Grass"] = lambda m: len(
                m.agents_by_type[GrassPatch].select(lambda a: a.fully_grown)
            )

        self.datacollector = DataCollector(model_reporters)

        # Create sheep:
        Sheep.create_agents(
            self,
            scenario.initial_sheep,
            energy=self.rng.random((scenario.initial_sheep,))
            * 2
            * scenario.sheep_gain_from_food,
            p_reproduce=scenario.sheep_reproduce,
            energy_from_food=scenario.sheep_gain_from_food,
            cell=self.random.choices(
                self.grid.all_cells.cells, k=scenario.initial_sheep
            ),
        )
        # Create Wolves:
        Wolf.create_agents(
            self,
            scenario.initial_wolves,
            energy=self.rng.random((scenario.initial_wolves,))
            * 2
            * scenario.wolf_gain_from_food,
            p_reproduce=scenario.wolf_reproduce,
            energy_from_food=scenario.wolf_gain_from_food,
            cell=self.random.choices(
                self.grid.all_cells.cells, k=scenario.initial_wolves
            ),
        )

        # Create grass patches if enabled
        if self.grass:
            possibly_fully_grown = [True, False]
            for cell in self.grid:
                fully_grown = self.random.choice(possibly_fully_grown)
                countdown = (
                    0
                    if fully_grown
                    else self.random.randrange(0, scenario.grass_regrowth_time)
                )
                GrassPatch(self, countdown, scenario.grass_regrowth_time, cell)

        # Collect initial data
        self.running = True
        self.datacollector.collect(self)

    def step(self):
        """Execute one step of the model."""
        # First activate all sheep, then all wolves, both in random order
        self.agents_by_type[Sheep].shuffle_do("step")
        self.agents_by_type[Wolf].shuffle_do("step")

        # Collect data
        self.datacollector.collect(self)
