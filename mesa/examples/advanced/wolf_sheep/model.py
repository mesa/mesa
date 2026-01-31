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
from mesa.experimental.devs import ABMSimulator


class WolfSheep(Model):
    """Wolf-Sheep Predation Model.

    A model for simulating wolf and sheep (predator-prey) ecosystem modelling.
    """

    description = (
        "A model for simulating wolf and sheep (predator-prey) ecosystem modelling."
    )

    def __init__(
        self,
        width=20,
        height=20,
        initial_sheep=100,
        initial_wolves=50,
        sheep_reproduce=0.04,
        wolf_reproduce=0.05,
        wolf_gain_from_food=20,
        grass=True,
        grass_regrowth_time=30,
        sheep_gain_from_food=4,
        rng=None,
        simulator: ABMSimulator = None,
    ):
        """Create a new Wolf-Sheep model with the given parameters.

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
            simulator: ABMSimulator instance for event scheduling
        """
        super().__init__(rng=rng)
        self.simulator = simulator
        self.simulator.setup(self)

        # Initialize model parameters
        self.height = height
        self.width = width
        self.grass = grass

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
        if grass:
            model_reporters["Grass"] = lambda m: len(
                m.agents_by_type[GrassPatch].select(lambda a: a.fully_grown)
            )

        self.datacollector = DataCollector(model_reporters)

        # Create sheep:
        Sheep.create_agents(
            self,
            initial_sheep,
            energy=self.rng.random((initial_sheep,)) * 2 * sheep_gain_from_food,
            p_reproduce=sheep_reproduce,
            energy_from_food=sheep_gain_from_food,
            cell=self.random.choices(self.grid.all_cells.cells, k=initial_sheep),
        )
        # Create Wolves:
        Wolf.create_agents(
            self,
            initial_wolves,
            energy=self.rng.random((initial_wolves,)) * 2 * wolf_gain_from_food,
            p_reproduce=wolf_reproduce,
            energy_from_food=wolf_gain_from_food,
            cell=self.random.choices(self.grid.all_cells.cells, k=initial_wolves),
        )

        # Create grass patches if enabled
        if grass:
            possibly_fully_grown = [True, False]
            for cell in self.grid:
                fully_grown = self.random.choice(possibly_fully_grown)
                countdown = (
                    0 if fully_grown else self.random.randrange(0, grass_regrowth_time)
                )
                GrassPatch(self, countdown, grass_regrowth_time, cell)

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
