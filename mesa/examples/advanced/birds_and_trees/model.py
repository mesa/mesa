"""
Birds and Trees Model
=====================

Birds fly through a continuous air space and eat fruit from trees on a grid below.
The air and the ground share coordinates, and each bird's ground cell follows its
position in the air, using ``mesa.experimental.locations``.
"""

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.discrete_space import OrthogonalMooreGrid
from mesa.examples.advanced.birds_and_trees.agents import Bird, Tree
from mesa.experimental.continuous_space import ContinuousSpace
from mesa.experimental.scenarios import Scenario


class BirdsAndTreesScenario(Scenario):
    """Scenario parameters for the Birds and Trees model.

    Args:
        size: Width and height of the air, and of the ground below it
        cell_size: Width and height of a ground cell, in the same units
        tree_density: Fraction of ground cells with a tree
        initial_birds: Number of birds to start with
        speed: Distance a bird flies each step
        vision: How many ground cells away a bird sees fruit
        fruit_regrowth: Chance each step that a tree without fruit grows one
        fruit_energy: Energy a bird gains from a fruit
        reproduce_energy: Energy at which a bird reproduces, splitting its energy
        rng: Random seed
    """

    size: float = 100.0
    cell_size: float = 4.0
    tree_density: float = 0.2
    initial_birds: int = 50
    speed: float = 3.0
    vision: int = 1
    fruit_regrowth: float = 0.05
    fruit_energy: float = 10.0
    reproduce_energy: float = 20.0


class BirdsAndTrees(Model):
    """Birds in the air foraging on trees on the ground below."""

    def __init__(self, scenario: BirdsAndTreesScenario = BirdsAndTreesScenario):
        super().__init__(scenario=scenario)
        size = scenario.size
        n_cells = round(size / scenario.cell_size)

        # Two spaces covering the same area, in the same coordinates
        self.air = ContinuousSpace(
            [[0, size], [0, size]], torus=True, random=self.random
        )
        self.ground = OrthogonalMooreGrid(
            (n_cells, n_cells),
            cell_size=size / n_cells,
            torus=True,
            random=self.random,
        )
        self.grid = self.ground  # the space SpaceRenderer draws

        for cell in self.ground.all_cells:
            if self.random.random() < scenario.tree_density:
                Tree(self, cell)

        for _ in range(scenario.initial_birds):
            position = self.rng.random(2) * size
            energy = self.random.uniform(1, scenario.reproduce_energy)
            Bird(self, position, energy)

        self.datacollector = DataCollector(
            {
                "Birds": lambda m: len(m.agents_by_type.get(Bird, [])),
                "Fruit": lambda m: len(
                    m.agents_by_type[Tree].select(lambda tree: tree.has_fruit)
                ),
            }
        )
        self.datacollector.collect(self)

    def step(self):
        """Grow fruit, then let all birds act in random order."""
        self.agents_by_type[Tree].do("step")
        if Bird in self.agents_by_type:
            self.agents_by_type[Bird].shuffle_do("step")
        self.datacollector.collect(self)
