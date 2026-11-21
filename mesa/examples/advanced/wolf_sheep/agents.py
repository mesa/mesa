import math

from mesa import Agent
from mesa.discrete_space import FixedAgent
from mesa.experimental.locations import CellLocation, ContinuousLocation


class Animal(Agent):
    """The base animal class."""

    position = ContinuousLocation("meadow")  # a point in model.meadow
    cell = CellLocation("grid", follows=position)  # the grass patch below it

    def __init__(
        self, model, energy=8, p_reproduce=0.04, energy_from_food=4, position=None
    ):
        """Initialize an animal.

        Args:
            model: Model instance
            energy: Starting amount of energy
            p_reproduce: Probability of reproduction (asexual)
            energy_from_food: Energy obtained from 1 unit of food
            position: Point in the meadow at which the animal starts
        """
        super().__init__(model)
        self.energy = energy
        self.p_reproduce = p_reproduce
        self.energy_from_food = energy_from_food
        self.heading = self.random.uniform(0, 2 * math.pi)
        self.position = position

    def spawn_offspring(self):
        """Create offspring by splitting energy and creating new instance."""
        self.energy /= 2
        self.__class__(
            self.model,
            self.energy,
            self.p_reproduce,
            self.energy_from_food,
            self.position,
        )

    def move(self):
        """Turn a little, as NetLogo's ``rt random 50 lt random 50``, and step forward."""
        self.heading += math.radians(
            self.random.randrange(50) - self.random.randrange(50)
        )
        self.position += [math.cos(self.heading), math.sin(self.heading)]

    def feed(self):
        """Abstract method to be implemented by subclasses."""

    def step(self):
        """Execute one step of the animal's behavior."""
        self.move()

        self.energy -= 1

        # Try to feed
        self.feed()

        # Handle death and reproduction
        if self.energy < 0:
            self.remove()
        elif self.random.random() < self.p_reproduce:
            self.spawn_offspring()


class Sheep(Animal):
    """A sheep that walks around, reproduces (asexually) and gets eaten."""

    def feed(self):
        """If possible, eat grass at current location."""
        grass_patch = next(
            (obj for obj in self.cell.agents if isinstance(obj, GrassPatch)), None
        )
        if grass_patch is not None and grass_patch.fully_grown:
            self.energy += self.energy_from_food
            grass_patch.get_eaten()


class Wolf(Animal):
    """A wolf that walks around, reproduces (asexually) and eats sheep."""

    def feed(self):
        """If possible, eat a sheep on the same patch, as NetLogo's ``sheep-here``."""
        sheep = [obj for obj in self.cell.agents if isinstance(obj, Sheep)]
        if sheep:  # If there are any sheep present
            sheep_to_eat = self.random.choice(sheep)
            self.energy += self.energy_from_food
            sheep_to_eat.remove()


class GrassPatch(FixedAgent):
    """A patch of grass that grows at a fixed rate and can be eaten by sheep."""

    def __init__(self, model, countdown, grass_regrowth_time, cell):
        """Create a new patch of grass.

        Args:
            model: Model instance
            countdown: Time until grass is fully grown again
            grass_regrowth_time: Time needed to regrow after being eaten
            cell: Cell to which this grass patch belongs
        """
        super().__init__(model)
        self.fully_grown = countdown == 0
        self.grass_regrowth_time = grass_regrowth_time
        self.cell = cell

        # Schedule initial growth if not fully grown
        if not self.fully_grown:
            self.model.schedule_event(self.regrow, after=countdown)

    def regrow(self):
        """Regrow the grass."""
        self.fully_grown = True

    def get_eaten(self):
        """Mark grass as eaten and schedule regrowth."""
        self.fully_grown = False
        self.model.schedule_event(self.regrow, after=self.grass_regrowth_time)
