import math

from mesa import Agent
from mesa.experimental.locations import CellLocation, ContinuousLocation


class Tree(Agent):
    """A tree on the ground that grows fruit."""

    cell = CellLocation()  # a plain cell: trees stay where they are

    def __init__(self, model, cell, has_fruit=True):
        super().__init__(model)
        self.cell = cell
        self.has_fruit = has_fruit

    def step(self):
        if not self.has_fruit and self.random.random() < self.scenario.fruit_regrowth:
            self.has_fruit = True


class Bird(Agent):
    """A bird that flies through the air and eats fruit from the trees below."""

    position = ContinuousLocation("air")  # a point in model.air
    cell = CellLocation("ground", follows="position")  # the ground cell below it

    def __init__(self, model, position, energy):
        super().__init__(model)
        self.position = position  # joins model.air, and the ground cell below
        self.heading = self.random.uniform(0, 2 * math.pi)
        self.energy = energy

    def step(self):
        if not self.forage():
            self.fly()

        self.energy -= 1
        if self.energy <= 0:
            self.remove()  # leaves the air and the ground
        elif self.energy >= self.scenario.reproduce_energy:
            self.energy /= 2
            Bird(self.model, self.position, self.energy)

    def forage(self):
        """Land on a tree with fruit within sight, and eat it."""
        neighborhood = self.cell.get_neighborhood(
            radius=self.scenario.vision, include_center=True
        )
        trees = [
            agent
            for agent in neighborhood.agents
            if isinstance(agent, Tree) and agent.has_fruit
        ]
        if not trees:
            return False

        tree = self.random.choice(trees)
        self.cell = tree.cell  # the bird's position moves to the center of that cell
        tree.has_fruit = False
        self.energy += self.scenario.fruit_energy
        return True

    def fly(self):
        """Fly on, turning a little; the ground cell below follows."""
        self.heading += self.random.uniform(-0.5, 0.5)
        speed = self.scenario.speed
        self.position += [
            speed * math.cos(self.heading),
            speed * math.sin(self.heading),
        ]
