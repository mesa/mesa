
from mesa.discrete_space import CellAgent, FixedAgent


class Animal(CellAgent):
    """The base animal class."""

    def __init__(
        self, model, energy=8, p_reproduce=0.04, energy_from_food=4, cell=None
    ):
        """Initialize an animal.

        Args:
            model: Model instance
            energy: Starting amount of energy
            p_reproduce: Probability of reproduction (asexual)
            energy_from_food: Energy obtained from 1 unit of food
            cell: Cell in which the animal starts
        """
        super().__init__(model)
        self.energy = energy
        self.p_reproduce = p_reproduce
        self.energy_from_food = energy_from_food
        self.cell = cell

    def spawn_offspring(self):
        """Create offspring by splitting energy and creating new instance."""
        self.energy /= 2
        self.__class__(
            self.model,
            self.energy,
            self.p_reproduce,
            self.energy_from_food,
            self.cell,
        )

    def feed(self):
        """Abstract method to be implemented by subclasses."""

    def move(self):
        """Abstract method to be implemented by subclasses."""

    def step(self):
        """Execute one step of the animal's behavior."""
        # Move to random neighboring cell
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
            obj for obj in self.cell.agents if isinstance(obj, GrassPatch)
        )
        if grass_patch.is_fully_grown():
            self.energy += self.energy_from_food
            grass_patch.get_eaten()

    def move(self):
        """Move towards a cell where there isn't a wolf, and preferably with grown grass."""
        # Get all surrounding available cells
        neighbors = self.cell.neighborhood

        safe_cells = []
        safe_grass_cells = []

        # Collect all safe cells and safe cells with grass
        for neighbor in neighbors:
            if not neighbor.wolves:
                safe_cells.append(neighbor)
                if neighbor.grass:
                    safe_grass_cells.append(neighbor)

        # If all surrounding cells have wolves, stay put
        if not safe_cells:
            return

        # Move to a cell with grass if available, otherwise move to any safe cell
        target_cells = safe_grass_cells if safe_grass_cells else safe_cells
        self.cell = self.random.choice(target_cells)

class Wolf(Animal):
    """A wolf that walks around, reproduces (asexually) and eats sheep."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cell.wolves = True

    def feed(self):
        """If possible, eat a sheep at current location."""
        sheep = [obj for obj in self.cell.agents if isinstance(obj, Sheep)]
        if sheep:  # If there are any sheep present
            sheep_to_eat = self.random.choice(sheep)
            self.energy += self.energy_from_food
            sheep_to_eat.remove()

    def move(self):
        """Move to a neighboring cell, preferably one with sheep."""
        cells_with_sheep = self.cell.neighborhood.select(
            lambda cell: any(isinstance(obj, Sheep) for obj in cell.agents)
        )
        target_cells = (
            cells_with_sheep if len(cells_with_sheep) > 0 else self.cell.neighborhood
        )

        # Mark the cell as unoccupied by a wolf
        self.cell.wolves = False

        self.cell = target_cells.select_random_cell()

        # Mark the cell as occupied by a wolf
        self.cell.wolves = True

    def remove(self):
        self.cell.wolves = False
        super().remove()


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
        self.grass_regrowth_time = grass_regrowth_time
        self.cell = cell
        self.cell.grass = countdown == 0

        # Schedule initial growth if not fully grown
        if not self.cell.grass:
            self.model.schedule_event(self.regrow, after=countdown)

    def regrow(self):
        """Regrow the grass."""
        self.cell.grass = True

    def get_eaten(self):
        """Mark grass as eaten and schedule regrowth."""
        self.cell.grass = False
        self.model.schedule_event(self.regrow, after=self.grass_regrowth_time)

    def is_fully_grown(self):
        """Return whether the grass patch is fully grown."""
        return self.cell.grass
