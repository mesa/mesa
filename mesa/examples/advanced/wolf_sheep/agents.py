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
        """If possible, eat grass at current location (using property layer)."""
        # x, y is the current cell corrdiante

        x, y = self.cell.coordinate

        # if the current cell has grass eat

        if self.model.grid.grass[x, y]:
            self.energy += self.energy_from_food
            self.model.grid.grass[x, y] = False

            # Schedule regrowth via the GrassPatch agent

            grass_patch = next(
                (obj for obj in self.cell.agents if isinstance(obj, GrassPatch)), None
            )
            if grass_patch is not None:
                grass_patch.get_eaten()

    def move(self):
        """Move towards a cell where there isn't a wolf, and preferably with grown grass (using property layers)."""

        # lists to store the cell without wolves and cell with grass
        cells_without_wolves = []
        cells_with_grass = []

        for cell in self.cell.neighborhood:
            x, y = cell.coordinate
            if self.model.grid.wolves[x, y] > 0:
                continue
            cells_without_wolves.append(cell)
            if self.model.grid.grass[x, y]:
                cells_with_grass.append(cell)
        if not cells_without_wolves:
            return
        target_cells = cells_with_grass if cells_with_grass else cells_without_wolves
        self.cell = self.random.choice(target_cells)


class Wolf(Animal):
    """A wolf that walks around, reproduces (asexually) and eats sheep."""

    def feed(self):
        """If possible, eat a sheep at current location."""
        sheep = [obj for obj in self.cell.agents if isinstance(obj, Sheep)]
        if sheep:
            sheep_to_eat = self.random.choice(sheep)
            self.energy += self.energy_from_food
            sheep_to_eat.remove()

    def move(self):
        """Move to a neighboring cell, preferably one with sheep."""
        # Decrement wolf count at old cell
        x0, y0 = self.cell.coordinate
        self.model.grid.wolves[x0, y0] -= 1
        cells_with_sheep = [
            cell
            for cell in self.cell.neighborhood
            if any(isinstance(obj, Sheep) for obj in cell.agents)
        ]
        target_cells = (
            cells_with_sheep if cells_with_sheep else list(self.cell.neighborhood)
        )
        new_cell = self.random.choice(target_cells)
        self.cell = new_cell
        # Increment wolf count at new cell
        x1, y1 = new_cell.coordinate
        self.model.grid.wolves[x1, y1] += 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Increment wolf count at initial cell
        x, y = self.cell.coordinate
        self.model.grid.wolves[x, y] += 1

    def remove(self):
        # Decrement wolf count at current cell
        x, y = self.cell.coordinate
        self.model.grid.wolves[x, y] -= 1
        super().remove()


class GrassPatch(FixedAgent):
    """A patch of grass that grows at a fixed rate and can be eaten by sheep."""

    def __init__(self, model, countdown, grass_regrowth_time, cell):
        """Create a new patch of grass using property layer."""
        super().__init__(model)
        self.grass_regrowth_time = grass_regrowth_time
        self.cell = cell
        x, y = cell.coordinate
        # Set initial grass state in property layer
        self.model.grid.grass[x, y] = countdown == 0
        # Schedule initial growth if not fully grown
        if countdown != 0:
            self.model.schedule_event(self.regrow, after=countdown)

    def regrow(self):
        """Regrow the grass (set property layer)."""
        x, y = self.cell.coordinate
        self.model.grid.grass[x, y] = True

    def get_eaten(self):
        """Mark grass as eaten in property layer and schedule regrowth."""
        x, y = self.cell.coordinate
        self.model.grid.grass[x, y] = False
        self.model.schedule_event(self.regrow, after=self.grass_regrowth_time)
