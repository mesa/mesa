from mesa.discrete_space import CellAgent, FixedAgent
from mesa.experimental.mesa_signals import HasObservables, Observable, SignalType


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


class Sheep(HasObservables, Animal):
    """A sheep that walks around, reproduces (asexually) and gets eaten."""

    energy = Observable()

    def __init__(
        self, model, energy=8, p_reproduce=0.04, energy_from_food=4, cell=None
    ):
        # FIX 1: Pre-initialize subscribers to prevent crash in Animal.__init__
        # because Animal sets 'energy' before HasObservables initializes itself.
        self.subscribers = {}

        # FIX 2: Initialize MRO chain correctly
        super().__init__(model, energy, p_reproduce, energy_from_food, cell)

        self.current_task_event = None

        # Wiring the "Brain"
        self.observe("energy", SignalType.CHANGE, self.on_energy_change)

    def on_energy_change(self, signal):
        """Reactive Logic: Triggers whenever energy changes."""
        if signal.new < 6 and self.current_task_event is None:
            grass_patch = next(
                (obj for obj in self.cell.agents if isinstance(obj, GrassPatch)), None
            )
            if grass_patch and grass_patch.fully_grown:
                self.start_grazing(grass_patch)

    def start_grazing(self, grass_patch):
        """Start a DURATIVE TASK."""
        duration = 5.0
        print(
            f"Sheep {self.unique_id}: Started grazing at {self.model.time} (Duration: {duration})"
        )
        self.current_task_event = self.model.simulator.schedule_event_relative(
            self.finish_grazing, time_delta=duration, function_args=[grass_patch]
        )

    def finish_grazing(self, grass_patch):
        """Task Completion Callback."""
        print(f"Sheep {self.unique_id}: Finished grazing at {self.model.time}")
        self.energy += self.energy_from_food
        grass_patch.fully_grown = False
        self.current_task_event = None

    def step(self):
        """
        Modified Step: Handles Busy State + Interruption Scanning
        """
        if self.current_task_event:
            wolves = [
                obj for obj in self.cell.neighborhood.agents if isinstance(obj, Wolf)
            ]
            if wolves:
                print(f"Sheep {self.unique_id}: !!!! INTERRUPTED grazing to flee!")
                self.model.simulator.cancel_event(self.current_task_event)
                self.current_task_event = None
                self.move()
            return
        self.move()
        self.energy -= 1
        if self.energy < 0:
            self.remove()
        elif self.random.random() < self.p_reproduce:
            self.spawn_offspring()

    def feed(self):
        """If possible, eat grass at current location."""
        grass_patch = next(
            obj for obj in self.cell.agents if isinstance(obj, GrassPatch)
        )
        if grass_patch.fully_grown:
            self.energy += self.energy_from_food
            grass_patch.fully_grown = False

    def move(self):
        """Move towards a cell where there isn't a wolf, and preferably with grown grass."""
        cells_without_wolves = self.cell.neighborhood.select(
            lambda cell: not any(isinstance(obj, Wolf) for obj in cell.agents)
        )
        # If all surrounding cells have wolves, stay put
        if len(cells_without_wolves) == 0:
            return

        # Among safe cells, prefer those with grown grass
        cells_with_grass = cells_without_wolves.select(
            lambda cell: any(
                isinstance(obj, GrassPatch) and obj.fully_grown for obj in cell.agents
            )
        )
        # Move to a cell with grass if available, otherwise move to any safe cell
        target_cells = (
            cells_with_grass if len(cells_with_grass) > 0 else cells_without_wolves
        )
        self.cell = target_cells.select_random_cell()


class Wolf(Animal):
    """A wolf that walks around, reproduces (asexually) and eats sheep."""

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
        self.cell = target_cells.select_random_cell()


class GrassPatch(FixedAgent):
    """A patch of grass that grows at a fixed rate and can be eaten by sheep."""

    @property
    def fully_grown(self):
        """Whether the grass patch is fully grown."""
        return self._fully_grown

    @fully_grown.setter
    def fully_grown(self, value: bool) -> None:
        """Set grass growth state and schedule regrowth if eaten."""
        self._fully_grown = value

        if not value:  # If grass was just eaten
            self.model.simulator.schedule_event_relative(
                setattr,
                self.grass_regrowth_time,
                function_args=[self, "fully_grown", True],
            )

    def __init__(self, model, countdown, grass_regrowth_time, cell):
        """Create a new patch of grass.

        Args:
            model: Model instance
            countdown: Time until grass is fully grown again
            grass_regrowth_time: Time needed to regrow after being eaten
            cell: Cell to which this grass patch belongs
        """
        super().__init__(model)
        self._fully_grown = countdown == 0
        self.grass_regrowth_time = grass_regrowth_time
        self.cell = cell

        # Schedule initial growth if not fully grown
        if not self.fully_grown:
            self.model.simulator.schedule_event_relative(
                setattr, countdown, function_args=[self, "fully_grown", True]
            )
