from mesa.discrete_space import CellAgent, FixedAgent
from mesa.experimental.mesa_signals import (
    Computable,
    Computed,
    ContinuousObservable,
    HasObservables,
    Observable,
)


class Animal(CellAgent, HasObservables):
    """The base animal class with reactive energy management."""

    # Energy depletes continuously over time
    energy = ContinuousObservable(
        initial_value=8.0, rate_func=lambda value, elapsed, agent: -agent.metabolic_rate
    )

    # Computed property: animal is hungry when energy is low
    is_hungry = Computable()

    # Computed property: animal can reproduce when energy is sufficient
    can_reproduce = Computable()

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

        # Set base metabolic rate (energy loss per time unit when idle)
        self.metabolic_rate = 0.5

        # Initialize energy (triggers continuous depletion)
        self.energy = energy
        self.p_reproduce = p_reproduce
        self.energy_from_food = energy_from_food
        self.cell = cell

        # Set up computed properties
        self.is_hungry = Computed(lambda: self.energy < self.energy_from_food * 2)
        self.can_reproduce = Computed(lambda: self.energy > self.energy_from_food * 4)

        # Register threshold: die when energy reaches zero
        self.add_threshold("energy", 0.0, self._on_energy_depleted)

        # Register threshold: become critically hungry at 25% of starting energy
        self.add_threshold("energy", energy * 0.25, self._on_critical_hunger)

    def _on_energy_depleted(self, signal):
        """Called when energy crosses zero - animal dies."""
        if signal.direction == "down":  # Only trigger on downward crossing
            self.remove()

    def _on_critical_hunger(self, signal):
        """Called when energy becomes critically low."""
        if signal.direction == "down":
            # Increase metabolic efficiency when starving (survival mode)
            self.metabolic_rate *= 0.8

    def spawn_offspring(self):
        """Create offspring by splitting energy and creating new instance."""
        self.energy /= 2  # This updates the continuous observable
        self.__class__(
            self.model,
            self.energy,
            self.p_reproduce,
            self.energy_from_food,
            self.cell,
        )

    def feed(self):
        """Abstract method to be implemented by subclasses."""
        raise NotImplementedError

    def step(self):
        """Execute one step of the animal's behavior."""
        # Move to neighboring cell (uses more energy than standing still)
        self.metabolic_rate = 1.0  # Movement costs more energy
        self.move()

        # Try to feed
        self.feed()

        # Return to resting metabolic rate
        self.metabolic_rate = 0.5

        # Reproduce if conditions are met (using computed property)
        if self.can_reproduce and self.random.random() < self.p_reproduce:
            self.spawn_offspring()


class Sheep(Animal):
    """A sheep that walks around, reproduces and gets eaten.

    Sheep prefer cells with grass and avoid wolves. They gain energy by
    eating grass, which continuously depletes over time.
    """

    def feed(self):
        """If possible, eat grass at current location."""
        grass_patch = next(
            obj for obj in self.cell.agents if isinstance(obj, GrassPatch)
        )
        if grass_patch.fully_grown:
            # Eating gives instant energy boost
            self.energy += self.energy_from_food
            grass_patch.fully_grown = False

    def move(self):
        """Move towards a cell where there isn't a wolf, and preferably with grown grass."""
        cells_without_wolves = self.cell.neighborhood.select(
            lambda cell: not any(isinstance(obj, Wolf) for obj in cell.agents)
        )

        # If all surrounding cells have wolves, stay put (fear overrides hunger)
        if len(cells_without_wolves) == 0:
            return

        # If critically hungry, prioritize grass over safety
        if self.is_hungry:  # Using computed property
            cells_with_grass = cells_without_wolves.select(
                lambda cell: any(
                    isinstance(obj, GrassPatch) and obj.fully_grown
                    for obj in cell.agents
                )
            )
            # Move to grass if available, otherwise any safe cell
            target_cells = (
                cells_with_grass if len(cells_with_grass) > 0 else cells_without_wolves
            )
        else:
            # Not hungry - just avoid wolves
            target_cells = cells_without_wolves

        self.cell = target_cells.select_random_cell()


class Wolf(Animal):
    """A wolf that walks around, reproduces and eats sheep.

    Wolves are more efficient predators, with higher base energy and
    metabolic rate. They actively hunt sheep and gain substantial energy
    from successful kills.
    """

    def __init__(
        self, model, energy=20, p_reproduce=0.05, energy_from_food=20, cell=None
    ):
        """Initialize a wolf with higher energy needs than sheep."""
        super().__init__(model, energy, p_reproduce, energy_from_food, cell)
        # Wolves have higher metabolic rate (they're larger predators)
        self.metabolic_rate = 1.0

    def feed(self):
        """If possible, eat a sheep at current location."""
        sheep = [obj for obj in self.cell.agents if isinstance(obj, Sheep)]
        if sheep:  # Successful hunt
            sheep_to_eat = self.random.choice(sheep)
            # Eating gives instant energy boost
            self.energy += self.energy_from_food
            sheep_to_eat.remove()

    def move(self):
        """Move to a neighboring cell, preferably one with sheep."""
        # When hungry, actively hunt for sheep
        if self.is_hungry:  # Using computed property
            cells_with_sheep = self.cell.neighborhood.select(
                lambda cell: any(isinstance(obj, Sheep) for obj in cell.agents)
            )
            target_cells = (
                cells_with_sheep
                if len(cells_with_sheep) > 0
                else self.cell.neighborhood
            )
        else:
            # When not hungry, wander randomly (conserve energy)
            target_cells = self.cell.neighborhood

        self.cell = target_cells.select_random_cell()


class GrassPatch(FixedAgent, HasObservables):
    """A patch of grass that grows at a fixed rate and can be eaten by sheep.

    Grass growth is modeled as a continuous process with a fixed regrowth time.
    """

    # Observable: grass growth state
    fully_grown = Observable()

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

        # Listen for when grass gets eaten, schedule regrowth
        self.observe("fully_grown", "change", self._on_growth_change)

        # Schedule initial growth if not fully grown
        if not self.fully_grown:
            self.model.simulator.schedule_event_relative(self._regrow, countdown)

    def _on_growth_change(self, signal):
        """React to grass being eaten - schedule regrowth."""
        if signal.new is False:  # Grass was just eaten
            self.model.simulator.schedule_event_relative(
                self._regrow, self.grass_regrowth_time
            )

    def _regrow(self):
        """Regrow the grass patch."""
        self.fully_grown = True
