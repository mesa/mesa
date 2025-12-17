from enum import Enum

from mesa.discrete_space import FixedAgent


class State(Enum):
    SUSCEPTIBLE = 0
    INFECTED = 1
    RESISTANT = 2


class VirusAgent(FixedAgent):
    """Individual Agent definition and its properties/interaction methods."""

    def __init__(
        self,
        model,
        initial_state,
        virus_spread_chance,
        virus_check_frequency,
        recovery_chance,
        gain_resistance_chance,
        cell,
    ):
        super().__init__(model)

        self.state = initial_state

        self.virus_spread_chance = virus_spread_chance
        self.virus_check_frequency = virus_check_frequency
        self.recovery_chance = recovery_chance
        self.gain_resistance_chance = gain_resistance_chance
        self.cell = cell

    def try_to_infect_neighbors(self):
        for agent in self.cell.neighborhood.agents:
            if (agent.state is State.SUSCEPTIBLE) and (
                self.rng.random() < self.virus_spread_chance
            ):
                agent.state = State.INFECTED

    def try_gain_resistance(self):
        if self.random.random() < self.gain_resistance_chance:
            self.state = State.RESISTANT

    def try_remove_infection(self):
        # Try to remove
        if self.random.random() < self.recovery_chance:
            # Success
            self.state = State.SUSCEPTIBLE
            self.try_gain_resistance()
        else:
            # Failed
            self.state = State.INFECTED

    def check_situation(self):
        if (self.state is State.INFECTED) and (
            self.random.random() < self.virus_check_frequency
        ):
            self.try_remove_infection()

    def step(self):
        if self.state is State.INFECTED:
            self.try_to_infect_neighbors()
        self.check_situation()
