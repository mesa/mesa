"""Agent definitions for the Wild Fire Spread example."""

import random
from enum import Enum

from mesa.discrete_space import CellAgent


class AgentState(Enum):
    """Possible fire states for a fuel cell."""

    HEALTHY = "HEALTHY"
    BURNING = "BURNING"
    BURNED = "BURNED"


class FuelAgent(CellAgent):
    """A single burnable cell that can ignite, spread fire, and burn out."""

    def __init__(self, model):
        """Initialize fuel, moisture, terrain, and fire-related attributes."""
        super().__init__(model)

        # --- State ---
        self.state = AgentState.HEALTHY
        self.burn_time = 0

        # --- Fuel ---
        self.fuel_type = random.choice(["tree", "grass"])
        self.flammability = 0.6 if self.fuel_type == "tree" else 0.9

        self.fuel = random.uniform(0.5, 1.0)
        self.moisture = model.humidity / 100
        self.slope = random.uniform(0, 1)

        # --- Fire ---
        self.intensity = 0

        # --- Firebreak ---
        self.is_firebreak = False

        # --- Constants ---
        self.P_BASE = 0.3
        self.min_burn_time = 3
        self.max_burn_time = 6

    def calculate_intensity(self):
        """Return the current fire intensity based on fuel, flammability, and moisture."""
        return self.fuel * self.flammability * (1 - self.moisture)

    def calculate_effective_wind(self):
        """Estimate local wind effect from global wind and burning neighbors."""
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False
        )

        burning_neighbors = sum(1 for n in neighbors if n.state == AgentState.BURNING)

        ratio = burning_neighbors / max(len(neighbors), 1)

        return self.model.wind_speed + 0.2 * ratio

    def compute_spread(self):
        """Compute the probability that this agent ignites a neighboring cell."""
        intensity = self.calculate_intensity()
        effective_wind = self.calculate_effective_wind()

        direction_factor = random.choice([2, 1, 0.5])

        P = (
            self.P_BASE
            * (1 + effective_wind * direction_factor)
            * (1 + self.slope)
            * intensity
        )

        return min(P, 1)  # cap probability

    def spread_fire(self):
        """Attempt to ignite healthy neighboring agents that are not firebreaks."""
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False
        )

        for neighbor in neighbors:
            if neighbor.state != AgentState.HEALTHY:
                continue

            if neighbor.is_firebreak:
                continue

            P_spread = self.compute_spread()

            if random.random() < P_spread:
                neighbor.state = AgentState.BURNING
                neighbor.burn_time = random.randint(
                    self.min_burn_time, self.max_burn_time
                )

                # 🔥 Preheating
                neighbor.flammability += 0.05

    def update_state(self):
        """Reduce fuel and burn time, then mark the agent as burned if exhausted."""

        if self.state == AgentState.BURNING:
            # fuel consumption
            self.fuel -= 0.1

            self.burn_time -= 1

            if self.fuel <= 0 or self.burn_time <= 0:
                self.state = AgentState.BURNED

    def step(self):
        """Advance the agent by one simulation step."""

        if self.state == AgentState.BURNING:
            self.spread_fire()

        self.update_state()
