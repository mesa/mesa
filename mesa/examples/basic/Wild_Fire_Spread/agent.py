import random
from enum import Enum
from mesa.discrete_space import CellAgent

import mesa
class AgentState(Enum):
    HEALTHY = "HEALTHY"
    BURNING = "BURNING"
    BURNED = "BURNED"


class FuelAgent(CellAgent):

    def __init__(self, model):
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

    # 🔥 Intensity
    def calculate_intensity(self):
        return self.fuel * self.flammability * (1 - self.moisture)

    # 🔥 Effective wind (ABWiSE)
    def calculate_effective_wind(self):
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False
        )

        burning_neighbors = sum(
            1 for n in neighbors if n.state == AgentState.BURNING
        )

        ratio = burning_neighbors / max(len(neighbors), 1)

        return self.model.wind_speed + 0.2 * ratio

    # 🔥 Spread probability
    def compute_spread(self):
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

    # 🔥 Spread fire
    def spread_fire(self):
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

    # 🔥 Update state
    def update_state(self):

        if self.state == AgentState.BURNING:

            # fuel consumption
            self.fuel -= 0.1

            self.burn_time -= 1

            if self.fuel <= 0 or self.burn_time <= 0:
                self.state = AgentState.BURNED

    def step(self):

        if self.state == AgentState.BURNING:
            self.spread_fire()

        self.update_state()