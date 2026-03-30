"""Hotelling's Law Model - Demonstrates spatial competition between firms.

This model simulates Harold Hotelling's 1929 model of spatial competition,
where firms choose locations on a line to maximize their market share.
"""

import mesa
from mesa.discrete_space import CellAgent, OrthogonalMooreGrid


class Firm(CellAgent):
    """A firm that chooses a location to maximize profit."""

    def __init__(self, model, firm_id):
        super().__init__(model)
        self.firm_id = firm_id
        self.profit = 0

    def step(self):
        """Firm decision-making step."""
        # Calculate profit based on current position
        self.calculate_profit()

    def calculate_profit(self):
        """Calculate profit based on current position."""
        if self.cell is not None:
            # Simple profit calculation based on distance from center
            center = self.model.grid.width // 2
            x_pos = self.cell.coordinate[0]
            distance_from_center = abs(x_pos - center)
            self.profit = max(0, 100 - distance_from_center * 10)
        else:
            self.profit = 0


class HotellingModel(mesa.Model):
    """Model representing Hotelling's spatial competition."""

    def __init__(self, num_firms=2, grid_width=10):
        super().__init__()
        self.num_firms = num_firms
        self.grid = OrthogonalMooreGrid(
            (grid_width, 1), random=self.random
        )  # 1D line represented as thin grid

        # Create firms and place them on the grid
        for i in range(self.num_firms):
            firm = Firm(self, i)
            # Place firms at random positions initially
            x = self.random.randrange(self.grid.width)
            y = 0  # Single row
            # Properly place agent on grid - this sets firm.cell automatically
            firm.cell = self.grid[x, y]

        self.datacollector = mesa.DataCollector(
            agent_reporters={
                "Profit": "profit",
                "Position": lambda agent: agent.cell.coordinate if agent.cell else None,
            }
        )

    def step(self):
        """Advance model by one step."""
        self.agents.do("step")
        self.datacollector.collect(self)
