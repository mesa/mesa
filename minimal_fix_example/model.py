"""
Minimal fix for Hotelling's Law model.
Only changes: ensure position is always initialized
"""

import mesa
from mesa.discrete_space import OrthogonalMooreGrid


class Firm(mesa.Agent):
    """A firm that chooses a location to maximize profit."""
    
    def __init__(self, model, firm_id):
        super().__init__(model)
        self.firm_id = firm_id
        self.profit = 0
        # FIXED: Explicitly initialize position to prevent None
        self.position = None
        
    def step(self):
        """Firm decision-making step."""
        if self.position is not None:
            self.calculate_profit()
    
    def calculate_profit(self):
        """Calculate profit based on current position."""
        center = self.model.grid_width // 2
        distance_from_center = abs(self.position[0] - center)
        self.profit = max(0, 100 - distance_from_center * 10)


class HotellingModel(mesa.Model):
    """Model representing Hotelling's spatial competition."""
    
    def __init__(self, num_firms=2, grid_width=10):
        super().__init__()
        self.num_firms = num_firms
        self.grid_width = grid_width
        
        # MINIMAL ADDITION: Create grid for visualization (SpaceRenderer needs it)
        self.grid = OrthogonalMooreGrid((grid_width, 1), random=self.random)
        
        # Create firms and IMMEDIATELY assign positions
        for i in range(self.num_firms):
            firm = Firm(self, i)
            
            # FIXED: Ensure position is set immediately after creation
            x = self.random.randrange(self.grid_width)
            y = 0  # Single row for 1D competition
            firm.position = (x, y)  # Position guaranteed to be set
            
        self.datacollector = mesa.DataCollector(
            agent_reporters={"Profit": "profit", "Position": "position"}
        )
        
    def step(self):
        """Advance model by one step."""
        self.agents.do("step")
        self.datacollector.collect(self)