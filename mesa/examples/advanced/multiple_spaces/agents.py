import numpy as np

from mesa import Agent


class MultiSpaceAgent(Agent):
    def __init__(self, model, position):
        super().__init__(model)

        # Set position once
        self.position = np.array(position, dtype=float)

        # Register in all spaces
        model.continuous.add(self)
        model.grid.add(self)
        model.hex_grid.add(self)

    def step(self):
        # Query current locations
        grid_cell = self.model.grid.cell(self)
        hex_cell = self.model.hex_grid.cell(self)

        print(f"Agent {self.unique_id} at position {self.position}")
        print(f"  Grid cell: {grid_cell.coordinate}")
        print(f"  Hex cell: {hex_cell.coordinate}")

        # Move via grid - hex updates automatically!
        if self.random.random() < 0.5:
            neighbors = list(grid_cell.connections.values())
            if neighbors:
                target = self.random.choice(neighbors)
                self.model.grid.move(self, target, align="center")
                print(f"  Moved to grid {target.coordinate}")
