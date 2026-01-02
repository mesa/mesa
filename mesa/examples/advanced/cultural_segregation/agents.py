import numpy as np

import mesa


class CultureAgent(mesa.Agent):
    """An agent with a cultural opinion."""

    def __init__(self, model, opinion, threshold=0.1):
        super().__init__(model)
        self.opinion = opinion
        self.threshold = threshold
        self.happy = False
        self.pos = None

    def step(self):
        region_id = self.model.space.get_region_id(self.pos)

        region_avg_opinion = self.model.region_opinions.get(region_id, self.opinion)

        diff = abs(self.opinion - region_avg_opinion)
        self.happy = diff <= self.threshold

        x_max = self.model.space.x_max
        y_max = self.model.space.y_max

        step_size = 10.0 if not self.happy else 0.5

        dx = self.random.uniform(-step_size, step_size)
        dy = self.random.uniform(-step_size, step_size)

        new_x = self.pos[0] + dx
        new_y = self.pos[1] + dy

        new_pos = (np.clip(new_x, 0, x_max - 0.01), np.clip(new_y, 0, y_max - 0.01))

        self.model.space.move_agent(self, new_pos)
