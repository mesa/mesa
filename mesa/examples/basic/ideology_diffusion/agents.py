from mesa import Agent


class Person(Agent):
    def __init__(self, model, opinion="neutral"):
        super().__init__(model)
        self.opinion = opinion
        self.resistance = self.random.uniform(0.2, 0.8)
        self.susceptibility = self.random.uniform(0.0, 1.0)

    def step(self):
        # External pressure
        external_pressure = (
            self.model.economic_crisis + self.model.propaganda
        ) * self.susceptibility

        # Social pressure
        neighbors = self.model.grid.get_neighbors(self.pos, moore=True)
        if neighbors:
            radicals = sum(1 for n in neighbors if n.opinion == "radical")
            social_pressure = radicals / len(neighbors)
        else:
            social_pressure = 0

        influence = external_pressure + social_pressure - self.resistance

        if influence > 0.4:
            if self.opinion == "neutral":
                self.opinion = "moderate"
            elif self.opinion == "moderate":
                self.opinion = "radical"
