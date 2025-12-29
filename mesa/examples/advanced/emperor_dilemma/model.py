import random

from agents import EmperorAgent

import mesa
from mesa.datacollection import DataCollector
from mesa.space import SingleGrid


def compute_compliance(model):
    # Use model.agents directly
    if not model.agents:
        return 0
    return sum(1 for a in model.agents if a.compliance == 1) / len(model.agents)


def compute_enforcement(model):
    if not model.agents:
        return 0
    return sum(1 for a in model.agents if a.enforcement == 1) / len(model.agents)


def compute_false_enforcement(model):
    # Fraction of Disbelievers (-1) who are enforcing the Norm (1)
    disbelievers = [a for a in model.agents if a.private_belief == -1]
    if not disbelievers:
        return 0
    return sum(1 for a in disbelievers if a.enforcement == 1) / len(disbelievers)


class EmperorModel(mesa.Model):
    def __init__(
        self,
        simulator=None,
        width=25,
        height=25,
        fraction_true_believers=0.01,
        k=0.125,
        homophily=False,
        seed=None,
    ):
        super().__init__()

        # Handle random seed
        if seed is not None:
            self.random.seed(seed)
            random.seed(seed)

        self.simulator = simulator
        self.simulator.setup(self)

        self.width = width
        self.height = height
        self.fraction_true_believers = fraction_true_believers
        self.k = k
        self.homophily = homophily

        # 2. Initialize Grid
        self.grid = SingleGrid(width, height, torus=True)

        # 3. Initialize DataCollector
        self.datacollector = DataCollector(
            model_reporters={
                "Compliance": compute_compliance,
                "Enforcement": compute_enforcement,
                "False Enforcement": compute_false_enforcement,
            }
        )

        self.init_agents()

        self.running = True
        self.datacollector.collect(self)

    def init_agents(self):
        num_agents = self.width * self.height
        num_believers = int(num_agents * self.fraction_true_believers)

        all_coords = [(x, y) for x in range(self.width) for y in range(self.height)]
        believer_coords = set()

        if self.homophily:
            # Cluster true believers in the center
            start_x = (self.width // 2) - int(num_believers**0.5) // 2
            start_y = (self.height // 2) - int(num_believers**0.5) // 2

            count = 0
            for x in range(start_x, self.width):
                for y in range(start_y, self.height):
                    if count < num_believers:
                        believer_coords.add((x, y))
                        count += 1
        else:
            believer_coords = set(random.sample(all_coords, num_believers))

        for x, y in all_coords:
            if (x, y) in believer_coords:
                p_belief = 1
                conviction = 1.0
            else:
                p_belief = -1
                conviction = random.uniform(0.0, 0.38)

            agent = EmperorAgent(self, p_belief, conviction, self.k)
            self.grid.place_agent(agent, (x, y))

    def step(self):
        self.agents.shuffle_do("step")

        self.datacollector.collect(self)
