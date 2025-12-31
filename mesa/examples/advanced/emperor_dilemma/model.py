import random

from agents import EmperorAgent

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.discrete_space.grid import OrthogonalMooreGrid


class EmperorModel(Model):
    """The Emperor's Dilemma Model.

    Simulates the spread of an unpopular norm using the Emperor's Dilemma logic.
    """

    def __init__(
        self,
        width=25,
        height=25,
        fraction_true_believers=0.05,
        k=0.125,
        homophily=False,
        seed=None,
    ):
        """Initializes the EmperorModel.

        Args:
            width (int): Width of the grid. Defaults to 25.
            height (int): Height of the grid. Defaults to 25.
            fraction_true_believers (float): Fraction of true believers. Defaults to 0.05.
            k (float): Enforcement cost. Defaults to 0.125.
            homophily (bool): Whether to cluster believers. Defaults to False.
            seed (int): Random seed. Defaults to None.
        """
        super().__init__(seed=seed)

        self.width = width
        self.height = height
        self.fraction_true_believers = fraction_true_believers
        self.k = k
        self.homophily = homophily

        self.grid = OrthogonalMooreGrid(
            (width, height), torus=True, capacity=1, random=self.random
        )

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
        """Initializes agents and places them on the grid.

        Distributes true believers either randomly or in a cluster based on the
        homophily setting.
        """
        num_agents = self.width * self.height
        num_believers = int(num_agents * self.fraction_true_believers)

        all_coords = [(x, y) for x in range(self.width) for y in range(self.height)]
        believer_coords = set()

        if self.homophily:
            # Pick a random center point
            center_x = self.random.randint(0, self.width - 1)
            center_y = self.random.randint(0, self.height - 1)

            start_x = center_x - int(num_believers**0.5) // 2
            start_y = center_y - int(num_believers**0.5) // 2

            for i in range(num_believers):
                bx = (start_x + (i % int(num_believers**0.5 + 1))) % self.width
                by = (start_y + (i // int(num_believers**0.5 + 1))) % self.height
                believer_coords.add((bx, by))
        else:
            believer_coords = set(random.sample(all_coords, num_believers))

        for x, y in all_coords:
            if (x, y) in believer_coords:
                p_belief = 1
                conviction = 1.0
            else:
                p_belief = -1
                conviction = random.uniform(0.01, 0.38)

            agent = EmperorAgent(self, p_belief, conviction, self.k)

            cell = self.grid[(x, y)]

            agent.cell = cell

            agent.pos = (x, y)

            self.agents.add(agent)

    def step(self):
        """Executes one step of the model.

        Shuffles agents and activates them (random order), then collects data.
        """
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)


def compute_compliance(model):
    """Computes the compliance rate of the population."""
    if not model.agents:
        return 0
    return sum(1 for a in model.agents if a.compliance == 1) / len(model.agents)


def compute_enforcement(model):
    """Computes the enforcement rate of the population."""
    if not model.agents:
        return 0
    return sum(1 for a in model.agents if a.enforcement == 1) / len(model.agents)


def compute_false_enforcement(model):
    """Computes the false enforcement rate.

    This is defined as the fraction of disbelievers who are enforcing the norm.
    """
    disbelievers = [a for a in model.agents if a.private_belief == -1]
    if not disbelievers:
        return 0
    return sum(1 for a in disbelievers if a.enforcement == 1) / len(disbelievers)
