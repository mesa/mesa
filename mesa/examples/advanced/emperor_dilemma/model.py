from mesa import Model
from mesa.datacollection import DataCollector
import random
from mesa.discrete_space.grid import OrthogonalMooreGrid
from agents import EmperorAgent
from mesa.experimental.devs.simulator import DEVSimulator, Priority

class EmperorModel(Model):
    """The Emperor's Dilemma Model.

    Simulates the spread of an unpopular norm using the Emperor's Dilemma logic.
    """

    def __init__(self, simulator: DEVSimulator = None, width=25, height=25, 
                 fraction_true_believers=0.05, k=0.125, homophily=False, seed=None):
        """Initialize the EmperorModel.

        Args:
            simulator (DEVSimulator): The DEVS simulator instance.
            width (int): Width of the grid.
            height (int): Height of the grid.
            fraction_true_believers (float): Fraction of population that are true believers.
            k (float): Cost of enforcement.
            homophily (bool): If True, true believers are clustered.
            seed (int): Random seed.
        """
        super().__init__(seed=seed)
            
        self.simulator = simulator
        self.simulator.setup(self)

        self.width = width
        self.height = height
        self.fraction_true_believers = fraction_true_believers
        self.k = k
        self.homophily = homophily

        self.grid = OrthogonalMooreGrid((width, height), torus=True, random=self.random)
        
        self.datacollector = DataCollector(
            model_reporters={
                "Compliance": compute_compliance,
                "Enforcement": compute_enforcement,
                "False Enforcement": compute_false_enforcement
            }
        )

        self.init_agents()
        self.running = True
        
        self.datacollector.collect(self)
        
        self.simulator.schedule_event_relative(self.step, time_delta=1.0, priority=Priority.HIGH)

    def init_agents(self):
        """Initialize agents and place them on the grid.
        
        Distributes true believers either randomly or in a cluster based on homophily setting.
        """
        num_agents = self.width * self.height
        num_believers = int(num_agents * self.fraction_true_believers)
        
        all_coords = [(x, y) for x in range(self.width) for y in range(self.height)]
        believer_coords = set()

        if self.homophily:
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

            random_start_time = random.uniform(0.0, 1.0)
            self.simulator.schedule_event_absolute(agent.step, time=random_start_time)

    def step(self):
        """Perform a model step.
        
        Collects data and schedules the next step event.
        """
        self.datacollector.collect(self)
        self.simulator.schedule_event_relative(self.step, time_delta=1.0, priority=Priority.HIGH)


def compute_compliance(model):
    """Compute the proportion of agents complying with the norm."""
    if not model.agents: return 0
    return sum(1 for a in model.agents if a.compliance == 1) / len(model.agents)

def compute_enforcement(model):
    """Compute the proportion of agents enforcing the norm."""
    if not model.agents: return 0
    return sum(1 for a in model.agents if a.enforcement == 1) / len(model.agents)

def compute_false_enforcement(model):
    """Compute the proportion of disbelievers who are enforcing the norm."""
    disbelievers = [a for a in model.agents if a.private_belief == -1]
    if not disbelievers: return 0
    return sum(1 for a in disbelievers if a.enforcement == 1) / len(disbelievers)