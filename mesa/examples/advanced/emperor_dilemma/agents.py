from mesa.experimental.devs.simulator import Priority
from mesa.discrete_space import FixedAgent

class EmperorAgent(FixedAgent):
    """An agent in the Emperor's Dilemma model.

    Inherits from FixedAgent because citizens do not move.
    """

    def __init__(self, model, private_belief, conviction, k):
        super().__init__(model)
        self.private_belief = private_belief
        self.conviction = conviction
        self.k = k
        
        self.compliance = self.private_belief
        self.enforcement = 0 
        

    def step(self):
        # 1. Observe Neighbors
        neighbors = []
        if hasattr(self, "cell") and self.cell is not None:
            neighbors = list(self.cell.neighborhood.agents)

        num_neighbors = len(neighbors)
        if num_neighbors > 0:
            # 2. Calculate Social Pressure (Eq 1)
            sum_enforcement = sum(n.enforcement for n in neighbors)
            pressure = (-self.private_belief / num_neighbors) * sum_enforcement
            
            if pressure > self.conviction:
                self.compliance = -self.private_belief 
            else:
                self.compliance = self.private_belief

            # 3. Enforcement Decision (Eq 2 & 3)
            deviant_neighbors = sum(1 for n in neighbors if n.compliance != self.private_belief)
            w_i = deviant_neighbors / num_neighbors

            if (self.compliance != self.private_belief) and (pressure > (self.conviction + self.k)):
                self.enforcement = -self.private_belief
            elif (self.compliance == self.private_belief) and ((self.conviction * w_i) > self.k):
                self.enforcement = self.private_belief
            else:
                self.enforcement = 0
        
        # EVENT SCHEDULING (DEVS):
        self.model.simulator.schedule_event_relative(
            self.step, 
            time_delta=1.0, 
            priority=Priority.DEFAULT
        )