import math
from enum import Enum

import mesa


class CitizenState(Enum):
    ACTIVE = 1
    QUIET = 2
    ARRESTED = 3


class EpsteinAgent(mesa.discrete_space.CellAgent):
    def update_neighbors(self):
        """
        Look around and see who my neighbors are
        """
        self.neighborhood = self.cell.get_neighborhood(radius=self.scenario.cop_vision)
        self.neighbors = self.neighborhood.agents
        self.empty_neighbors = [c for c in self.neighborhood if c.is_empty]

    def move(self):
        """Move to a random empty neighboring cell if movement is enabled."""
        if self.scenario.movement and self.empty_neighbors:
            new_pos = self.random.choice(self.empty_neighbors)
            self.move_to(new_pos)


class Citizen(EpsteinAgent):
    """
    A member of the general population, may or may not be in active rebellion.
    Summary of rule: If grievance - risk > threshold, rebel.

    Attributes:
        hardship: Agent's 'perceived hardship (i.e., physical or economic
            privation).' Exogenous, drawn from U(0,1).
        risk_aversion: Exogenous, drawn from U(0,1).
        state: Can be CitizenState.QUIET, ACTIVE, or ARRESTED
        jail_sentence: remaining jail time (0 if not in jail)
        grievance: deterministic function of hardship and regime_legitimacy;
            how aggrieved is agent at the regime?
        arrest_probability: agent's assessment of arrest probability, given rebellion

    Notes:
        Parameters accessed via model.scenario: legitimacy, active_threshold, citizen_vision, arrest_prob_constant
    """

    def __init__(self, model):
        """
        Create a new Citizen.

        Args:
            model: the model to which the agent belongs
        """
        super().__init__(model)
        self.hardship = self.random.random()
        self.risk_aversion = self.random.random()
        self.state = CitizenState.QUIET
        self.jail_sentence = 0
        self.grievance = self.hardship * (1 - self.scenario.legitimacy)
        self.arrest_probability = None
        self.neighborhood = []
        self.neighbors = []
        self.empty_neighbors = []

    def step(self):
        """
        Decide whether to activate, then move if applicable.
        """
        if self.jail_sentence:
            self.jail_sentence -= 1
            return  # no other changes or movements if agent is in jail.
        self.update_neighbors()
        self.update_estimated_arrest_probability()

        net_risk = self.risk_aversion * self.arrest_probability
        if (self.grievance - net_risk) > self.scenario.active_threshold:
            self.state = CitizenState.ACTIVE
        else:
            self.state = CitizenState.QUIET

        self.move()

    def update_estimated_arrest_probability(self):
        """
        Based on the ratio of cops to actives in my neighborhood, estimate the
        p(Arrest | I go active).
        """
        cops_in_vision = 0
        actives_in_vision = 1  # citizen counts herself
        for neighbor in self.neighbors:
            if isinstance(neighbor, Cop):
                cops_in_vision += 1
            elif neighbor.state == CitizenState.ACTIVE:
                actives_in_vision += 1

        # there is a body of literature on this equation
        # the round is not in the pnas paper but without it, its impossible to replicate
        # the dynamics shown there.
        self.arrest_probability = 1 - math.exp(
            -1
            * self.scenario.arrest_prob_constant
            * round(cops_in_vision / actives_in_vision)
        )


class Cop(EpsteinAgent):
    """
    A cop for life. No defection.
    Summary of rule: Inspect local vision and arrest a random active agent.

    Notes:
        Parameters accessed via model.scenario: cop_vision, max_jail_term
    """

    def __init__(self, model):
        """
        Create a new Cop.

        Args:
            model: model instance
        """
        super().__init__(model)

    def step(self):
        """
        Inspect local vision and arrest a random active agent. Move if
        applicable.
        """
        self.update_neighbors()
        active_neighbors = []
        for agent in self.neighbors:
            if isinstance(agent, Citizen) and agent.state == CitizenState.ACTIVE:
                active_neighbors.append(agent)
        if active_neighbors:
            arrestee = self.random.choice(active_neighbors)
            arrestee.jail_sentence = self.random.randint(0, self.scenario.max_jail_term)
            arrestee.state = CitizenState.ARRESTED

        self.move()
