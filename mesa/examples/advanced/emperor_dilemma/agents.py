import mesa


class EmperorAgent(mesa.Agent):
    """
    An agent in the Emperor's Dilemma model.
    Attributes:
        private_belief (int): 1 (True Believer) or -1 (Disbeliever).
        conviction (float): Strength of conviction (S_i).
        compliance (int): Current public behavior (1 = Comply, -1 = Deviate).
        enforcement (int): Current enforcement behavior (1 = Enforce Compliance, -1 = Enforce Deviance, 0 = None).
        k (float): Cost of enforcement.
    """

    def __init__(self, model, private_belief, conviction, k):
        super().__init__(model)
        self.private_belief = private_belief
        self.conviction = conviction
        self.k = k

        # Initial state: conform to private belief, no enforcement
        self.compliance = self.private_belief
        self.enforcement = 0

    def step(self):
        """1. Observe Neighbors
        Get neighboring agents (Moore neighborhood, excluding self)

        """
        neighbors = self.model.grid.get_neighbors(
            self.pos, moore=True, include_center=False
        )
        num_neighbors = len(neighbors)
        if num_neighbors == 0:
            return

        """ 2. Calculate Social Pressure
        Sum of neighbors' enforcement decisions (E_j)

        """
        sum_enforcement = sum(n.enforcement for n in neighbors)

        """ 3. Compliance Decision (Eq 1)

        Pressure opposes belief if (-B_i * sum_enforcement) is positive.
        We normalize by N (num_neighbors) as implied by the fraction context in Eq 1.

        """
        pressure = (-self.private_belief / num_neighbors) * sum_enforcement
        if pressure > self.conviction:
            self.compliance = -self.private_belief  # Falsify/Flip
        else:
            self.compliance = self.private_belief  # True to belief

        """4. Enforcement Decision (Eq 2 & 3)

        Calculate Need for Enforcement (W_i): Proportion of neighbors deviating from agent's belief
        If agent believes 1, count neighbors doing -1. If agent believes -1, count neighbors doing 1.

        """
        deviant_neighbors = sum(
            1 for n in neighbors if n.compliance != self.private_belief
        )
        w_i = deviant_neighbors / num_neighbors

        """ Determine Enforcement State

        Case A: False Enforcement (Enforcing against own belief)
        Condition: Pressure > Conviction + Cost

        Case B: True Enforcement (Enforcing own belief)
        Condition: (Conviction * Need) > Cost

        Case C: No Enforcement
        """

        if (self.compliance != self.private_belief) and (
            pressure > (self.conviction + self.k)
        ):
            self.enforcement = -self.private_belief

        elif (self.compliance == self.private_belief) and (
            (self.conviction * w_i) > self.k
        ):
            self.enforcement = self.private_belief

        else:
            self.enforcement = 0
