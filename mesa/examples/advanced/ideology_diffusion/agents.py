from mesa.discrete_space import CellAgent


class IndividualAgent(CellAgent):
    """
    An agent representing an individual in an ideology diffusion model.
    
    political_ideology:
        0 = neutral
        1 = moderate
        2 = radical
    """

    def __init__(
        self,
        model,
        cell,
        economic_dissatisfaction: float,
        propaganda_susceptibility: float,
        resistance_to_change: float,
        political_ideology: int = 0,
    ):
        """
        Args:
            model: The model instance.
            cell: The cell where the agent is located.
            economic_dissatisfaction: [0, 1] dissatisfaction with economy.
            propaganda_susceptibility: [0, 1] susceptibility to media influence.
            resistance_to_change: [0, 1] resistance to ideological change.
            political_ideology: Initial ideology (0–2).
        """
        super().__init__(model, cell)

        self.economic_dissatisfaction = economic_dissatisfaction
        self.propaganda_susceptibility = propaganda_susceptibility
        self.resistance_to_change = resistance_to_change
        self.political_ideology = political_ideology

    def __repr__(self):
        return (
            f"IndividualAgent("
            f"ideology={self.political_ideology}, "
            f"econ={self.economic_dissatisfaction:.2f}, "
            f"media={self.propaganda_susceptibility:.2f}, "
            f"resistance={self.resistance_to_change:.2f})"
        )

    #Define the step function, and the hole agent behavior

    def step(self):
        """Execute one step of the agent."""
        self.ideology_dissemination()

    def ideology_dissemination(self):
        """
        Update agent ideology based on global conditions and individual traits.
        """
        delta = 0

        # Economic crisis effect
        if (
            self.model.economic_crisis
            and self.model.unemployment_increase > 0.1
            and self.economic_dissatisfaction > 0.5
        ):
            delta += 1

        # Media influence
        if (
            self.model.media_influence
            and self.random.random() < self.propaganda_susceptibility
        ):
            delta += 1

        # Government repression
        if self.model.government_repression:
            if self.random.random() < self.resistance_to_change:
                delta -= 1
            else:
                delta += 1

        # Apply change and clamp
        self.political_ideology += delta
        self.political_ideology = max(0, min(2, self.political_ideology))
