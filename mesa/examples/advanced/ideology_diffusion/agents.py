from mesa.discrete_space import CellAgent


class IndividualAgent(CellAgent):
    """An agent that represents an individual in the ideology diffusion model."""

    def __init__(
        self,
        model,
        cell,
        economic_dissatisfaction: float,
        propaganda_susceptibility: float,
        political_ideology: str,
        resistance_to_change: float,
    ):
        """
        Args:
            model: The model instance.
            cell: The cell where the agent is located.
            economic_dissatisfaction: Level of economic dissatisfaction (0–1).
            propaganda_susceptibility: Susceptibility to propaganda (0–1).
            political_ideology: Current ideological state.
            resistance_to_change: Resistance to ideological change (0–1).
        """
        super().__init__(model, cell)

        self.economic_dissatisfaction = economic_dissatisfaction
        self.propaganda_susceptibility = propaganda_susceptibility
        self.political_ideology = political_ideology
        self.resistance_to_change = resistance_to_change

    def __repr__(self): 
        return (
            f"IndividualAgent(economic_dissatisfaction={self.economic_dissatisfaction}, "
            f"propaganda_susceptibility={self.propaganda_susceptibility}, "
            f"political_ideology='{self.political_ideology}', "
            f"resistance_to_change={self.resistance_to_change})"
        )

   
    def step(self): 
        """Defines the agent's behavior at each step."""
        pass

