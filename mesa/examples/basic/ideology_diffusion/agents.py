from mesa.discrete_space import CellAgent


class IndividualAgent(CellAgent):
    
    """An individual agent with a political ideology."""
    
    def __init__(
        self,
        model,
        cell,
        economic_dissatisfaction: float,
        propaganda_susceptibility: float,
        resistance_to_change: float,
        political_ideology: int = 0,
    ):
        """"
        Args:
        model: The model instance.
        cell: The cell the agent occupies.
        economic_dissatisfaction: Level of economic dissatisfaction (0 to 1).   
        propaganda_susceptibility: Susceptibility to media propaganda (0 to 1).
        resistance_to_change: Resistance to changing political views (0 to 1).
        political_ideology: Initial political ideology (0: Neutral, 1: Moderated, 2: Radical).
        """
        
        super().__init__(model, cell)
        self.economic_dissatisfaction = economic_dissatisfaction
        self.propaganda_susceptibility = propaganda_susceptibility
        self.resistance_to_change = resistance_to_change
        self.political_ideology = political_ideology

    def step(self):
        self.ideology_dissemination()

    def ideology_dissemination(self):
        pressure = 0

        # Economic crisis as a catalyst for radicalization
        if (
            self.model.economic_crisis
            and self.model.unemployment_increase > 0.1
            and self.economic_dissatisfaction > 0.5
        ):
            pressure += 1

        # Media influence based on individual susceptibility
        if (
            self.model.media_influence
            and self.random.random() < self.propaganda_susceptibility
        ):
            pressure += 1

        # Government repression can either intimidate or provoke backlash
        if self.model.government_repression:
            if self.random.random() < self.resistance_to_change:
                pressure += 1 
            else:
                pressure -= 1 

        # Ensure ideology remains within 0 (Neutral) and 2 (Radical)
        new_ideology = self.political_ideology + pressure
        self.political_ideology = max(0, min(2, new_ideology))