from mesa.discrete_space import CellAgent


class SchellingAgent(CellAgent):
    """Schelling segregation agent."""

    def __init__(
        self, model, cell, agent_type: int, homophily: float = 0.4, radius: int = 1
    ) -> None:
        """Create a new Schelling agent.
        Args:
            model: The model instance the agent belongs to
            agent_type: Indicator for the agent's type (minority=1, majority=0)
            homophily: Minimum number of similar neighbors needed for happiness
            radius: Search radius for checking neighbor similarity
        """
        super().__init__(model)
        self.cell = cell
        self.type = agent_type
        self.homophily = homophily
        self.radius = radius
        self.happy = False

    def assign_state(self) -> None:
        """Determine if agent is happy and move if necessary."""
        neighbors = list(self.cell.get_neighborhood(radius=self.radius).agents)

        # Count similar neighbors
        similar_neighbors = len([n for n in neighbors if n.type == self.type])

        # Calculate the fraction of similar neighbors
        if (valid_neighbors := len(neighbors)) > 0:
            similarity_fraction = similar_neighbors / valid_neighbors
        else:
            # If there are no neighbors, the similarity fraction is 0
            similarity_fraction = 0.0

        if similarity_fraction < self.homophily:
            self.happy = False
        else:
            self.happy = True
            self.model.happy += 1

    def step(self) -> None:
        # Move if unhappy
        if not self.happy:
            self.cell = self.model.grid.select_random_empty_cell()
