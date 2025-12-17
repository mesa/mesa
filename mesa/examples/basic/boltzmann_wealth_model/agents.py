from mesa.discrete_space import CellAgent


class MoneyAgent(CellAgent):
    """An agent with fixed initial wealth.

    Each agent starts with 1 unit of wealth and can give 1 unit to other agents
    if they occupy the same cell.

    Attributes:
        wealth (int): The agent's current wealth (starts at 1)
    """

    def __init__(self, model, cell):
        """Create a new agent.

        Args:
            model (Model): The model instance that contains the agent
        """
        super().__init__(model)
        self.cell = cell
        self.wealth = 1

    def move(self):
        """Move the agent to a random neighboring cell."""
        self.cell = self.cell.neighborhood.select_random_cell()

    def give_money(self):
        """Give 1 unit of wealth to a random agent in the same cell."""
        cellmates = [a for a in self.cell.agents if a is not self]

        if cellmates:  # Only give money if there are other agents present
            other = (
                cellmates[0]
                if len(cellmates) == 1
                else cellmates[self.rng.integers(0, len(cellmates))]
            )
            other.wealth += 1
            self.wealth -= 1

    def step(self):
        """Execute one step for the agent:
        1. Move to a neighboring cell
        2. If wealth > 0, maybe give money to another agent in the same cell
        """
        self.move()
        if self.wealth > 0:
            self.give_money()
