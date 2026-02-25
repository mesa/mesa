from mesa.discrete_space import CellAgent


class PDAgent(CellAgent):
    """Agent member of the iterated, spatial prisoner's dilemma model."""

    def __init__(self, model, starting_move=None, cell=None):
        """
        Create a new Prisoner's Dilemma agent.

        Args:
            model: model instance
            starting_move: If provided, determines the agent's initial state:
                           C(ooperating) or D(efecting). Otherwise, random.
        """
        super().__init__(model)
        self.score = 0
        self.cell = cell
        if starting_move:
            self.move = starting_move
        else:
            self.move = self.random.choice(["C", "D"])
        self.next_move = None

    @property
    def is_cooroperating(self):
        return self.move == "C"

    def step(self):
        """Get the best neighbor's move, and change own move accordingly
        if better than own score."""

        # neighbors = self.model.grid.get_neighbors(self.pos, True, include_center=True)
        neighbors = [*list(self.cell.neighborhood.agents), self]
        best_neighbor = max(neighbors, key=lambda a: a.score)
        self.next_move = best_neighbor.move

    def advance(self):
        self.move = self.next_move
        self.score += self.increment_score()

    def increment_score(self):
        neighbors = self.cell.neighborhood.agents
        if self.model.activation_order == "Simultaneous":
            moves = [neighbor.next_move for neighbor in neighbors]
        else:
            moves = [neighbor.move for neighbor in neighbors]
        return sum(self.model.payoff[(self.move, move)] for move in moves)
