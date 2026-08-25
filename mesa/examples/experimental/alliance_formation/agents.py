import mesa


class AllianceAgent(mesa.Agent):
    """
    Agent has three attributes power (float), position (float) and level (int)

    """

    def __init__(self, model, power, position, level=0):
        super().__init__(model)
        self.power = power
        self.position = position
        self.level = level

    """
    For this demo model agent only need attributes.

    More complex models could have functions that define agent behavior.
    """
