class ActionSpace:
    def __init__(self, constraints=None):
        self.constraints = constraints or []

    def validate(self, agent, action):
        for constraint in self.constraints:
            valid, action = constraint.validate(agent, action)
            if not valid:
                return False, action
        return True, action
