class Action:
    def __init__(self, name, **params):
        self.name = name
        self.params = params

    def __repr__(self):
        return f"Action({self.name}, {self.params})"