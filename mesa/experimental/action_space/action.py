"""Defines the Action class for agent intent."""


class Action:
    """Represents an agent's intended action."""

    def __init__(self, name, **params):
        """Initialize an action with a name and optional parameters."""
        self.name = name
        self.params = params
