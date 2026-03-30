"""Defines ActionSpace for validating agent actions."""


class ActionSpace:
    """Handles validation of actions using constraints."""

    def __init__(self, constraints=None):
        """Initialize with optional list of constraints."""
        self.constraints = constraints or []

    def validate(self, agent, action):
        """Validate an action against all constraints."""
        for constraint in self.constraints:
            valid, action = constraint.validate(agent, action)
            if not valid:
                return False, action
        return True, action
