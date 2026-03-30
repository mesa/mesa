"""Defines constraint base class for action validation."""


class Constraint:
    """Base class for constraints applied to actions."""

    def validate(self, agent, action):
        """Validate an action against the constraint."""
        return True, action
