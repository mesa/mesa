"""Tests for ActionSpace module."""

from mesa.experimental.action_space.action import Action
from mesa.experimental.action_space.action_space import ActionSpace
from mesa.experimental.action_space.constraint import Constraint


class DummyAgent:
    """Simple agent with energy attribute."""

    def __init__(self, energy):
        """Initialize agent with energy."""
        self.energy = energy


class EnergyConstraint(Constraint):
    """Constraint that checks agent energy."""

    def validate(self, agent, action):
        """Block action if energy is zero or below."""
        if agent.energy <= 0:
            return False, action
        return True, action


def test_action_valid():
    """Test that action is valid when energy is positive."""
    agent = DummyAgent(10)
    space = ActionSpace([EnergyConstraint()])
    valid, _ = space.validate(agent, Action("move"))
    assert valid


def test_action_invalid():
    """Test that action is invalid when energy is zero."""
    agent = DummyAgent(0)
    space = ActionSpace([EnergyConstraint()])
    valid, _ = space.validate(agent, Action("move"))
    assert not valid