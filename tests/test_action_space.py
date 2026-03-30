from mesa.experimental.action_space.action import Action
from mesa.experimental.action_space.action_space import ActionSpace
from mesa.experimental.action_space.constraint import Constraint


class DummyAgent:
    def __init__(self, energy):
        self.energy = energy


class EnergyConstraint(Constraint):
    def validate(self, agent, action):
        if agent.energy <= 0:
            return False, action
        return True, action


def test_action_valid():
    agent = DummyAgent(10)
    space = ActionSpace([EnergyConstraint()])

    valid, _ = space.validate(agent, Action("move"))

    assert valid


def test_action_invalid():
    agent = DummyAgent(0)
    space = ActionSpace([EnergyConstraint()])

    valid, _ = space.validate(agent, Action("move"))

    assert not valid