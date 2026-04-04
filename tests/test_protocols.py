"""Tests for mesa.protocols (Locatable, HasPosition, PositionLike)."""

import random

import numpy as np

from mesa import Agent, Model
from mesa.discrete_space import Cell, CellAgent, FixedAgent
from mesa.experimental.continuous_space import ContinuousSpace, ContinuousSpaceAgent
from mesa.protocols import HasPosition, Locatable


class PositionAgent(Agent, HasPosition):
    """Test agent using the HasPosition mixin."""


# HasPosition tests
def test_has_position_default_none():
    model = Model()
    agent = PositionAgent(model)
    assert agent.position is None


def test_has_position_set_ndarray():
    model = Model()
    agent = PositionAgent(model)
    pos = np.array([1.5, 2.5])
    agent.position = pos
    np.testing.assert_array_equal(agent.position, pos)


def test_has_position_set_tuple():
    model = Model()
    agent = PositionAgent(model)
    agent.position = (3, 4)
    assert agent.position == (3, 4)


def test_has_position_reset_none():
    model = Model()
    agent = PositionAgent(model)
    agent.position = (1.0, 2.0)
    agent.position = None
    assert agent.position is None


def test_has_position_independent_instances():
    model = Model()
    a = PositionAgent(model)
    b = PositionAgent(model)
    a.position = np.array([1.0, 2.0])
    assert b.position is None


# Locatable protocol tests
def test_has_position_satisfies_locatable():
    model = Model()
    agent = PositionAgent(model)
    assert isinstance(agent, Locatable)


def test_cell_satisfies_locatable():
    cell = Cell((3, 4), capacity=None, random=random.Random())
    assert isinstance(cell, Locatable)
    pos = cell.position
    assert isinstance(pos, np.ndarray)
    np.testing.assert_array_equal(pos, np.array([3.0, 4.0]))


def test_cell_agent_satisfies_locatable():
    model = Model()
    agent = CellAgent(model)
    assert isinstance(agent, Locatable)


def test_cell_agent_position_none_when_unplaced():
    model = Model()
    agent = CellAgent(model)
    assert agent.position is None


def test_cell_agent_position_after_placement():
    model = Model()
    agent = CellAgent(model)
    cell = Cell((1, 2), capacity=None, random=random.Random())
    agent.cell = cell
    assert isinstance(agent.position, np.ndarray)
    np.testing.assert_array_equal(agent.position, np.array([1.0, 2.0]))


def test_fixed_agent_satisfies_locatable():
    model = Model()
    agent = FixedAgent(model)
    assert isinstance(agent, Locatable)


def test_fixed_agent_position_after_placement():
    model = Model()
    agent = FixedAgent(model)
    cell = Cell((5, 6), capacity=None, random=random.Random())
    agent.cell = cell
    assert isinstance(agent.position, np.ndarray)
    np.testing.assert_array_equal(agent.position, np.array([5.0, 6.0]))


def test_continuous_space_agent_satisfies_locatable():
    model = Model()
    space = ContinuousSpace([[0, 10], [0, 10]], random=model.random)
    agent = ContinuousSpaceAgent(space, model)
    assert isinstance(agent, Locatable)
