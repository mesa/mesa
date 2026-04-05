"""Tests for mesa.protocols (Locatable, HasPosition, PositionLike)."""

import random

import numpy as np

from mesa import Agent, Model
from mesa.discrete_space import Cell, CellAgent, FixedAgent
from mesa.experimental.continuous_space import ContinuousSpace, ContinuousSpaceAgent
from mesa.protocols import HasPosition, Locatable


class PositionAgent(Agent, HasPosition):
    """Test agent using the HasPosition mixin."""


def test_has_position_default_none():
    """HasPosition position is None before any assignment."""
    model = Model()
    agent = PositionAgent(model)
    assert agent.position is None


def test_has_position_set_ndarray():
    """HasPosition stores and returns a numpy array position correctly."""
    model = Model()
    agent = PositionAgent(model)
    pos = np.array([1.5, 2.5])
    agent.position = pos
    np.testing.assert_array_equal(agent.position, pos)


def test_has_position_set_tuple():
    """HasPosition stores and returns a tuple position correctly."""
    model = Model()
    agent = PositionAgent(model)
    agent.position = (3, 4)
    assert agent.position == (3, 4)


def test_has_position_reset_none():
    """HasPosition position can be reset to None after being set."""
    model = Model()
    agent = PositionAgent(model)
    agent.position = (1.0, 2.0)
    agent.position = None
    assert agent.position is None


def test_has_position_independent_instances():
    """HasPosition position is instance-level, not shared across agents."""
    model = Model()
    a = PositionAgent(model)
    b = PositionAgent(model)
    a.position = np.array([1.0, 2.0])
    assert b.position is None


def test_has_position_satisfies_locatable():
    """Agent with HasPosition mixin satisfies the Locatable protocol."""
    model = Model()
    agent = PositionAgent(model)
    assert isinstance(agent, Locatable)


def test_cell_satisfies_locatable():
    """Cell satisfies Locatable and returns correct physical position as ndarray."""
    cell = Cell((3, 4), capacity=None, random=random.Random())
    assert isinstance(cell, Locatable)
    np.testing.assert_array_equal(cell.position, np.array([3.0, 4.0]))


def test_cell_agent_satisfies_locatable():
    """CellAgent satisfies the Locatable protocol."""
    model = Model()
    agent = CellAgent(model)
    assert isinstance(agent, Locatable)


def test_cell_agent_position_none_when_unplaced():
    """CellAgent position is None when not placed in any cell."""
    model = Model()
    agent = CellAgent(model)
    assert agent.position is None


def test_cell_agent_position_after_placement():
    """CellAgent position matches cell physical position after placement."""
    model = Model()
    agent = CellAgent(model)
    cell = Cell((1, 2), capacity=None, random=random.Random())
    agent.cell = cell
    np.testing.assert_array_equal(agent.position, np.array([1.0, 2.0]))


def test_fixed_agent_satisfies_locatable():
    """FixedAgent satisfies the Locatable protocol."""
    model = Model()
    agent = FixedAgent(model)
    assert isinstance(agent, Locatable)


def test_fixed_agent_position_after_placement():
    """FixedAgent position matches cell physical position after placement."""
    model = Model()
    agent = FixedAgent(model)
    cell = Cell((5, 6), capacity=None, random=random.Random())
    agent.cell = cell
    np.testing.assert_array_equal(agent.position, np.array([5.0, 6.0]))


def test_continuous_space_agent_satisfies_locatable():
    """ContinuousSpaceAgent satisfies the Locatable protocol."""
    model = Model()
    space = ContinuousSpace([[0, 10], [0, 10]], random=model.random)
    agent = ContinuousSpaceAgent(space, model)
    assert isinstance(agent, Locatable)


def test_continuous_space_agent_position_returns_ndarray():
    """ContinuousSpaceAgent position returns a numpy array."""
    model = Model()
    space = ContinuousSpace([[0, 10], [0, 10]], random=model.random)
    agent = ContinuousSpaceAgent(space, model)
    agent.position = np.array([5.0, 5.0])
    assert isinstance(agent.position, np.ndarray)
    np.testing.assert_array_equal(agent.position, np.array([5.0, 5.0]))


def test_cell_agent_position_updates_on_move():
    """CellAgent position updates correctly when agent moves to a new cell."""
    model = Model()
    agent = CellAgent(model)
    cell1 = Cell((1, 2), capacity=None, random=random.Random())
    cell2 = Cell((3, 4), capacity=None, random=random.Random())
    agent.cell = cell1
    np.testing.assert_array_equal(agent.position, np.array([1.0, 2.0]))
    agent.cell = cell2
    np.testing.assert_array_equal(agent.position, np.array([3.0, 4.0]))


def test_has_position_read_before_write():
    model = Model()
    a = PositionAgent(model)
    b = PositionAgent(model)
    assert a.position is None
    assert b.position is None


def test_locatable_position_access():
    model = Model()
    agent = PositionAgent(model)
    pos = agent.position
    assert pos is None
