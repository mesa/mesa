"""Tests for mesa.protocols (Locatable, HasPosition, PositionLike)."""

import random

import numpy as np
import pytest

from mesa import Agent, Model
from mesa.discrete_space import Cell, CellAgent, FixedAgent
from mesa.protocols import HasPosition, Locatable


class PositionAgent(Agent, HasPosition):
    """Test agent using the HasPosition mixin."""


def test_has_position_default():
    """HasPosition starts as None."""
    model = Model()
    agent = PositionAgent(model)
    assert agent.position is None


def test_has_position_set_tuple():
    """HasPosition accepts tuple positions."""
    model = Model()
    agent = PositionAgent(model)

    agent.position = (5.0, 3.0)
    assert agent.position == (5.0, 3.0)

    agent.position = (1, 2, 3)
    assert agent.position == (1, 2, 3)


def test_has_position_set_ndarray():
    """HasPosition accepts numpy array positions."""
    model = Model()
    agent = PositionAgent(model)

    pos = np.array([1.5, 2.5])
    agent.position = pos
    np.testing.assert_array_equal(agent.position, pos)


def test_has_position_set_none():
    """HasPosition can be reset to None."""
    model = Model()
    agent = PositionAgent(model)

    agent.position = (1.0, 2.0)
    agent.position = None
    assert agent.position is None


def test_has_position_rejects_invalid_type():
    """HasPosition rejects non-tuple/non-ndarray types."""
    model = Model()
    agent = PositionAgent(model)

    with pytest.raises(TypeError, match="position must be a tuple or numpy array"):
        agent.position = [1, 2]

    with pytest.raises(TypeError, match="position must be a tuple or numpy array"):
        agent.position = "invalid"


def test_locatable_protocol_has_position():
    """HasPosition agents satisfy the Locatable protocol."""
    model = Model()
    agent = PositionAgent(model)
    assert isinstance(agent, Locatable)


def test_locatable_protocol_cell_agent():
    """CellAgent satisfies the Locatable protocol."""
    model = Model()
    agent = CellAgent(model)
    assert isinstance(agent, Locatable)


def test_locatable_protocol_fixed_agent():
    """FixedAgent satisfies the Locatable protocol."""
    model = Model()
    agent = FixedAgent(model)
    assert isinstance(agent, Locatable)


def test_locatable_protocol_cell():
    """Cell satisfies the Locatable protocol."""
    cell = Cell((3, 4), capacity=None, random=random.Random())
    assert isinstance(cell, Locatable)
