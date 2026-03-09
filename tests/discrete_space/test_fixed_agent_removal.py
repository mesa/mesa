"""Tests for FixedAgent removal functionality and property consistency."""

import pytest

from mesa import Model
from mesa.discrete_space import Cell, FixedAgent


def test_fixed_agent_removal_via_public_api():
    """Verify that FixedAgent can now be removed via public API (agent.cell = None)."""
    model = Model()
    cell = Cell((0, 0))
    agent = FixedAgent(model)

    # Assign to cell
    agent.cell = cell
    assert agent.cell == cell
    assert agent in cell.agents

    # Now this should work!
    agent.cell = None
    assert agent.cell is None
    assert agent not in cell.agents


def test_fixed_agent_remove_method_works():
    """Verify that the .remove() method still works and uses the new logic."""
    model = Model()
    cell = Cell((0, 0))
    agent = FixedAgent(model)
    agent.cell = cell

    assert agent in cell.agents

    agent.remove()

    assert agent.cell is None
    assert agent not in cell.agents
    assert agent not in model.agents


def test_prevent_teleportation_loophole():
    """Verify that a FixedAgent cannot be moved by setting to None and then to a new cell."""
    model = Model()
    cell1 = Cell((0, 0))
    cell2 = Cell((0, 1))
    agent = FixedAgent(model)

    # 1. Place in first cell
    agent.cell = cell1
    assert agent.cell == cell1

    # 2. Remove from space
    agent.cell = None
    assert agent.cell is None
    assert agent not in cell1.agents

    # 3. Attempt to place in a NEW cell (This should FAIL and raise ValueError)
    with pytest.raises(ValueError, match="Cannot move agent in FixedCell"):
        agent.cell = cell2

    # Verify it's still not in either cell
    assert agent.cell is None
    assert agent not in cell1.agents
    assert agent not in cell2.agents
