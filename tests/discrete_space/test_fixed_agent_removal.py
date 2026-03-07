"""Tests for FixedAgent removal functionality and property consistency."""

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
