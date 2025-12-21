"""Tests for Cell.is_full property with capacity=None fix (Issue #2980)."""

import pytest

from mesa import Model
from mesa.discrete_space import Cell
from mesa.discrete_space.cell_agent import CellAgent


def test_cell_is_full_with_none_capacity():
    """Test that is_full correctly handles capacity=None (infinite capacity)."""
    cell = Cell((0, 0), capacity=None)
    
    assert cell.is_full is False
    
    model = Model()
    for _ in range(100):
        agent = CellAgent(model)
        agent._mesa_cell = cell
        cell._agents.append(agent)
    
    assert cell.is_full is False


def test_cell_is_full_with_finite_capacity():
    """Test that is_full works correctly with finite capacity."""
    cell = Cell((0, 0), capacity=3)
    model = Model()
    
    assert cell.is_full is False
    
    agent1 = CellAgent(model)
    agent1._mesa_cell = cell
    cell._agents.append(agent1)
    assert cell.is_full is False
    
    agent2 = CellAgent(model)
    agent2._mesa_cell = cell
    cell._agents.append(agent2)
    assert cell.is_full is False
    
    agent3 = CellAgent(model)
    agent3._mesa_cell = cell
    cell._agents.append(agent3)
    assert cell.is_full is True


def test_cell_is_full_with_capacity_exceeded():
    """Test is_full when agents exceed capacity (edge case)."""
    cell = Cell((0, 0), capacity=2)
    model = Model()
    
    for _ in range(3):
        agent = CellAgent(model)
        agent._mesa_cell = cell
        cell._agents.append(agent)
    
    assert cell.is_full is True
    assert cell.is_full is True
