"""Tests for the Cell class in discrete_space module."""

import random

import pytest

from mesa.discrete_space.cell import Cell
from mesa.discrete_space.cell_agent import CellAgent
from mesa.model import Model


class TestCell:
    """Tests for the Cell class."""

    def test_cell_initialization(self):
        """Test Cell initializes with correct attributes."""
        cell = Cell(coordinate=(1, 2))
        assert cell.coordinate == (1, 2)
        assert cell.capacity is None
        assert cell.connections == {}
        assert cell._agents == []
        assert cell.random is None

    def test_cell_initialization_with_capacity(self):
        """Test Cell initializes with capacity."""
        cell = Cell(coordinate=(0, 0), capacity=5)
        assert cell.capacity == 5

    def test_cell_initialization_with_random(self):
        """Test Cell initializes with random generator."""
        rng = random.Random(42)
        cell = Cell(coordinate=(0, 0), random=rng)
        assert cell.random is rng

    def test_cell_connect(self):
        """Test connecting two cells."""
        cell1 = Cell(coordinate=(0, 0))
        cell2 = Cell(coordinate=(0, 1))
        cell1.connect(cell2)
        assert (0, 1) in cell1.connections
        assert cell1.connections[(0, 1)] is cell2

    def test_cell_connect_with_custom_key(self):
        """Test connecting cells with a custom key."""
        cell1 = Cell(coordinate=(0, 0))
        cell2 = Cell(coordinate=(0, 1))
        cell1.connect(cell2, key=(0, 1))
        assert (0, 1) in cell1.connections

    def test_cell_disconnect(self):
        """Test disconnecting cells."""
        cell1 = Cell(coordinate=(0, 0))
        cell2 = Cell(coordinate=(0, 1))
        cell1.connect(cell2)
        cell1.disconnect(cell2)
        assert len(cell1.connections) == 0

    def test_cell_is_empty_initially(self):
        """Test cell is empty when created."""
        cell = Cell(coordinate=(0, 0))
        assert cell.is_empty is True

    def test_cell_is_full_with_no_capacity(self):
        """Test cell is never full with no capacity limit."""
        cell = Cell(coordinate=(0, 0))
        assert cell.is_full is False

    def test_cell_is_full_with_capacity(self):
        """Test cell is full when at capacity."""
        cell = Cell(coordinate=(0, 0), capacity=1)
        model = Model()
        agent = CellAgent(model)
        agent.cell = cell
        assert cell.is_full is True

    def test_cell_add_agent(self):
        """Test adding agent to cell."""
        cell = Cell(coordinate=(0, 0))
        model = Model()
        agent = CellAgent(model)
        cell.add_agent(agent)
        assert agent in cell._agents
        assert cell.is_empty is False

    def test_cell_add_agent_exceeds_capacity(self):
        """Test adding agent to full cell raises exception."""
        cell = Cell(coordinate=(0, 0), capacity=1)
        model = Model()
        agent1 = CellAgent(model)
        agent2 = CellAgent(model)
        cell.add_agent(agent1)
        with pytest.raises(Exception, match="Cell is full"):
            cell.add_agent(agent2)

    def test_cell_remove_agent(self):
        """Test removing agent from cell."""
        cell = Cell(coordinate=(0, 0))
        model = Model()
        agent = CellAgent(model)
        cell.add_agent(agent)
        cell.remove_agent(agent)
        assert agent not in cell._agents
        assert cell.is_empty is True

    def test_cell_agents_property_returns_copy(self):
        """Test agents property returns a copy of the list."""
        cell = Cell(coordinate=(0, 0))
        model = Model()
        agent = CellAgent(model)
        cell.add_agent(agent)
        agents = cell.agents
        agents.clear()
        assert len(cell._agents) == 1

    def test_cell_repr(self):
        """Test Cell __repr__ method."""
        cell = Cell(coordinate=(1, 2))
        repr_str = repr(cell)
        assert "Cell" in repr_str
        assert "(1, 2)" in repr_str

    def test_cell_neighborhood(self):
        """Test cell neighborhood property."""
        rng = random.Random(42)
        cell1 = Cell(coordinate=(0, 0), random=rng)
        cell2 = Cell(coordinate=(0, 1), random=rng)
        cell3 = Cell(coordinate=(1, 0), random=rng)
        cell1.connect(cell2)
        cell1.connect(cell3)

        neighborhood = cell1.neighborhood
        assert cell2 in neighborhood
        assert cell3 in neighborhood

    def test_cell_get_neighborhood_radius(self):
        """Test get_neighborhood with different radii."""
        rng = random.Random(42)
        cell1 = Cell(coordinate=(0, 0), random=rng)
        cell2 = Cell(coordinate=(0, 1), random=rng)
        cell3 = Cell(coordinate=(0, 2), random=rng)
        cell1.connect(cell2)
        cell2.connect(cell3)

        # Radius 1 should only include direct neighbors
        neighborhood1 = cell1.get_neighborhood(radius=1)
        assert cell2 in neighborhood1
        assert cell3 not in neighborhood1

        # Radius 2 should include neighbors of neighbors
        neighborhood2 = cell1.get_neighborhood(radius=2)
        assert cell2 in neighborhood2
        assert cell3 in neighborhood2

    def test_cell_get_neighborhood_include_center(self):
        """Test get_neighborhood with include_center option."""
        rng = random.Random(42)
        cell1 = Cell(coordinate=(0, 0), random=rng)
        cell2 = Cell(coordinate=(0, 1), random=rng)
        cell1.connect(cell2)

        neighborhood_without_center = cell1.get_neighborhood(include_center=False)
        assert cell1 not in neighborhood_without_center

        neighborhood_with_center = cell1.get_neighborhood(include_center=True)
        assert cell1 in neighborhood_with_center

    def test_cell_get_neighborhood_invalid_radius(self):
        """Test get_neighborhood with invalid radius raises ValueError."""
        cell = Cell(coordinate=(0, 0), random=random.Random(42))
        with pytest.raises(ValueError, match="radius must be larger than one"):
            cell.get_neighborhood(radius=0)

    def test_cell_getstate(self):
        """Test __getstate__ for pickling."""
        cell1 = Cell(coordinate=(0, 0))
        cell2 = Cell(coordinate=(0, 1))
        cell1.connect(cell2)

        state = cell1.__getstate__()
        # Connections should be empty in state to avoid infinite recursion
        assert state[1]["connections"] == {}

    def test_cell_clear_cache(self):
        """Test _clear_cache method."""
        rng = random.Random(42)
        cell1 = Cell(coordinate=(0, 0), random=rng)
        cell2 = Cell(coordinate=(0, 1), random=rng)
        cell1.connect(cell2)

        # Access neighborhood to populate cache
        _ = cell1.neighborhood

        # Clear cache should not raise
        cell1._clear_cache()
