"""Tests for the CellAgent classes in discrete_space module."""

import pytest

from mesa.discrete_space.cell import Cell
from mesa.discrete_space.cell_agent import (
    BasicMovement,
    CellAgent,
    FixedAgent,
    FixedCell,
    Grid2DMovingAgent,
    HasCell,
)
from mesa.model import Model


class TestHasCell:
    """Tests for the HasCell mixin."""

    def test_has_cell_initial_state(self):
        """Test HasCell starts with no cell."""
        model = Model()
        agent = CellAgent(model)
        assert agent.cell is None

    def test_has_cell_setter(self):
        """Test setting cell property."""
        model = Model()
        agent = CellAgent(model)
        cell = Cell(coordinate=(0, 0))
        agent.cell = cell
        assert agent.cell is cell
        assert agent in cell._agents

    def test_has_cell_move_to_new_cell(self):
        """Test moving agent from one cell to another."""
        model = Model()
        agent = CellAgent(model)
        cell1 = Cell(coordinate=(0, 0))
        cell2 = Cell(coordinate=(1, 0))
        
        agent.cell = cell1
        assert agent in cell1._agents
        
        agent.cell = cell2
        assert agent not in cell1._agents
        assert agent in cell2._agents

    def test_has_cell_set_to_none(self):
        """Test setting cell to None removes agent from cell."""
        model = Model()
        agent = CellAgent(model)
        cell = Cell(coordinate=(0, 0))
        agent.cell = cell
        agent.cell = None
        assert agent not in cell._agents
        assert agent.cell is None


class TestBasicMovement:
    """Tests for the BasicMovement mixin."""

    def test_move_to(self):
        """Test move_to method."""
        model = Model()
        agent = CellAgent(model)
        cell = Cell(coordinate=(0, 0))
        agent.move_to(cell)
        assert agent.cell is cell

    def test_move_relative_valid(self):
        """Test move_relative with valid direction."""
        model = Model()
        agent = CellAgent(model)
        cell1 = Cell(coordinate=(0, 0))
        cell2 = Cell(coordinate=(0, 1))
        cell1.connect(cell2, key=(0, 1))
        
        agent.cell = cell1
        agent.move_relative((0, 1))
        assert agent.cell is cell2

    def test_move_relative_invalid(self):
        """Test move_relative with invalid direction raises ValueError."""
        model = Model()
        agent = CellAgent(model)
        cell = Cell(coordinate=(0, 0))
        agent.cell = cell
        
        with pytest.raises(ValueError, match="No cell in direction"):
            agent.move_relative((1, 0))


class TestFixedCell:
    """Tests for the FixedCell mixin."""

    def test_fixed_cell_initial_placement(self):
        """Test FixedAgent can be placed initially."""
        model = Model()
        agent = FixedAgent(model)
        cell = Cell(coordinate=(0, 0))
        agent.cell = cell
        assert agent.cell is cell
        assert agent in cell._agents

    def test_fixed_cell_cannot_move(self):
        """Test FixedAgent cannot be moved once placed."""
        model = Model()
        agent = FixedAgent(model)
        cell1 = Cell(coordinate=(0, 0))
        cell2 = Cell(coordinate=(1, 0))
        agent.cell = cell1
        
        with pytest.raises(ValueError, match="Cannot move agent"):
            agent.cell = cell2


class TestCellAgent:
    """Tests for the CellAgent class."""

    def test_cell_agent_creation(self):
        """Test CellAgent can be created."""
        model = Model()
        agent = CellAgent(model)
        assert agent.cell is None

    def test_cell_agent_remove(self):
        """Test removing CellAgent also removes from cell."""
        model = Model()
        agent = CellAgent(model)
        cell = Cell(coordinate=(0, 0))
        agent.cell = cell
        
        agent.remove()
        assert agent not in cell._agents
        assert agent.cell is None


class TestFixedAgent:
    """Tests for the FixedAgent class."""

    def test_fixed_agent_creation(self):
        """Test FixedAgent can be created."""
        model = Model()
        agent = FixedAgent(model)
        assert agent.cell is None

    def test_fixed_agent_remove(self):
        """Test removing FixedAgent also removes from cell."""
        model = Model()
        agent = FixedAgent(model)
        cell = Cell(coordinate=(0, 0))
        agent.cell = cell
        
        agent.remove()
        assert agent not in cell._agents


class TestGrid2DMovingAgent:
    """Tests for the Grid2DMovingAgent class."""

    def test_direction_map_exists(self):
        """Test DIRECTION_MAP contains expected directions."""
        assert "n" in Grid2DMovingAgent.DIRECTION_MAP
        assert "north" in Grid2DMovingAgent.DIRECTION_MAP
        assert "up" in Grid2DMovingAgent.DIRECTION_MAP
        assert "s" in Grid2DMovingAgent.DIRECTION_MAP
        assert "e" in Grid2DMovingAgent.DIRECTION_MAP
        assert "w" in Grid2DMovingAgent.DIRECTION_MAP
        assert "ne" in Grid2DMovingAgent.DIRECTION_MAP
        assert "sw" in Grid2DMovingAgent.DIRECTION_MAP

    def test_direction_map_values(self):
        """Test DIRECTION_MAP contains correct values."""
        assert Grid2DMovingAgent.DIRECTION_MAP["n"] == (-1, 0)
        assert Grid2DMovingAgent.DIRECTION_MAP["s"] == (1, 0)
        assert Grid2DMovingAgent.DIRECTION_MAP["e"] == (0, 1)
        assert Grid2DMovingAgent.DIRECTION_MAP["w"] == (0, -1)

    def test_move_valid_direction(self):
        """Test move with valid cardinal direction."""
        model = Model()
        agent = Grid2DMovingAgent(model)
        cell1 = Cell(coordinate=(1, 1))
        cell2 = Cell(coordinate=(0, 1))  # north of cell1
        cell1.connect(cell2, key=(-1, 0))
        
        agent.cell = cell1
        agent.move("north")
        assert agent.cell is cell2

    def test_move_case_insensitive(self):
        """Test move is case insensitive."""
        model = Model()
        agent = Grid2DMovingAgent(model)
        cell1 = Cell(coordinate=(1, 1))
        cell2 = Cell(coordinate=(0, 1))
        cell1.connect(cell2, key=(-1, 0))
        
        agent.cell = cell1
        agent.move("NORTH")
        assert agent.cell is cell2

    def test_move_invalid_direction(self):
        """Test move with invalid direction raises ValueError."""
        model = Model()
        agent = Grid2DMovingAgent(model)
        cell = Cell(coordinate=(0, 0))
        agent.cell = cell
        
        with pytest.raises(ValueError, match="Invalid direction"):
            agent.move("invalid_direction")

    def test_move_with_distance(self):
        """Test move with distance parameter."""
        model = Model()
        agent = Grid2DMovingAgent(model)
        cell1 = Cell(coordinate=(2, 0))
        cell2 = Cell(coordinate=(1, 0))
        cell3 = Cell(coordinate=(0, 0))
        cell1.connect(cell2, key=(-1, 0))
        cell2.connect(cell3, key=(-1, 0))
        
        agent.cell = cell1
        agent.move("north", distance=2)
        assert agent.cell is cell3
