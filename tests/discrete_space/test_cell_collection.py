"""Tests for the CellCollection class in discrete_space module."""

import random
import warnings

import pytest

from mesa.discrete_space.cell import Cell
from mesa.discrete_space.cell_agent import CellAgent
from mesa.discrete_space.cell_collection import CellCollection
from mesa.model import Model


class TestCellCollection:
    """Tests for the CellCollection class."""

    def test_collection_initialization_with_iterable(self):
        """Test CellCollection can be initialized with an iterable of cells."""
        rng = random.Random(42)
        cells = [Cell(coordinate=(i, 0), random=rng) for i in range(3)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            collection = CellCollection(cells)
        assert len(collection) == 3

    def test_collection_initialization_with_dict(self):
        """Test CellCollection can be initialized with a dict."""
        rng = random.Random(42)
        cell1 = Cell(coordinate=(0, 0), random=rng)
        cell2 = Cell(coordinate=(1, 0), random=rng)
        cells_dict = {cell1: cell1._agents, cell2: cell2._agents}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            collection = CellCollection(cells_dict)
        assert len(collection) == 2

    def test_collection_initialization_with_random(self):
        """Test CellCollection accepts random generator."""
        rng = random.Random(42)
        cells = [Cell(coordinate=(i, 0)) for i in range(3)]
        collection = CellCollection(cells, random=rng)
        assert collection.random is rng

    def test_collection_warns_without_random(self):
        """Test CellCollection warns when no random is provided."""
        cells = [Cell(coordinate=(i, 0)) for i in range(3)]
        with pytest.warns(UserWarning, match="Random number generator not specified"):
            collection = CellCollection(cells)

    def test_collection_iteration(self):
        """Test iterating over CellCollection."""
        rng = random.Random(42)
        cells = [Cell(coordinate=(i, 0), random=rng) for i in range(3)]
        collection = CellCollection(cells, random=rng)
        iterated_cells = list(collection)
        assert len(iterated_cells) == 3
        for cell in cells:
            assert cell in iterated_cells

    def test_collection_getitem(self):
        """Test __getitem__ returns agents for a cell."""
        rng = random.Random(42)
        cell = Cell(coordinate=(0, 0), random=rng)
        model = Model()
        agent = CellAgent(model)
        cell.add_agent(agent)
        
        collection = CellCollection([cell], random=rng)
        assert agent in collection[cell]

    def test_collection_len(self):
        """Test __len__ returns number of cells."""
        rng = random.Random(42)
        cells = [Cell(coordinate=(i, 0), random=rng) for i in range(5)]
        collection = CellCollection(cells, random=rng)
        assert len(collection) == 5

    def test_collection_repr(self):
        """Test __repr__ returns string representation."""
        rng = random.Random(42)
        cells = [Cell(coordinate=(0, 0), random=rng)]
        collection = CellCollection(cells, random=rng)
        repr_str = repr(collection)
        assert "CellCollection" in repr_str

    def test_collection_cells_property(self):
        """Test cells property returns list of cells."""
        rng = random.Random(42)
        cells = [Cell(coordinate=(i, 0), random=rng) for i in range(3)]
        collection = CellCollection(cells, random=rng)
        collection_cells = collection.cells
        assert len(collection_cells) == 3
        for cell in cells:
            assert cell in collection_cells

    def test_collection_agents_property(self):
        """Test agents property returns all agents across cells."""
        rng = random.Random(42)
        cell1 = Cell(coordinate=(0, 0), random=rng)
        cell2 = Cell(coordinate=(1, 0), random=rng)
        model = Model()
        agent1 = CellAgent(model)
        agent2 = CellAgent(model)
        cell1.add_agent(agent1)
        cell2.add_agent(agent2)
        
        collection = CellCollection([cell1, cell2], random=rng)
        agents = list(collection.agents)
        assert len(agents) == 2
        assert agent1 in agents
        assert agent2 in agents

    def test_collection_select_random_cell(self):
        """Test select_random_cell returns a cell from collection."""
        rng = random.Random(42)
        cells = [Cell(coordinate=(i, 0), random=rng) for i in range(3)]
        collection = CellCollection(cells, random=rng)
        
        selected = collection.select_random_cell()
        assert selected in cells

    def test_collection_select_random_agent(self):
        """Test select_random_agent returns an agent from collection."""
        rng = random.Random(42)
        cell = Cell(coordinate=(0, 0), random=rng)
        model = Model()
        agent = CellAgent(model)
        cell.add_agent(agent)
        
        collection = CellCollection([cell], random=rng)
        selected = collection.select_random_agent()
        assert selected is agent

    def test_collection_select_random_agent_empty_raises(self):
        """Test select_random_agent raises LookupError when empty."""
        rng = random.Random(42)
        cell = Cell(coordinate=(0, 0), random=rng)
        collection = CellCollection([cell], random=rng)
        
        with pytest.raises(LookupError, match="Cannot select random agent from empty"):
            collection.select_random_agent()

    def test_collection_select_random_agent_empty_with_default(self):
        """Test select_random_agent returns default when empty and default provided."""
        rng = random.Random(42)
        cell = Cell(coordinate=(0, 0), random=rng)
        collection = CellCollection([cell], random=rng)
        
        result = collection.select_random_agent(default=None)
        assert result is None

    def test_collection_select_no_filter(self):
        """Test select without filter returns self."""
        rng = random.Random(42)
        cells = [Cell(coordinate=(i, 0), random=rng) for i in range(3)]
        collection = CellCollection(cells, random=rng)
        
        selected = collection.select()
        assert selected is collection

    def test_collection_select_with_filter(self):
        """Test select with filter function."""
        rng = random.Random(42)
        cells = [Cell(coordinate=(i, 0), random=rng) for i in range(5)]
        collection = CellCollection(cells, random=rng)
        
        # Filter for cells with even x coordinate
        selected = collection.select(filter_func=lambda c: c.coordinate[0] % 2 == 0)
        assert len(selected) == 3  # 0, 2, 4

    def test_collection_select_with_at_most_int(self):
        """Test select with at_most as integer."""
        rng = random.Random(42)
        cells = [Cell(coordinate=(i, 0), random=rng) for i in range(10)]
        collection = CellCollection(cells, random=rng)
        
        selected = collection.select(at_most=3)
        assert len(selected) == 3

    def test_collection_select_with_at_most_float(self):
        """Test select with at_most as float fraction."""
        rng = random.Random(42)
        cells = [Cell(coordinate=(i, 0), random=rng) for i in range(10)]
        collection = CellCollection(cells, random=rng)
        
        selected = collection.select(at_most=0.5)
        assert len(selected) == 5

    def test_collection_select_combined(self):
        """Test select with both filter and at_most."""
        rng = random.Random(42)
        cells = [Cell(coordinate=(i, 0), random=rng) for i in range(10)]
        collection = CellCollection(cells, random=rng)
        
        # Filter for even coordinates, but limit to 2
        selected = collection.select(
            filter_func=lambda c: c.coordinate[0] % 2 == 0,
            at_most=2
        )
        assert len(selected) == 2
