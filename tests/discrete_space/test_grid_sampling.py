"""Unit tests for Grid.select_random_empty_cell heuristic behavior."""

import random

import pytest

from mesa.agent import Agent
from mesa.discrete_space.discrete_space import DiscreteSpace
from mesa.discrete_space.grid import OrthogonalMooreGrid
from mesa.model import Model


@pytest.fixture
def rng():
    """Fixture for a reproducible random number generator."""
    return random.Random(42)


@pytest.fixture
def model():
    """Fixture to provide a basic model instance."""
    return Model()


def test_select_random_empty_cell_full_grid(model, rng, mocker):
    """Test (1): Full grid triggers fallback to parent method."""
    grid = OrthogonalMooreGrid((5, 5), random=rng)
    # Fill all cells
    for cell in grid.all_cells:
        cell.add_agent(Agent(model))

    # Spy on the parent class method to ensure it gets called
    spy_fallback = mocker.spy(DiscreteSpace, "select_random_empty_cell")

    # Expect IndexError because the grid is full and fallback raises it
    with pytest.raises(IndexError):
        grid.select_random_empty_cell()

    # Verify fallback was indeed called
    spy_fallback.assert_called_once()


def test_select_random_empty_cell_sparse_grid(model, rng, mocker):
    """Test (2): Sparse grid returns cell quickly without fallback."""
    grid = OrthogonalMooreGrid((10, 10), random=rng)

    spy_fallback = mocker.spy(DiscreteSpace, "select_random_empty_cell")

    cell = grid.select_random_empty_cell()

    assert cell.is_empty
    # Fallback should NOT be called because random sampling should succeed
    spy_fallback.assert_not_called()


def test_select_random_empty_cell_nearly_full_grid(model, rng):
    """Test (3): Nearly-full grid eventually finds the cell."""
    grid = OrthogonalMooreGrid((5, 5), random=rng)
    # Fill all but one cell
    all_cells = list(grid.all_cells)
    empty_cell = all_cells[0]
    for cell in all_cells[1:]:
        cell.add_agent(Agent(model))

    # We don't spy here because behavior is probabilistic (fast path OR fallback)
    # We just care that it returns the correct cell successfully.
    selected_cell = grid.select_random_empty_cell()

    assert selected_cell == empty_cell
    assert selected_cell.is_empty


def test_select_random_empty_cell_disabled_random(model, rng, mocker):
    """Test (4): _try_random=False forces immediate fallback."""
    grid = OrthogonalMooreGrid((5, 5), random=rng)
    grid._try_random = False

    spy_fallback = mocker.spy(DiscreteSpace, "select_random_empty_cell")

    cell = grid.select_random_empty_cell()

    assert cell.is_empty
    # Must call fallback immediately
    spy_fallback.assert_called_once()
