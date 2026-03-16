"""Tests for VoronoiGrid capacity fix — Issue #3543."""

from __future__ import annotations

import random as stdlib_random

import pytest

from mesa import Model
from mesa.discrete_space.cell_agent import CellAgent
from mesa.discrete_space.voronoi import VoronoiGrid, round_float
from mesa.exceptions import CellFullException

RNG = stdlib_random.Random(42)

CENTROIDS: list[list[float]] = [
    [0.0, 0.0],
    [1.0, 0.0],
    [0.5, 1.0],
    [1.5, 1.0],
    [0.0, 2.0],
    [1.0, 2.0],
]


def make_model() -> Model:
    """Return a freshly seeded Model instance."""
    return Model(rng=42)


def make_agent(model: Model) -> CellAgent:
    """Return a new CellAgent attached to model."""
    return CellAgent(model)


def test_voronoi_default_no_capacity() -> None:
    """Default VoronoiGrid (no capacity, no function) leaves cells with None capacity."""
    grid = VoronoiGrid(CENTROIDS, random=RNG)
    for cell in grid._cells.values():
        assert cell.capacity is None, (
            f"Cell {cell.coordinate} has capacity={cell.capacity!r}, expected None. "
            "Default should not assign any capacity."
        )


def test_voronoi_explicit_capacity_is_preserved() -> None:
    """Explicit capacity must be used directly on every cell."""
    grid = VoronoiGrid(CENTROIDS, capacity=5, random=RNG)
    for cell in grid._cells.values():
        assert cell.capacity == 5, (
            f"Cell {cell.coordinate} has capacity={cell.capacity!r}, expected 5."
        )


def test_voronoi_capacity_function_used_when_no_capacity() -> None:
    """capacity_function is applied per-cell when capacity=None."""
    grid = VoronoiGrid(
        CENTROIDS, capacity=None, capacity_function=round_float, random=RNG
    )
    for cell in grid._cells.values():
        assert cell.capacity is not None
        assert isinstance(cell.capacity, int)
        assert cell.capacity >= 0


def test_voronoi_explicit_capacity_enforced_at_runtime() -> None:
    """CellFullException must fire at runtime when capacity=1."""
    model = make_model()
    grid = VoronoiGrid(CENTROIDS, capacity=1, random=RNG)
    cell = next(iter(grid._cells.values()))
    bob, julie = make_agent(model), make_agent(model)
    bob.move_to(cell)
    with pytest.raises(CellFullException):
        julie.move_to(cell)


def test_voronoi_custom_capacity_function_applied() -> None:
    """Custom capacity_function is applied when capacity=None."""
    always_ten: callable = lambda area: 10  # noqa: E731
    grid = VoronoiGrid(
        CENTROIDS, capacity=None, capacity_function=always_ten, random=RNG
    )
    for cell in grid._cells.values():
        assert cell.capacity == 10


def test_voronoi_raises_when_both_capacity_and_function_provided() -> None:
    """Providing both capacity and capacity_function must raise ValueError."""
    always_ten: callable = lambda area: 10  # noqa: E731
    with pytest.raises(ValueError, match="Cannot provide both"):
        VoronoiGrid(CENTROIDS, capacity=3, capacity_function=always_ten, random=RNG)


def test_voronoi_capacity_none_and_no_function_allows_unlimited() -> None:
    """With no capacity and no function, cells accept unlimited agents."""
    model = make_model()
    grid = VoronoiGrid(CENTROIDS, random=RNG)
    cell = next(iter(grid._cells.values()))
    for _ in range(10):
        make_agent(model).move_to(cell)
    assert len(cell._agents) == 10
