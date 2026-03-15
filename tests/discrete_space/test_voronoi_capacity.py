"""Tests for VoronoiGrid capacity bug fix — Issue #3543.

Covers:
- Explicit capacity is preserved (not overwritten by area function)
- capacity=None still applies area-derived capacities
- CellFullException fires at runtime with explicit capacity
- Custom capacity_function is only used when capacity=None
"""

from __future__ import annotations

import random as stdlib_random

import pytest

from mesa import Model
from mesa.discrete_space.cell_agent import CellAgent
from mesa.discrete_space.voronoi import VoronoiGrid
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


def test_voronoi_explicit_capacity_is_preserved() -> None:
    """VoronoiGrid must NOT overwrite an explicit capacity with the area function."""
    grid = VoronoiGrid(CENTROIDS, capacity=5, random=RNG)
    for cell in grid._cells.values():
        assert cell.capacity == 5, (
            f"Cell {cell.coordinate} has capacity={cell.capacity!r}, expected 5. "
            "VoronoiGrid._build_cell_polygons is still overwriting user capacity."
        )


def test_voronoi_capacity_none_uses_area_function() -> None:
    """When capacity=None the default area-based capacity_function must apply."""
    grid = VoronoiGrid(CENTROIDS, capacity=None, random=RNG)
    for cell in grid._cells.values():
        assert cell.capacity is not None
        assert isinstance(cell.capacity, int)
        assert cell.capacity >= 0


def test_voronoi_explicit_capacity_enforced_at_runtime() -> None:
    """CellFullException must fire at runtime when an explicit capacity is set."""
    model = make_model()
    grid = VoronoiGrid(CENTROIDS, capacity=1, random=RNG)
    cell = next(iter(grid._cells.values()))
    bob, julie = make_agent(model), make_agent(model)
    bob.move_to(cell)
    with pytest.raises(CellFullException):
        julie.move_to(cell)


def test_voronoi_custom_capacity_function_only_used_when_capacity_none() -> None:
    """A custom capacity_function must be ignored when capacity is explicit."""
    always_ten: callable = lambda area: 10  # noqa: E731

    grid_explicit = VoronoiGrid(
        CENTROIDS, capacity=3, random=RNG, capacity_function=always_ten
    )
    grid_auto = VoronoiGrid(
        CENTROIDS, capacity=None, random=RNG, capacity_function=always_ten
    )

    for cell in grid_explicit._cells.values():
        assert cell.capacity == 3, (
            "Explicit capacity must take precedence over capacity_function."
        )
    for cell in grid_auto._cells.values():
        assert cell.capacity == 10, (
            "Custom capacity_function must be used when capacity=None."
        )


def test_voronoi_capacity_none_produces_int_capacities() -> None:
    """Default round_float function must produce non-negative int capacities."""
    grid = VoronoiGrid(CENTROIDS, capacity=None, random=RNG)
    for cell in grid._cells.values():
        assert isinstance(cell.capacity, int)
        assert cell.capacity >= 0
