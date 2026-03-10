"""Tests for Grid cell capacity enforcement — Issue #3505.

Covers:
- CellFullException raised by move_to and direct cell-property assignment
- available_cells property (new in this PR)
- select_random_available_cell() method (new in this PR)
- VoronoiGrid capacity-overwrite bugfix
- Cell.is_full edge cases
- Regression tests parametrized across OrthogonalMooreGrid,
  OrthogonalVonNeumannGrid, and HexGrid
"""

from __future__ import annotations

import random as stdlib_random

import pytest

from mesa import Model
from mesa.discrete_space import (
    Cell,
    OrthogonalMooreGrid,
    OrthogonalVonNeumannGrid,
)
from mesa.discrete_space.cell_agent import CellAgent
from mesa.discrete_space.grid import HexGrid
from mesa.discrete_space.voronoi import VoronoiGrid
from mesa.exceptions import CellFullException

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

RNG = stdlib_random.Random(42)

# All concrete Grid subclasses parametrized so every test runs on each type.
GRID_FACTORIES = [
    pytest.param(
        lambda cap: OrthogonalMooreGrid((4, 4), torus=False, capacity=cap, random=RNG),
        id="OrthogonalMooreGrid",
    ),
    pytest.param(
        lambda cap: OrthogonalVonNeumannGrid(
            (4, 4), torus=False, capacity=cap, random=RNG
        ),
        id="OrthogonalVonNeumannGrid",
    ),
    pytest.param(
        lambda cap: HexGrid((4, 4), torus=False, capacity=cap, random=RNG),
        id="HexGrid",
    ),
]

# Six-point Voronoi test fixture
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


# ---------------------------------------------------------------------------
# Section 1 — CellFullException  (regression tests for Issue #3505)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", GRID_FACTORIES)
def test_move_to_raises_when_cell_full(factory) -> None:
    """move_to must raise CellFullException once capacity=1 is occupied."""
    model = make_model()
    grid = factory(1)
    cell = grid._celllist[0]
    bob, julie = make_agent(model), make_agent(model)
    bob.move_to(cell)
    with pytest.raises(CellFullException):
        julie.move_to(cell)


@pytest.mark.parametrize("factory", GRID_FACTORIES)
def test_direct_cell_property_raises_when_full(factory) -> None:
    """Direct assignment ``agent.cell = <full cell>`` must also raise."""
    model = make_model()
    grid = factory(1)
    cell = grid._celllist[0]
    bob, julie = make_agent(model), make_agent(model)
    bob.cell = cell
    with pytest.raises(CellFullException):
        julie.cell = cell


@pytest.mark.parametrize("factory", GRID_FACTORIES)
def test_capacity_2_allows_exactly_two_agents(factory) -> None:
    """capacity=2: two agents fit; a third must be rejected."""
    model = make_model()
    grid = factory(2)
    cell = grid._celllist[0]
    agents = [make_agent(model) for _ in range(3)]
    agents[0].move_to(cell)
    agents[1].move_to(cell)
    with pytest.raises(CellFullException):
        agents[2].move_to(cell)


@pytest.mark.parametrize("factory", GRID_FACTORIES)
def test_unlimited_capacity_never_raises(factory) -> None:
    """capacity=None: no CellFullException no matter how many agents enter."""
    model = make_model()
    grid = factory(None)
    cell = grid._celllist[0]
    for _ in range(20):
        make_agent(model).move_to(cell)
    assert len(cell._agents) == 20


@pytest.mark.parametrize("factory", GRID_FACTORIES)
def test_vacating_agent_frees_capacity(factory) -> None:
    """After an agent leaves, the freed slot must accept a new occupant."""
    model = make_model()
    grid = factory(1)
    cell = grid._celllist[0]
    bob, julie = make_agent(model), make_agent(model)
    bob.move_to(cell)
    bob.cell = None
    julie.move_to(cell)
    assert julie in cell._agents


# ---------------------------------------------------------------------------
# Section 2 — available_cells property
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", GRID_FACTORIES)
def test_available_cells_full_grid_all_returned(factory) -> None:
    """On a fresh (empty) grid every cell is available."""
    grid = factory(2)
    assert len(list(grid.available_cells)) == len(grid._celllist)


@pytest.mark.parametrize("factory", GRID_FACTORIES)
def test_available_cells_excludes_full_cell(factory) -> None:
    """A cell filled to capacity must not appear in available_cells."""
    model = make_model()
    grid = factory(1)
    cell = grid._celllist[0]
    make_agent(model).move_to(cell)
    available = list(grid.available_cells)
    assert cell not in available
    assert len(available) == len(grid._celllist) - 1


@pytest.mark.parametrize("factory", GRID_FACTORIES)
def test_available_cells_includes_partially_filled_cell(factory) -> None:
    """A cell at capacity=3 holding 2 agents must still appear in available_cells."""
    model = make_model()
    grid = factory(3)
    cell = grid._celllist[0]
    for _ in range(2):
        make_agent(model).move_to(cell)
    assert cell in list(grid.available_cells)


@pytest.mark.parametrize("factory", GRID_FACTORIES)
def test_available_cells_unlimited_always_includes_cell(factory) -> None:
    """With capacity=None a cell is always available regardless of agent count."""
    model = make_model()
    grid = factory(None)
    cell = grid._celllist[0]
    for _ in range(50):
        make_agent(model).move_to(cell)
    assert cell in list(grid.available_cells)


@pytest.mark.parametrize("factory", GRID_FACTORIES)
def test_available_cells_recovers_after_agent_leaves(factory) -> None:
    """A full cell must re-appear in available_cells after an agent departs."""
    model = make_model()
    grid = factory(1)
    cell = grid._celllist[0]
    agent = make_agent(model)
    agent.move_to(cell)
    assert cell not in list(grid.available_cells)
    agent.cell = None
    assert cell in list(grid.available_cells)


# ---------------------------------------------------------------------------
# Section 3 — select_random_available_cell()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", GRID_FACTORIES)
def test_select_random_available_cell_not_full(factory) -> None:
    """Returned cell must have remaining capacity."""
    model = make_model()
    grid = factory(2)
    half = len(grid._celllist) // 2
    for cell in grid._celllist[:half]:
        for _ in range(2):
            make_agent(model).move_to(cell)
    chosen = grid.select_random_available_cell()
    assert not chosen.is_full


@pytest.mark.parametrize("factory", GRID_FACTORIES)
def test_select_random_available_cell_raises_when_all_full(factory) -> None:
    """IndexError must be raised when every cell is at capacity."""
    model = make_model()
    grid = factory(1)
    for cell in grid._celllist:
        make_agent(model).move_to(cell)
    with pytest.raises(IndexError, match="No available cells"):
        grid.select_random_available_cell()


@pytest.mark.parametrize("factory", GRID_FACTORIES)
def test_select_random_available_cell_consistent_with_available_cells(factory) -> None:
    """Every result from select_random_available_cell must be in available_cells."""
    model = make_model()
    grid = factory(3)
    for cell in grid._celllist[::3]:
        for _ in range(2):
            make_agent(model).move_to(cell)
    available_set = set(grid.available_cells)
    for _ in range(30):
        chosen = grid.select_random_available_cell()
        assert chosen in available_set


@pytest.mark.parametrize("factory", GRID_FACTORIES)
def test_select_random_available_cell_on_empty_grid(factory) -> None:
    """On a fresh grid any cell is a valid result."""
    grid = factory(5)
    chosen = grid.select_random_available_cell()
    assert chosen in grid._celllist
    assert not chosen.is_full


# ---------------------------------------------------------------------------
# Section 4 — VoronoiGrid capacity-overwrite bugfix
# ---------------------------------------------------------------------------


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


def test_voronoi_unlimited_capacity_none_gets_area_derived_int() -> None:
    """VoronoiGrid with capacity=None and default function must produce int capacities."""
    grid = VoronoiGrid(CENTROIDS, capacity=None, random=RNG)
    for cell in grid._cells.values():
        assert isinstance(cell.capacity, int)


# ---------------------------------------------------------------------------
# Section 5 — Cell.is_full unit tests
# ---------------------------------------------------------------------------


def test_cell_is_full_with_capacity_none() -> None:
    """is_full must always be False for a cell with unlimited capacity."""
    cell = Cell(coordinate=(0, 0), capacity=None, random=RNG)
    assert not cell.is_full


def test_cell_is_full_transitions_correctly() -> None:
    """is_full must reflect the exact agent count vs capacity boundary."""
    model = make_model()
    cell = Cell(coordinate=(0, 0), capacity=2, random=RNG)
    a1, a2 = make_agent(model), make_agent(model)

    assert not cell.is_full  # 0 / 2
    cell.add_agent(a1)
    assert not cell.is_full  # 1 / 2
    cell.add_agent(a2)
    assert cell.is_full  # 2 / 2
    cell.remove_agent(a1)
    assert not cell.is_full  # 1 / 2


def test_cell_add_agent_raises_when_full() -> None:
    """add_agent must raise CellFullException directly, not silently overflow."""
    model = make_model()
    cell = Cell(coordinate=(0, 0), capacity=1, random=RNG)
    cell.add_agent(make_agent(model))
    with pytest.raises(CellFullException):
        cell.add_agent(make_agent(model))
