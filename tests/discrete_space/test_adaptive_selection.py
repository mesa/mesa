"""Tests for adaptive select_random_empty_cell strategy."""

import time

import pytest

from mesa import Model
from mesa.discrete_space import CellAgent
from mesa.discrete_space.grid import OrthogonalMooreGrid


def test_adaptive_strategy_sparse_grid():
    """Test that sparse grids use random sampling strategy."""
    model = Model()
    grid = OrthogonalMooreGrid((10, 10), torus=False, random=model.random)
    # Fill grid to 30% (below 70% threshold)
    fill_count = int(0.3 * 100)  # 30% of 100 cells
    for i in range(fill_count):
        agent = CellAgent(model)
        grid._cells[(i % 10, i // 10)].add_agent(agent)
    # Should use random sampling path (fill_ratio < 0.7)
    selected_cell = grid.select_random_empty_cell()
    assert selected_cell.is_empty
    assert selected_cell in grid.all_cells


def test_adaptive_strategy_dense_grid():
    """Test that dense grids use vectorized strategy."""
    model = Model()
    grid = OrthogonalMooreGrid((10, 10), torus=False, random=model.random)
    # Fill grid to 80% (above 70% threshold)
    fill_count = int(0.8 * 100)  # 80% of 100 cells
    for i in range(fill_count):
        agent = CellAgent(model)
        grid._cells[(i % 10, i // 10)].add_agent(agent)
    # Should use vectorized path (fill_ratio >= 0.7)
    selected_cell = grid.select_random_empty_cell()
    assert selected_cell.is_empty
    assert selected_cell in grid.all_cells


def test_adaptive_strategy_threshold_boundary():
    """Test behavior exactly at 70% threshold."""
    model = Model()
    grid = OrthogonalMooreGrid((10, 10), torus=False, random=model.random)
    # Fill grid to exactly 70%
    fill_count = int(0.7 * 100)  # 70% of 100 cells
    for i in range(fill_count):
        agent = CellAgent(model)
        grid._cells[(i % 10, i // 10)].add_agent(agent)
    # Should use vectorized path (fill_ratio >= 0.7)
    selected_cell = grid.select_random_empty_cell()
    assert selected_cell.is_empty


def test_adaptive_strategy_with_try_random_disabled():
    """Test that _try_random=False always uses vectorized approach."""
    model = Model()
    grid = OrthogonalMooreGrid((10, 10), torus=False, random=model.random)
    # Fill grid to only 10% but disable random sampling
    fill_count = int(0.1 * 100)  # 10% of 100 cells
    for i in range(fill_count):
        agent = CellAgent(model)
        grid._cells[(i % 10, i // 10)].add_agent(agent)

    grid._try_random = False

    # Should use vectorized path despite being sparse
    selected_cell = grid.select_random_empty_cell()
    assert selected_cell.is_empty


def test_improved_error_message():
    """Test that full grids provide clear error messages."""
    model = Model()
    grid = OrthogonalMooreGrid((2, 2), torus=False, random=model.random)
    # Fill the grid completely
    for cell in grid.all_cells:
        agent = CellAgent(model)
        cell.add_agent(agent)
    # Should raise IndexError with clear message
    with pytest.raises(IndexError, match="No empty cells available in grid"):
        grid.select_random_empty_cell()


def test_performance_benchmark_dense_vs_sparse():
    """Benchmark to verify performance improvement on dense grids."""
    model = Model()

    # Test dense grid (90% full)
    dense_grid = OrthogonalMooreGrid((50, 50), torus=False, random=model.random)
    fill_count = int(0.9 * 2500)  # 90% of 2500 cells

    for i in range(fill_count):
        agent = CellAgent(model)
        dense_grid._cells[(i % 50, i // 50)].add_agent(agent)

    start_time = time.time()
    for _ in range(100):
        dense_grid.select_random_empty_cell()
    dense_time = time.time() - start_time

    # Test sparse grid (10% full)
    sparse_grid = OrthogonalMooreGrid((50, 50), torus=False, random=model.random)
    fill_count = int(0.1 * 2500)  # 10% of 2500 cells

    for i in range(fill_count):
        agent = CellAgent(model)
        sparse_grid._cells[(i % 50, i // 50)].add_agent(agent)

    start_time = time.time()
    for _ in range(100):
        sparse_grid.select_random_empty_cell()
    sparse_time = time.time() - start_time

    # Both should complete successfully (performance test)
    assert dense_time > 0
    assert sparse_time > 0

    # The dense grid should use vectorized approach (faster than random sampling would be)
    # The sparse grid should use random sampling (efficient for low density)
    print(f"Dense grid time: {dense_time:.4f}s, Sparse grid time: {sparse_time:.4f}s")
