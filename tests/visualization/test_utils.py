"""Tests for the visualization utils module."""

import pytest


def test_update_counter_exists():
    """Test that update_counter reactive variable exists."""
    from mesa.visualization.utils import update_counter

    # update_counter should be a solara reactive
    assert hasattr(update_counter, "value")


def test_update_counter_initial_value():
    """Test that update_counter starts at 0."""
    import importlib

    import mesa.visualization.utils as utils_module

    # Reimport to get fresh state
    importlib.reload(utils_module)
    assert utils_module.update_counter.value == 0


def test_force_update_increments_counter():
    """Test that force_update increments the update counter."""
    import importlib

    import mesa.visualization.utils as utils_module

    # Reimport to get fresh state
    importlib.reload(utils_module)

    initial_value = utils_module.update_counter.value
    utils_module.force_update()
    assert utils_module.update_counter.value == initial_value + 1


def test_force_update_multiple_calls():
    """Test that multiple force_update calls increment the counter each time."""
    import importlib

    import mesa.visualization.utils as utils_module

    # Reimport to get fresh state
    importlib.reload(utils_module)

    initial_value = utils_module.update_counter.value
    utils_module.force_update()
    utils_module.force_update()
    utils_module.force_update()
    assert utils_module.update_counter.value == initial_value + 3
