"""Tests for experimental Simulator classes."""

from unittest.mock import MagicMock, Mock

import pytest

from mesa import Model
from mesa.experimental.devs.simulator import ABMSimulator, DEVSimulator
from mesa.time import (
    Event,
    Priority,
)

# Ignore deprecation warnings for Simulator classes in this test file
pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


def test_devs_simulator():
    """Tests devs simulator."""
    simulator = DEVSimulator()

    # setup
    model = Model()
    simulator.setup(model)

    assert len(simulator.event_list) == 0
    assert simulator.model == model
    assert model.time == 0.0

    # schedule_event_now
    fn1 = MagicMock()
    event1 = simulator.schedule_event_now(fn1)
    assert event1 in simulator.event_list
    assert len(simulator.event_list) == 1

    # schedule_event_absolute
    fn2 = MagicMock()
    event2 = simulator.schedule_event_absolute(fn2, 1.0)
    assert event2 in simulator.event_list
    assert len(simulator.event_list) == 2

    # schedule_event_relative
    fn3 = MagicMock()
    event3 = simulator.schedule_event_relative(fn3, 0.5)
    assert event3 in simulator.event_list
    assert len(simulator.event_list) == 3

    # run_for
    simulator.run_for(0.8)
    fn1.assert_called_once()
    fn3.assert_called_once()
    assert model.time == 0.8

    simulator.run_for(0.2)
    fn2.assert_called_once()
    assert model.time == 1.0

    simulator.run_for(0.2)
    assert model.time == 1.2

    with pytest.raises(ValueError):
        simulator.schedule_event_absolute(fn2, 0.5)

    # schedule_event_relative with negative time_delta (causality violation)
    with pytest.raises(ValueError, match="Cannot schedule event in the past"):
        simulator.schedule_event_relative(fn2, -0.5)

    # step
    simulator = DEVSimulator()
    model = Model()
    simulator.setup(model)

    fn = MagicMock()
    simulator.schedule_event_absolute(fn, 1.0)
    simulator.run_next_event()
    fn.assert_called_once()
    assert model.time == 1.0
    simulator.run_next_event()
    assert model.time == 1.0

    simulator = DEVSimulator()
    with pytest.raises(Exception):
        simulator.run_next_event()

    # cancel_event
    simulator = DEVSimulator()
    model = Model()
    simulator.setup(model)
    fn = MagicMock()
    event = simulator.schedule_event_relative(fn, 0.5)
    simulator.cancel_event(event)
    assert event.CANCELED

    # simulator reset
    simulator.reset()
    assert len(simulator.event_list) == 0
    assert simulator.model is model

    # run_for without setup
    simulator = DEVSimulator()
    with pytest.raises(RuntimeError, match="Simulator not set up"):
        simulator.run_for(1.0)

    # run_until without setup
    simulator = DEVSimulator()
    with pytest.raises(Exception):
        simulator.run_until(10)

    # setup with time advanced
    simulator = DEVSimulator()
    model = Model()
    model.time = 1.0  # Advance time before setup
    with pytest.raises(ValueError):
        simulator.setup(model)

    # setup with event scheduled
    simulator = DEVSimulator()
    with pytest.raises(RuntimeError, match="Simulator not set up"):
        simulator.event_list.add_event(Event(1.0, Mock(), Priority.DEFAULT))


def test_abm_simulator():
    """Tests abm simulator."""
    simulator = ABMSimulator()

    # setup
    model = Model()
    simulator.setup(model)

    # schedule_event_next_tick
    fn = MagicMock()
    simulator.schedule_event_next_tick(fn)
    assert len(simulator.event_list) == 2

    simulator.run_for(3)
    assert model.time == 3.0

    # run_until without setup
    simulator = ABMSimulator()
    with pytest.raises(Exception):
        simulator.run_until(10)

    # run_for without setup
    simulator = ABMSimulator()
    with pytest.raises(RuntimeError, match="Simulator not set up"):
        simulator.run_for(3)


def test_simulator_time_deprecation():
    """Test that simulator.time emits future warning."""
    simulator = DEVSimulator()
    model = Model()
    simulator.setup(model)

    with pytest.warns(FutureWarning, match="simulator.time is deprecated"):
        _ = simulator.time


def test_simulator_uses_model_event_list():
    """Test that simulator uses model's internal event list."""
    model = Model()
    simulator = DEVSimulator()
    simulator.setup(model)

    # Simulator's event_list property should return model's event list
    assert simulator.event_list is model._event_list

    # Events scheduled through simulator appear in model's event list
    fn = MagicMock()
    simulator.schedule_event_absolute(fn, 1.0)
    assert len(model._event_list) == 1
