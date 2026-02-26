"""Tests for the expanded Mesa exception hierarchy."""

import numpy as np
import pytest

from mesa.agent import Agent
from mesa.discrete_space import OrthogonalMooreGrid
from mesa.exceptions import (
    CallbackTypeError,
    CallbackValueError,
    EmptyEventListException,
    InvalidOptionException,
    InvalidScheduleException,
    PastEventException,
    RNGMismatchException,
    UnsupportedBackendException,
    UnsupportedSpaceException,
)
from mesa.experimental.scenarios import Scenario
from mesa.model import Model
from mesa.time import Schedule
from mesa.visualization.space_renderer import SpaceRenderer


def test_rng_mismatch_exception():
    """Test that RNGMismatchException is raised when RNGs don't match."""
    scenario = Scenario(rng=np.random.default_rng(42))
    different_rng = np.random.default_rng(43)

    with pytest.raises(RNGMismatchException):
        Model(rng=different_rng, scenario=scenario)

def test_past_event_exception_schedule_event():
    """Test that PastEventException is raised for past scheduling."""
    model = Model()
    model.run_until(10)

    with pytest.raises(PastEventException):
        model.schedule_event(lambda: None, at=5)

def test_past_event_exception_schedule_recurring():
    """Test that PastEventException is raised for past recurring schedule."""
    model = Model()
    model.run_until(10)
    schedule = Schedule(interval=1.0, start=3.0)

    with pytest.raises(PastEventException):
        model.schedule_recurring(lambda: None, schedule)

def test_callback_type_error():
    """Test that CallbackTypeError is raised for non-callables."""
    model = Model()
    with pytest.raises(CallbackTypeError):
        model.schedule_event("not a callable", at=1)

def test_callback_value_error_lambda():
    """Test that CallbackValueError is raised for lambda callbacks."""
    model = Model()
    with pytest.raises(CallbackValueError, match="function must be alive"):
        model.schedule_event(lambda: None, at=1)

def test_invalid_schedule_exception():
    """Test that InvalidScheduleException is raised for invalid schedule params."""
    model = Model()
    with pytest.raises(InvalidScheduleException, match="Specify exactly one"):
        model.schedule_event(lambda: None, at=1, after=1)

    with pytest.raises(InvalidScheduleException, match="interval must be > 0"):
        Schedule(interval=0)

def test_empty_event_list_exception():
    """Test that EmptyEventListException is raised when popping empty list."""
    model = Model()
    model._event_list.clear() # Clear default schedule
    with pytest.raises(EmptyEventListException):
        model._event_list.pop_event()

def test_invalid_option_exception():
    """Test that InvalidOptionException is raised for unknown options in AgentSet."""
    model = Model()

    class TestAgent(Agent):
        pass

    TestAgent(model)

    with pytest.raises(InvalidOptionException):
        model.agents.get("unique_id", handle_missing="invalid")
def test_unsupported_backend_exception():
    """Test that UnsupportedBackendException is raised for invalid backends."""
    model = Model()
    model.grid = OrthogonalMooreGrid((10, 10))
    with pytest.raises(UnsupportedBackendException):
        SpaceRenderer(model, backend="invalid")

def test_unsupported_space_exception():
    """Test that UnsupportedSpaceException is raised for unsupported space types."""
    model = Model()
    model.grid = "not a mesa space"
    with pytest.raises(UnsupportedSpaceException):
        SpaceRenderer(model)

def test_backward_compatibility():
    """Test that new exceptions can be caught using generic Python exceptions."""
    model = Model()

    with pytest.raises(ValueError):
        model.schedule_event(lambda: None, at=1, after=1)

    model._event_list.clear()
    with pytest.raises(IndexError):
        model._event_list.pop_event()

    with pytest.raises(TypeError):
        model.schedule_event("not a callable", at=1)
