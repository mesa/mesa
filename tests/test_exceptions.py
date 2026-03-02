"""Tests for Mesa's exception hierarchy definitions."""

from mesa.exceptions import (
    AgentException,
    AgentNotRegisteredException,
    AgentSetException,
    CallbackTypeError,
    CallbackValueError,
    CellFullException,
    DimensionException,
    DuplicateAgentIDException,
    EmptyEventListException,
    InvalidCallbackException,
    InvalidOptionException,
    InvalidScheduleException,
    MesaException,
    ModelException,
    PastEventException,
    RNGMismatchException,
    SpaceException,
    TimeException,
    UnsupportedBackendException,
    UnsupportedSpaceException,
    VisualizationException,
)


def test_exception_hierarchy():
    """Verify inheritance structure for newly introduced exception classes."""
    assert issubclass(SpaceException, MesaException)
    assert issubclass(ModelException, MesaException)
    assert issubclass(TimeException, MesaException)
    assert issubclass(AgentException, MesaException)
    assert issubclass(AgentSetException, MesaException)
    assert issubclass(VisualizationException, MesaException)

    # Built-in compatibility for selective catching.
    assert issubclass(DimensionException, ValueError)
    assert issubclass(RNGMismatchException, ValueError)
    assert issubclass(PastEventException, ValueError)
    assert issubclass(InvalidCallbackException, TimeException)
    assert issubclass(CallbackTypeError, TypeError)
    assert issubclass(CallbackValueError, ValueError)
    assert issubclass(InvalidScheduleException, ValueError)
    assert issubclass(EmptyEventListException, IndexError)
    assert issubclass(AgentNotRegisteredException, LookupError)
    assert issubclass(DuplicateAgentIDException, KeyError)
    assert issubclass(InvalidOptionException, ValueError)
    assert issubclass(UnsupportedBackendException, ValueError)
    assert issubclass(UnsupportedSpaceException, ValueError)


def test_existing_space_exception_still_works():
    """Keep a sanity check for an existing exception payload."""
    err = CellFullException((1, 2))
    assert err.coordinate == (1, 2)
