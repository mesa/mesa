"""Tests for Mesa's exception hierarchy definitions."""

from mesa.exceptions import (
    AgentSetException,
    CallbackTypeError,
    CallbackValueError,
    DimensionException,
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
    assert issubclass(InvalidOptionException, ValueError)
    assert issubclass(UnsupportedBackendException, ValueError)
    assert issubclass(UnsupportedSpaceException, ValueError)
