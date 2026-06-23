"""Exceptions for the scenarios module."""

from mesa.exceptions import MesaException


class ScenarioNotFoundException(MesaException):
    """Exception raised when a scenario cannot be found."""


class ScenarioNotReadyException(MesaException):
    """Exception raised when a scenario run has not yet been completed."""


class ScenarioFailedException(MesaException):
    """Exception raised when a scenario run failed."""


class ModelInstantiationException(MesaException):
    """Raised when a model cannot be instantiated for a scenario."""
