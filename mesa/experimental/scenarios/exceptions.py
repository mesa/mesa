"""Exceptions for the scenarios module."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mesa.exceptions import MesaException

if TYPE_CHECKING:
    from mesa.experimental.scenarios.scenario import Scenario
    from mesa.experimental.scenarios.store import RunId

class ScenarioNotFoundException(MesaException):
    """Exception raised when a scenario cannot be found."""
    def __init__(self, run_id: RunId | None = None):
        """Initialize a scenario not found exception."""
        msg = f"No run found for {run_id}" if run_id else "Run not found"
        super().__init__(msg)
        self.run_id = run_id

class ScenarioNotReadyException(MesaException):
    """Exception raised when a scenario run has not yet been completed."""
    def __init__(self, run_id: RunId | None = None):
        """Initialize a scenario not ready exception."""
        msg = f"Run for {run_id} is not ready"
        super().__init__(msg)
        self.run_id = run_id

class ScenarioFailedException(MesaException):
    """Exception raised when a scenario run failed."""
    def __init__(self, run_id: RunId | None = None, failure: tuple | None = None):
        """Initialize a scenario failed exception."""
        msg = f"Run {run_id} failed"
        if failure:
            origin, exc_type, message, _ = failure
            msg += f": {exc_type} in {origin}: {message}"
        super().__init__(msg)
        self.run_id = run_id
        self.failure = failure

class ModelInstantiationException(MesaException):
    """Raised when a model cannot be instantiated for a scenario."""
    def __init__(self, model_class:type, model_args:list[Any], model_kwargs:dict[str, Any], scenario: Scenario):
        """Initialize a model instantiation exception."""
        msg = (f"Failed to instantiate {model_class.__name__} "
              f"Please check your model_args and model_kwargs.\n"
              f" - Passed args: {model_args}\n"
              f" - Passed kwargs: {model_kwargs}\n"
              f" - for scenario_id: {scenario.scenario_id}, replication_id: {scenario.replication_id}")
        super().__init__(msg)
        self.model_class = model_class
        self.model_args = model_args
        self.model_kwargs = model_kwargs
        self.scenario = scenario
