"""Exceptions for the scenarios module."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from mesa.exceptions import MesaException

if TYPE_CHECKING:
    from mesa.experimental.scenarios.scenario import Scenario
    from mesa.experimental.scenarios.store import RunId


# ---------------------------------------------------------------------------
# Run-stage exceptions
#
# Raised inside RunConfiguration to attribute a failure to a specific stage
# of a run. Each carries a FailureOrigin via the class-level `origin`, so the
# runner can record where a run broke without inspecting which method raised.
# Always raised with `raise ... from e` so the underlying exception is
# preserved as __cause__ and the recorded failure reports the real error type.
# ---------------------------------------------------------------------------


class FailureOrigin(Enum):
    """Enum describing where a scenario run failed."""

    INSTANTIATING = "instantiating"
    RUNNING = "running"
    EXTRACTING = "extracting"
    WRITING = "writing"
    ABORTED = "aborted"


@dataclass(frozen=True)
class FailureInfo:
    """Structured diagnostics for a failed run.

    ensures traceback information can be safely sent from workers to root.

    """

    origin: FailureOrigin
    exception_type: str
    message: str
    traceback: str


class RunStageException(MesaException):
    """Base for failures attributable to a stage of a single run."""

    #: The stage this exception represents. Set by each subclass.
    origin: FailureOrigin

    def __init__(self, message: str):
        """Initialize a run-stage exception.

        Args:
            message: human-readable description of the stage failure. The
                underlying cause should be attached via ``raise ... from e``.
        """
        super().__init__(message)


class ModelInstantiationException(RunStageException):
    """Raised when a model cannot be instantiated for a scenario."""

    origin: FailureOrigin = FailureOrigin.INSTANTIATING

    def __init__(
        self,
        model_class: type,
        model_args: list[Any],
        model_kwargs: dict[str, Any],
        scenario: Scenario,
    ):
        """Initialize a model instantiation exception.

        Args:
            model_class: the model class that failed to instantiate
            model_args: the positional args passed to the model
            model_kwargs: the keyword args passed to the model
            scenario: the scenario being instantiated
        """
        msg = (
            f"Failed to instantiate {model_class.__name__}. "
            f"Please check your model_args and model_kwargs.\n"
            f" - Passed args: {model_args}\n"
            f" - Passed kwargs: {model_kwargs}\n"
            f" - for scenario_id: {scenario.scenario_id}, "
            f"replication_id: {scenario.replication_id}"
        )
        super().__init__(msg)
        self.model_class = model_class
        self.model_args = model_args
        self.model_kwargs = model_kwargs
        self.scenario = scenario


class ModelRunException(RunStageException):
    """Raised when a model fails while advancing (run_model)."""

    origin: FailureOrigin = FailureOrigin.RUNNING

    def __init__(self, run_id: RunId):
        """Initialize a model run exception.

        Args:
            run_id: identifier of the run that failed during execution
        """
        super().__init__(f"Model run failed for {run_id}")
        self.run_id = run_id


class OutcomeExtractionException(RunStageException):
    """Raised when extracting outcomes from a finished model fails."""

    origin = FailureOrigin.EXTRACTING

    def __init__(self, run_id: RunId, outcomes: list[str] | None):
        """Initialize an outcome extraction exception.

        Args:
            run_id: identifier of the run whose extraction failed
            outcomes: the requested outcome keys (None means "all")
        """
        requested = "all" if outcomes is None else outcomes
        super().__init__(
            f"Outcome extraction failed for {run_id} (outcomes={requested})"
        )
        self.run_id = run_id
        self.outcomes = outcomes


# ---------------------------------------------------------------------------
# Store-side (read path) exceptions
#
# Raised by the store when querying run results, unrelated to run stages.
# ---------------------------------------------------------------------------


class ScenarioNotFoundException(MesaException):
    """Raised when no run is recorded for a given RunId."""

    def __init__(self, run_id: RunId | None = None):
        """Initialize a scenario-not-found exception.

        Args:
            run_id: the RunId that could not be found, if known
        """
        msg = f"No run found for {run_id}" if run_id else "Run not found"
        super().__init__(msg)
        self.run_id = run_id


class ScenarioNotReadyException(MesaException):
    """Raised when a run's output is requested before it has completed."""

    def __init__(self, run_id: RunId | None = None):
        """Initialize a scenario-not-ready exception.

        Args:
            run_id: the RunId whose output is not yet available
        """
        super().__init__(f"Run for {run_id} is not ready")
        self.run_id = run_id


class ScenarioFailedException(MesaException):
    """Raised when a failed run's output is requested."""

    def __init__(self, run_id: RunId | None = None, failure: FailureInfo | None = None):
        """Initialize a scenario-failed exception.

        Args:
            run_id: the RunId of the failed run
            failure: structured failure diagnostics for the run, if available
        """
        msg = f"Run {run_id} failed"
        if failure is not None:
            msg += (
                f": {failure.exception_type} in {failure.origin.value}: "
                f"{failure.message}"
            )
        super().__init__(msg)
        self.run_id = run_id
        self.failure = failure


class ScenarioAbortedException(MesaException):
    """Raised when an aborted run's output is requested."""

    def __init__(self, run_id: RunId | None = None, failure: FailureInfo | None = None):
        """Initialize a scenario-aborted exception.

        Args:
            run_id: the RunId of the aborted run
            failure: structured failure diagnostics for the run, if available
        """
        msg = f"Run {run_id} aborted"
        if failure is not None:
            msg += (
                f": {failure.exception_type} in {failure.origin.value}: "
                f"{failure.message}"
            )
        super().__init__(msg)
        self.run_id = run_id
        self.failure = failure
