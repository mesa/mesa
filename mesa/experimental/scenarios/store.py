from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import pandas as pd

from mesa.exceptions import MesaException

if TYPE_CHECKING:
    from mesa.experimental.scenarios.scenario import Scenario

class Status(Enum):
    """Enumeration for scenario run status."""
    PENDING = "PENDING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"

@dataclass(frozen=True)
class RunId:
    """Identifier for a specific scenario replication combination."""
    scenario_id: int
    replication_id: int | None = None


@runtime_checkable
class Writer(Protocol):
    """Worker-side handle that produces references.

    Picklable; carries only configuration, never the store's durable record. T
    his is the ONLY store capability a worker receives.
    """
    def to_reference(
        self, scenario_id: int, replication_id: int | None, outcome: dict
    ) -> Reference:
        """Persist a run's outcome and return a reference to it."""
        ...


@runtime_checkable
class Store(Protocol):
    """The Store interface."""

    def writer(self) -> Writer:
        """Return the picklable, write-only handle to hand to workers."""
        ...

    def retrieve_output(self, run_id: RunId) -> dict[str, pd.DataFrame]:
        """Resolve a reference back to its outcome."""
        ...

    # --- design: root, once, up front ---
    def write_scenarios(self, scenarios: list[Scenario]) -> None:
        """Record the full ensemble of scenarios before dispatch."""
        ...

    def read_scenarios(self) -> list[Scenario]:
        """Return the recorded scenarios."""
        ...

    # --- status: root, incremental (durability deferred to a later PR) ---
    def mark_succeeded(self, ref: Reference) -> None:
        """Record that a run completed and its outcome was received."""
        ...

    def mark_failed(
        self,
        run_id: RunId,
        *,
        origin: str,
        exception_type: str,
        message: str,
        traceback: str,
    ) -> None:
        """Record that a run failed, with its origin and diagnostics."""
        ...

    def status(self) -> pd.DataFrame:
        """One row per design point: pending / succeeded / failed."""
        ...

    def check_status(self, run_id: RunId) -> Status:
        """Check the status of the reference."""
        ...


@runtime_checkable
class Reference(Protocol):
    """A small, picklable handle to a single run's outcome.

    Returned by ``Store.write`` (worker side) and consumed by ``Store.read``
    (root side). Carries the run identity so it is self-describing — ``read``
    needs only the reference, not the ids passed separately.

    Must pickle cheaply: a reference crosses the process/rank boundary as the
    return value of the per-run worker call.
    """
    run_id: RunId
    payload: Any

@dataclass(frozen=True)
class InMemoryReference:
    """In-memory reference for scenario runs."""
    run_id: RunId
    payload: dict[str, pd.DataFrame]   # rides the boundary inline


class InMemoryWriter:
    """Writer for in-memory store."""

    def to_reference(
        self, scenario_id: int, replication_id: int | None, outcome: dict
    ) -> Reference:
        """Persist a run's outcome and return a reference to it."""
        return InMemoryReference(RunId(scenario_id, replication_id), outcome)

class InMemoryStore:
    """Implements in memory store following store protocol."""

    def __init__(self):
        """Initialize in-memory store."""
        self._scenarios: dict[RunId, Scenario] = {}
        self._outputs: dict[RunId, dict[str, pd.DataFrame]] = {}
        self._statuses: dict[RunId, Status] = {}
        self._failures: dict[RunId, tuple] = {}

    def writer(self) -> Writer:
        """Return the picklable, write-only handle to hand to workers."""
        return InMemoryWriter()

    def retrieve_output(self, run_id:RunId) -> dict[str, pd.DataFrame]:
        """Retrieve a run's output."""
        status = self._statuses.get(run_id)
        if status is None:
            raise ScenarioNotFoundException()
        if status is Status.PENDING:
            raise ScenarioNotReadyException()  # not yet run
        if status is Status.FAILED:
            raise ScenarioFailedException()  # ran, but failed — see status()/failure
        return self._outputs[run_id]

    def write_scenarios(self, scenarios: list[Scenario]) -> None:
        """Record the full ensemble of scenarios before dispatch."""
        for scenario in scenarios:
            key = RunId(scenario.scenario_id, scenario.replication_id)
            self._scenarios[key] = scenario
            self._statuses[key] = Status.PENDING

    def read_scenarios(self) -> list[Scenario]:
        """Return the recorded scenarios."""
        return list(self._scenarios.values())

    # --- status: root, incremental (durability deferred to a later PR) ---
    def mark_succeeded(self, ref: Reference) -> None:
        """Record that a run completed and its outcome was received."""
        key = ref.run_id
        self._statuses[key] = Status.SUCCEEDED
        self._outputs[key] = ref.payload

    def mark_failed(
        self,
        run_id: RunId,
        *,
        origin: str,
        exception_type: str,
        message: str,
        traceback: str,
    ) -> None:
        """Record that a run failed, with its origin and diagnostics."""
        self._statuses[run_id] = Status.FAILED
        self._failures[run_id] = (origin, exception_type, message, traceback)

    def status(self) -> pd.DataFrame:
        """One row per design point: pending / succeeded / failed."""
        # Extract field names from the dataclass
        field_names = ["scenario_id", "replication_id"]

        # Build MultiIndex from the dataclass fields
        statuses = self._statuses

        idx = pd.MultiIndex.from_tuples(
            [(k.scenario_id, k.replication_id) for k in statuses], names=field_names
        )
        return pd.DataFrame(list(statuses.values()), index=idx, columns=["status"])

    def check_status(self, run_id:RunId) -> Status:
        """Check the status of the reference."""
        try:
            return self._statuses[run_id]
        except KeyError as e:
            raise ScenarioNotFoundException() from e


class ScenarioNotFoundException(MesaException):
    """Exception raised when a scenario cannot be found."""


class ScenarioNotReadyException(MesaException):
    """Exception raised when a scenario run has not yet been completed."""


class ScenarioFailedException(MesaException):
    """Exception raised when a scenario run failed."""

