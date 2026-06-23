"""Storage for parameter sweeps."""

from __future__ import annotations

from dataclasses import astuple, dataclass, fields
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import pandas as pd

from mesa.experimental.scenarios.exceptions import (
    ScenarioFailedException,
    ScenarioNotFoundException,
    ScenarioNotReadyException,
)

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


@dataclass
class RunRecord:
    """All state associated with a single run."""

    scenario: Scenario
    status: Status = Status.PENDING
    output: dict[str, pd.DataFrame] | None = None
    failure: tuple[str, str, str, str] | None = None  # origin, type, message, traceback


@runtime_checkable
class Writer(Protocol):
    """Worker-side handle that produces references.

    Picklable; carries only configuration, never the store's durable record.
    This is the ONLY store capability a worker receives.
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

    def write_scenarios(self, scenarios: list[Scenario]) -> None:
        """Record the full ensemble of scenarios before dispatch."""
        ...

    def read_scenarios(self) -> list[Scenario]:
        """Return the recorded scenarios."""
        ...

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

    def get_runs_by_status(self, status: Status) -> list[RunId]:
        """Return all RunIds with the given status."""
        ...

    def succeeded(self) -> list[RunId]:
        """Return all succeeded RunIds."""
        ...

    def failed(self) -> list[RunId]:
        """Return all failed RunIds."""
        ...

    def pending(self) -> list[RunId]:
        """Return all pending RunIds."""
        ...

    def get_failed_records(self) -> dict[RunId, RunRecord]:
        """Return all failed RunRecords, including failure diagnostics."""
        ...


@runtime_checkable
class Reference(Protocol):
    """A small, picklable handle to a single run's outcome."""

    run_id: RunId
    payload: Any


@dataclass(frozen=True)
class InMemoryReference:
    """In-memory reference for scenario runs."""

    run_id: RunId
    payload: dict[str, pd.DataFrame]


class InMemoryWriter:
    """Writer for in-memory store."""

    def to_reference(
        self, scenario_id: int, replication_id: int | None, outcome: dict
    ) -> InMemoryReference:
        """Persist a run's outcome and return a reference to it."""
        return InMemoryReference(RunId(scenario_id, replication_id), outcome)


class InMemoryStore:
    """Implements in-memory store following the Store protocol."""

    def __init__(self):
        """Initialize in-memory store."""
        self._runs: dict[RunId, RunRecord] = {}

    def _get_record(self, run_id: RunId) -> RunRecord:
        """Look up a record or raise ScenarioNotFoundException."""
        try:
            return self._runs[run_id]
        except KeyError as e:
            raise ScenarioNotFoundException() from e

    def writer(self) -> InMemoryWriter:
        """Return the pickleable, write-only handle to hand to workers."""
        return InMemoryWriter()

    def retrieve_output(self, run_id: RunId) -> dict[str, pd.DataFrame]:
        """Retrieve a run's output."""
        record = self._get_record(run_id)
        if record.status == Status.PENDING:
            raise ScenarioNotReadyException()
        if record.status == Status.FAILED:
            raise ScenarioFailedException()
        return record.output

    def write_scenarios(self, scenarios: list[Scenario]) -> None:
        """Record the full ensemble of scenarios before dispatch."""
        for scenario in scenarios:
            key = RunId(scenario.scenario_id, scenario.replication_id)
            self._runs[key] = RunRecord(scenario=scenario)

    def read_scenarios(self) -> list[Scenario]:
        """Return the recorded scenarios."""
        return [r.scenario for r in self._runs.values()]

    def mark_succeeded(self, ref: Reference) -> None:
        """Record that a run completed and its outcome was received."""
        record = self._get_record(ref.run_id)
        record.status = Status.SUCCEEDED
        record.output = ref.payload

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
        record = self._get_record(run_id)
        record.status = Status.FAILED
        record.failure = (origin, exception_type, message, traceback)

    def status(self) -> pd.DataFrame:
        """One row per design point: pending / succeeded / failed."""
        idx = pd.MultiIndex.from_tuples(
            [astuple(run_id) for run_id in self._runs],
            names=[f.name for f in fields(RunId)],
        )
        return pd.DataFrame(
            [r.status for r in self._runs.values()],
            index=idx,
            columns=["status"],
        )

    def check_status(self, run_id: RunId) -> Status:
        """Check the status of a run."""
        return self._get_record(run_id).status

    def get_runs_by_status(self, status: Status) -> list[RunId]:
        """Return all RunIds with the given status."""
        return [rid for rid, r in self._runs.items() if r.status == status]

    def succeeded(self) -> list[RunId]:
        """Return all succeeded RunIds."""
        return self.get_runs_by_status(Status.SUCCEEDED)

    def failed(self) -> list[RunId]:
        """Return all failed RunIds."""
        return self.get_runs_by_status(Status.FAILED)

    def pending(self) -> list[RunId]:
        """Return all pending RunIds."""
        return self.get_runs_by_status(Status.PENDING)

    def get_failed_records(self) -> dict[RunId, RunRecord]:
        """Return all failed RunRecords, including failure diagnostics."""
        return {rid: r for rid, r in self._runs.items() if r.status == Status.FAILED}
