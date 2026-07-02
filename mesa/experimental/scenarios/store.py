"""Storage for parameter sweeps."""

from __future__ import annotations

from dataclasses import astuple, dataclass, fields
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import pandas as pd

from mesa.experimental.scenarios.exceptions import (
    ScenarioAbortedException,
    ScenarioFailedException,
    ScenarioNotFoundException,
    ScenarioNotReadyException,
)

if TYPE_CHECKING:
    from mesa.experimental.scenarios.exceptions import FailureInfo
    from mesa.experimental.scenarios.scenario import Scenario


class Status(Enum):
    """Enumeration for scenario run status."""

    PENDING = "PENDING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"


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
    failure: FailureInfo | None = None


@runtime_checkable
class Writer(Protocol):
    """Worker-side handle that produces references.

    Picklable; carries only configuration, never the store's durable record.
    This is the ONLY store capability a worker receives.
    """

    def to_reference(self, run_id: RunId, outcome: dict) -> Reference:
        """Persist a run's outcome and return a reference to it."""
        ...


@runtime_checkable
class Store(Protocol):
    """The Store interface."""

    def writer(self) -> Writer:
        """Return the pickleable, write-only handle to hand to workers."""
        ...

    def retrieve_output(self, run_id: RunId) -> dict[str, pd.DataFrame]:
        """Resolve a reference back to its outcome."""
        ...

    def write_scenarios(self, scenarios: list[Scenario]) -> None:
        """Record the full ensemble of scenarios before dispatch.

        It is critical that this method is called prior to executing any runs, because mark_succeeded and mark_failed
        will check against the registered runs.

        """
        ...

    def read_scenarios(self) -> list[Scenario]:
        """Return the recorded scenarios."""
        ...

    def mark_succeeded(self, ref: Reference) -> None:
        """Record that a run completed and its outcome was received.

        For a run to be marked, the scenario should first have been registered via write_scenarios.

        """
        ...

    def mark_failed(self, run_id: RunId, failure: FailureInfo) -> None:
        """Record that a run failed, with its origin and diagnostics.

        For a run to be marked, the scenario should first have been registered via write_scenarios.

        """
        ...

    def mark_aborted(self, run_id: RunId, failure: FailureInfo) -> None:
        """Record that a run was aborted (e.g. because the executor pool broke).

        For a run to be marked, the scenario should first have been registered via write_scenarios.

        """
        ...

    def status(self) -> pd.DataFrame:
        """One row per scenario: pending / succeeded / failed."""
        ...

    def check_status(self, run_id: RunId) -> Status:
        """Check the status of the run id."""
        ...

    def succeeded(self) -> dict[RunId, RunRecord]:
        """Return all succeeded RunIds and their run record."""
        ...

    def failed(self) -> dict[RunId, RunRecord]:
        """Return all failed RunIds and their run record."""
        ...

    def pending(self) -> dict[RunId, RunRecord]:
        """Return all pending RunIds and their run record."""
        ...

    def aborted(self) -> dict[RunId, RunRecord]:
        """Return all aborted RunIds and their run record."""
        ...


@runtime_checkable
class Reference(Protocol):
    """A small, picklable handle to a single run's outcome."""

    @property
    def run_id(self) -> RunId:
        """Return the run_id."""
        ...

    @property
    def payload(self) -> Any:
        """Return the payload."""
        ...


@dataclass(frozen=True)
class InMemoryReference:
    """In-memory reference for scenario runs."""

    run_id: RunId
    payload: dict[str, pd.DataFrame]


class InMemoryWriter:
    """Writer for in-memory store."""

    def to_reference(self, run_id: RunId, outcome: dict) -> InMemoryReference:
        """Persist a run's outcome and return a reference to it."""
        return InMemoryReference(run_id, outcome)


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
            raise ScenarioNotFoundException(run_id) from e

    def writer(self) -> InMemoryWriter:
        """Return the pickleable, write-only handle to hand to workers."""
        return InMemoryWriter()

    def retrieve_output(self, run_id: RunId) -> dict[str, pd.DataFrame]:
        """Retrieve a run's output."""
        record = self._get_record(run_id)
        if record.status == Status.PENDING:
            raise ScenarioNotReadyException(run_id)
        if record.status == Status.FAILED:
            raise ScenarioFailedException(run_id, record.failure)
        if record.status == Status.ABORTED:
            raise ScenarioAbortedException(run_id, record.failure)
        if record.status != Status.SUCCEEDED:
            raise ScenarioNotReadyException(run_id)
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

    def mark_failed(self, run_id: RunId, failure: FailureInfo) -> None:
        """Record that a run failed, with its origin and diagnostics."""
        record = self._get_record(run_id)
        record.status = Status.FAILED
        record.failure = failure

    def mark_aborted(self, run_id: RunId, failure: FailureInfo) -> None:
        """Record that a run was aborted (e.g. because the executor pool broke)."""
        record = self._get_record(run_id)
        record.status = Status.ABORTED
        record.failure = failure

    def status(self) -> pd.DataFrame:
        """One row per design point: pending / succeeded / failed."""
        idx = pd.MultiIndex.from_tuples(
            [astuple(run_id) for run_id in self._runs],
            names=[f.name for f in fields(RunId)],
        )
        return pd.DataFrame(
            [r.status.value for r in self._runs.values()],
            index=idx,
            columns=["status"],
        )

    def check_status(self, run_id: RunId) -> Status:
        """Check the status of a run."""
        return self._get_record(run_id).status

    def succeeded(self) -> dict[RunId, RunRecord]:
        """Return all succeeded runs."""
        return {rid: r for rid, r in self._runs.items() if r.status == Status.SUCCEEDED}

    def failed(self) -> dict[RunId, RunRecord]:
        """Return all failed runs."""
        return {rid: r for rid, r in self._runs.items() if r.status == Status.FAILED}

    def pending(self) -> dict[RunId, RunRecord]:
        """Return all pending runs."""
        return {rid: r for rid, r in self._runs.items() if r.status == Status.PENDING}

    def aborted(self) -> dict[RunId, RunRecord]:
        """Return all aborted runs."""
        return {rid: r for rid, r in self._runs.items() if r.status == Status.ABORTED}
