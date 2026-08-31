"""Storage for parameter sweeps."""

from __future__ import annotations

from dataclasses import astuple, dataclass, fields
from enum import Enum
from pathlib import Path
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
    from mesa.model import Model


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
    replication_id: int


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

    def write_scenarios(
        self, scenarios: list[Scenario], config: Any | None = None
    ) -> None:
        """Record the full ensemble of scenarios before dispatch.

        ``config`` is accepted for Store protocol compatibility and unused.
        """
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

    def to_directory(
        self,
        store_dir: str | Path,
        *,
        model_class: type[Model] | None = None,
        extra_provenance: dict[str, Any] | None = None,
    ) -> None:
        """Persist the in-memory store to disk.

        Writes store metadata (manifest, scenarios, status log) and all
        succeeded run outcomes to the specified directory.

        Layout::

            store_dir/
            ├── store.json
            ├── scenarios.json
            ├── status.log
            └── outputs/
                └── {output_name}/
                    └── data.arrow

        Args:
            store_dir: root directory of the store. Created if it does not exist.
            model_class: optional Model class to derive git provenance from.
            extra_provenance: optional extra provenance metadata to record in store.json.

        Raises:
            FileExistsError: if store.json already exists in store_dir.
        """
        import uuid  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        import pyarrow as pa  # noqa: PLC0415
        import pyarrow.ipc as pa_ipc  # noqa: PLC0415

        from mesa.experimental.scenarios import store_metadata  # noqa: PLC0415

        store_dir = Path(store_dir)
        store_dir.mkdir(parents=True, exist_ok=True)
        outputs_dir = store_dir / "outputs"
        outputs_dir.mkdir(exist_ok=True)

        session = uuid.uuid4().hex[:12]
        provenance = store_metadata.collect_provenance(
            model_class=model_class, extra=extra_provenance
        )
        store_metadata.write_store_manifest(
            store_dir, session=session, provenance=provenance
        )
        store_metadata.write_scenarios_manifest(store_dir, self.read_scenarios())

        for run_id, record in self._runs.items():
            if record.status != Status.PENDING:
                store_metadata.append_status(
                    store_dir, run_id, record.status, record.failure
                )

        output_names: set[str] = set()
        for record in self.succeeded().values():
            if record.output:
                output_names.update(record.output.keys())

        for output_name in sorted(output_names):
            out_dir = outputs_dir / output_name
            out_dir.mkdir(exist_ok=True)

            tables = []
            for run_id, record in self.succeeded().items():
                if record.output and output_name in record.output:
                    df = record.output[output_name]
                    if not isinstance(df, pd.DataFrame):
                        df = pd.DataFrame(df)
                    tagged_df = df.assign(
                        scenario_id=run_id.scenario_id,
                        replication_id=run_id.replication_id,
                    )
                    table = pa.Table.from_pandas(tagged_df, preserve_index=False)
                    tables.append(table)

            if tables:
                unified_table = pa.concat_tables(tables, promote_options="permissive")
                arrow_path = out_dir / f"worker-{session}-inmemory.arrow"
                with (
                    pa.OSFile(str(arrow_path), "wb") as sink,
                    pa_ipc.new_stream(sink, unified_table.schema) as writer,
                ):
                    writer.write_table(unified_table)

    to_disk = to_directory

    @classmethod
    def from_directory(
        cls,
        store_dir: str | Path,
        scenario_class: type[Scenario] | None = None,
    ) -> InMemoryStore:
        """Reconstruct an InMemoryStore from a persisted store directory.

        Reads manifests, replays the status log, and loads output DataFrames
        from outputs/ Arrow files back into memory.

        Args:
            store_dir: root directory of the store.
            scenario_class: the concrete Scenario subclass to instantiate.
                Defaults to Scenario.

        Returns:
            A populated InMemoryStore instance.
        """
        from pathlib import Path  # noqa: PLC0415

        import pyarrow as pa  # noqa: PLC0415
        import pyarrow.compute as pc  # noqa: PLC0415
        import pyarrow.ipc as pa_ipc  # noqa: PLC0415

        from mesa.experimental.scenarios import store_metadata  # noqa: PLC0415
        from mesa.experimental.scenarios.scenario import Scenario  # noqa: PLC0415

        if scenario_class is None:
            scenario_class = Scenario

        store_dir = Path(store_dir)
        manifest = store_metadata.read_store_manifest(store_dir)
        if manifest["store_format_version"] != store_metadata.STORE_FORMAT_VERSION:
            raise ValueError(
                f"store format version {manifest['store_format_version']} != "
                f"supported {store_metadata.STORE_FORMAT_VERSION}"
            )

        scenarios = store_metadata.read_scenarios_manifest(store_dir, scenario_class)
        statuses = store_metadata.read_status(store_dir)

        outputs_dir = store_dir / "outputs"
        output_tables: dict[str, pa.Table] = {}
        if outputs_dir.exists():
            for output_folder in sorted(outputs_dir.iterdir()):
                if output_folder.is_dir():
                    output_name = output_folder.name
                    tables = []
                    for path in sorted(output_folder.glob("*.arrow")):
                        try:
                            with pa.OSFile(str(path), "rb") as source:
                                reader = pa_ipc.open_stream(source)
                                batches = []
                                try:
                                    for batch in reader:
                                        batches.append(batch)
                                except pa.ArrowInvalid:
                                    pass
                                if batches:
                                    tables.append(pa.Table.from_batches(batches))
                        except (pa.ArrowInvalid, OSError):
                            pass
                    if tables:
                        output_tables[output_name] = pa.concat_tables(
                            tables, promote_options="permissive"
                        )

        store = cls()
        for scenario in scenarios:
            run_id = RunId(scenario.scenario_id, scenario.replication_id)
            status, failure = statuses.get(run_id, (Status.PENDING, None))
            output = None
            if status == Status.SUCCEEDED:
                output = {}
                for output_name, table in output_tables.items():
                    mask = pc.and_(
                        pc.equal(table["scenario_id"], run_id.scenario_id),
                        pc.equal(table["replication_id"], run_id.replication_id),
                    )
                    filtered_table = table.filter(mask)
                    cols_to_drop = [
                        c
                        for c in ["scenario_id", "replication_id"]
                        if c in filtered_table.column_names
                    ]
                    df = filtered_table.drop(cols_to_drop).to_pandas()
                    output[output_name] = df

            store._runs[run_id] = RunRecord(
                scenario=scenario,
                status=status,
                output=output,
                failure=failure,
            )
        return store

    from_disk = from_directory
