"""DiskStore — durable, resumable Store backed by Arrow IPC and JSON manifests.

The root-side counterpart to ``disk_writer``. It owns the store directory,
writes the JSON manifests, records status, and reads outcomes back by fanning
over the per-worker Arrow files.

Layout::

    store_dir/
    ├── store.json          # format version, sessions, provenance (written once)
    ├── scenarios.json      # authoritative, seed-complete (written once)
    ├── status.log          # append-only, root-written, torn-tail-tolerant
    └── outputs/
        └── {output_name}/
            └── worker-{session}-{host}-{uuid}.arrow

Two construction paths:

- ``DiskStore(store_dir, ...)`` prepares a NEW store directory and mints a
  session token, but does not yet write ``store.json`` — that happens in
  ``write_scenarios``, once a ``RunConfiguration`` is available to derive
  provenance from (see ``write_scenarios``). ``write_scenarios`` writes
  ``store.json`` exclusively (failing if the directory already holds one).
- ``DiskStore.from_directory(store_dir, scenario_class)`` attaches to an
  EXISTING store to read it back: it reads the manifests and the status log
  into the run set instead of writing anything. Read-only — resuming an
  incomplete sweep (dispatching only the still-pending runs) is not yet
  supported.

Status lives on the RunRecord in ``self._records`` and is the in-process source
of truth; ``status.log`` is its durable, replayable shadow. Every ``mark_*``
writes to both — appends a log line and updates the record — so they
never disagree, and a fresh process reconstructs the records by replaying the
log. The RunRecord ``output`` field is always None: DiskStore never
funnels outcomes to root, so output is resolved on demand by retrieve_output.
"""

from __future__ import annotations

import uuid
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as pa_ipc

from mesa.experimental.scenarios import store_metadata
from mesa.experimental.scenarios.disk_writer import DiskStreamWriter, _validate_schemas
from mesa.experimental.scenarios.exceptions import (
    ScenarioAbortedException,
    ScenarioFailedException,
    ScenarioNotFoundException,
    ScenarioNotReadyException,
)
from mesa.experimental.scenarios.store import RunId, RunRecord, Status

if TYPE_CHECKING:
    from mesa.experimental.scenarios.exceptions import FailureInfo
    from mesa.experimental.scenarios.runner import RunConfiguration
    from mesa.experimental.scenarios.scenario import Scenario
    from mesa.experimental.scenarios.store import Reference

_ID_COLUMNS = ["scenario_id", "replication_id"]


class DiskStore:
    """A durable Store persisting scenarios, status, and outcomes to disk."""

    def __init__(
        self,
        store_dir: str | Path,
        *,
        schemas: dict[str, pa.Schema] | None = None,
        on_schema_conflict: str = "warn",
    ):
        """Create a new store rooted at ``store_dir``.

        Only prepares the directory; ``store.json`` is not written until
        ``write_scenarios`` runs, because provenance is derived from the
        ``RunConfiguration`` supplied there (see ``write_scenarios``), not from
        anything known at construction time.

        Args:
            store_dir: root directory of the store. The directory is created if
                       it does not yet exist on disk.
            on_schema_conflict: how the read path resolves outputs whose schemas
                differ across workers — ``"warn"`` unifies with null-fill and
                warns, ``"raise"`` raises. Within-worker deviation is always an
                error, caught earlier by the writer.
            schemas: optional per-output Arrow schema, keyed by output name.
                Passed straight through to every ``DiskStreamWriter`` (see its
                ``schemas`` parameter). If passed, this will be used instead
                of inferring it from whichever run happens to write it first on
                a given worker — needed as soon as a run can return a zero-row
                frame for that output, since an empty column's inferred type
                does not reliably match the type inferred from a later non-empty
                batch of the same output.

        Raises:
            ValueError: if ``on_schema_conflict`` is invalid, or if any
                schema in ``schemas`` declares a reserved identity column
                (``scenario_id`` or ``replication_id`` — see
                ``disk_writer._validate_schemas``).

        """
        if on_schema_conflict not in ("warn", "raise"):
            raise ValueError(
                f"on_schema_conflict must be 'warn' or 'raise', "
                f"got {on_schema_conflict!r}"
            )
        self.store_dir = Path(store_dir)
        self.on_schema_conflict = on_schema_conflict
        self.schemas = schemas or {}
        _validate_schemas(self.schemas)
        self.session: str | None = uuid.uuid4().hex[
            :12
        ]  # unique token to identify a single session

        self.store_dir.mkdir(parents=True, exist_ok=True)
        (self.store_dir / "outputs").mkdir(exist_ok=True)

        # The run set: RunId -> RunRecord. Seeded by write_scenarios; each
        # record's status is updated in place by mark_* and is the in-process
        # source of truth. status.log is the durable shadow.
        self._records: dict[RunId, RunRecord] = {}

    # -- construction --------------------------------------------------------

    @classmethod
    def from_directory(
        cls,
        store_dir: str | Path,
        scenario_class: type[Scenario],
        *,
        on_schema_conflict: str = "warn",
    ) -> DiskStore:
        """Attach to an existing store to read it back, without resuming it.

        Reconstructs the run set from ``scenarios.json`` and ``status.log``
        instead of writing anything: every recorded scenario becomes a
        ``RunRecord``, with its status taken from the replayed log (runs the
        log never mentions default to PENDING, same as a fresh
        ``write_scenarios`` call).

        This is read-back only: no session is minted, so ``self.session`` is
        ``None`` rather than reused from the manifest — reusing a prior,
        already-closed session would let ``writer()`` silently mint writes
        under a stale identity instead of failing loudly. ``writer()`` raises
        if called on a store returned by this method. Resuming an incomplete
        sweep (minting a new session, reconciling the
        write-succeeded-but-status-missing gap, and dispatching only the
        still-pending runs) is a separate, not yet implemented, capability.

        Args:
            store_dir: root directory of the store.
            scenario_class: the concrete Scenario subclass to instantiate.
                JSON cannot carry the class safely, so the caller supplies it.
            on_schema_conflict: see ``__init__``.
        """
        if on_schema_conflict not in ("warn", "raise"):
            raise ValueError(
                f"on_schema_conflict must be 'warn' or 'raise', "
                f"got {on_schema_conflict!r}"
            )
        store_dir = Path(store_dir)
        manifest = store_metadata.read_store_manifest(store_dir)
        if manifest["store_format_version"] != store_metadata.STORE_FORMAT_VERSION:
            raise ValueError(
                f"store format version {manifest['store_format_version']} != "
                f"supported {store_metadata.STORE_FORMAT_VERSION}"
            )
        scenarios = store_metadata.read_scenarios_manifest(store_dir, scenario_class)
        statuses = store_metadata.read_status(store_dir)

        self = cls.__new__(cls)
        self.store_dir = store_dir
        self.on_schema_conflict = on_schema_conflict
        self.session = None

        self._records = {}
        for scenario in scenarios:
            run_id = RunId(scenario.scenario_id, scenario.replication_id)
            status, failure = statuses.get(run_id, (Status.PENDING, None))
            self._records[run_id] = RunRecord(
                scenario=scenario, status=status, failure=failure
            )
        return self

    # -- write side ----------------------------------------------------------

    def writer(self) -> DiskStreamWriter:
        """Return the per-job-pickleable, write-only handle for workers.

        ``schemas`` is passed through already-validated (at ``__init__``);
        ``DiskStreamWriter`` validates it again itself — redundant on this
        path, but that class is also constructed directly in isolation by
        tests, where the same check needs to run without going through a
        ``DiskStore`` at all.
        """
        if self.session is None:
            raise RuntimeError(
                "this DiskStore has no active session (it was loaded via "
                "from_directory, which is read-only); resuming a sweep is "
                "not yet supported, so writer() is unavailable"
            )
        return DiskStreamWriter(self.store_dir, self.session, schemas=self.schemas)

    def write_scenarios(
        self, scenarios: list[Scenario], config: RunConfiguration
    ) -> None:
        """Record the full ensemble before dispatch.

        Writes ``store.json`` exclusively (failing if the directory already
        holds one, so a new sweep never silently overwrites an existing
        store), then the authoritative ``scenarios.json``, then seeds the run
        set. Every run starts PENDING (the RunRecord default, and absent from
        the status log); terminal states are recorded in place as the sweep
        runs.

        Provenance is derived from ``config.model_class`` — the model that
        will actually execute these scenarios — rather than from a value
        supplied separately at construction time, so recorded provenance can
        never diverge from the config that actually ran.
        """
        store_metadata.write_store_manifest(
            self.store_dir,
            session=self.session,
            provenance=store_metadata.collect_provenance(
                model_class=config.model_class
            ),
        )
        store_metadata.write_scenarios_manifest(self.store_dir, scenarios)
        for scenario in scenarios:
            run_id = RunId(scenario.scenario_id, scenario.replication_id)
            self._records[run_id] = RunRecord(scenario=scenario)

    def read_scenarios(self) -> list[Scenario]:
        """Return the recorded scenarios."""
        return [r.scenario for r in self._records.values()]

    def mark_succeeded(self, ref: Reference) -> None:
        """Record that a run completed. Writes through to log and record.

        Unlike the in-memory store, no output is carried on the reference — the
        outcome is already on disk in the worker's Arrow files. Only status is
        recorded here; the output is resolved lazily by ``retrieve_output``.
        """
        self._record_status(ref.run_id, Status.SUCCEEDED, None)

    def mark_failed(self, run_id: RunId, failure: FailureInfo) -> None:
        """Record that a run failed, with diagnostics."""
        self._record_status(run_id, Status.FAILED, failure)

    def mark_aborted(self, run_id: RunId, failure: FailureInfo) -> None:
        """Record that a run was aborted (e.g. the executor pool broke)."""
        self._record_status(run_id, Status.ABORTED, failure)

    def _record_status(
        self, run_id: RunId, status: Status, failure: FailureInfo | None
    ) -> None:
        """Write one status transition through to the log and the record."""
        record = self._records.get(run_id)
        if record is None:
            raise ScenarioNotFoundException(run_id)
        store_metadata.append_status(self.store_dir, run_id, status, failure)
        record.status = status
        record.failure = failure

    # -- status queries ------------------------------------------------------

    def check_status(self, run_id: RunId) -> Status:
        """Check the status of a run."""
        record = self._records.get(run_id)
        if record is None:
            raise ScenarioNotFoundException(run_id)
        return record.status

    def status(self) -> pd.DataFrame:
        """One row per run: its current status."""
        rows = [
            (rid.scenario_id, rid.replication_id, record.status.value)
            for rid, record in self._records.items()
        ]
        frame = pd.DataFrame(rows, columns=[*_ID_COLUMNS, "status"])
        return frame.set_index(_ID_COLUMNS)

    def succeeded(self) -> dict[RunId, RunRecord]:
        """Return all succeeded runs."""
        return {
            rid: r for rid, r in self._records.items() if r.status == Status.SUCCEEDED
        }

    def failed(self) -> dict[RunId, RunRecord]:
        """Return all failed runs."""
        return {rid: r for rid, r in self._records.items() if r.status == Status.FAILED}

    def pending(self) -> dict[RunId, RunRecord]:
        """Return all pending runs."""
        return {
            rid: r for rid, r in self._records.items() if r.status == Status.PENDING
        }

    def aborted(self) -> dict[RunId, RunRecord]:
        """Return all aborted runs."""
        return {
            rid: r for rid, r in self._records.items() if r.status == Status.ABORTED
        }

    # -- read side -----------------------------------------------------------

    def retrieve_output(self, run_id: RunId) -> dict[str, pd.DataFrame]:
        """Resolve a run's outcome by reading it back from disk.

        Status-gated to match ``InMemoryStore``: a pending run is not ready, a
        failed or aborted run raises with its diagnostics, and only a succeeded
        run's outcome is returned. The outcome is assembled by fanning over
        every output directory's worker files, filtering to this ``run_id``.

        Every output present in the sweep appears as a key. A frame may be
        empty (zero rows): the run wrote a valid zero-row batch, or — in
        sweeps where not every run produces every output — the run never
        wrote to that output at all. The two are indistinguishable on read
        (the reference is key-only; per-run output membership is not
        recorded). The frame carries the output's *unified* schema (columns
        as reconciled across workers), which equals the run's original
        columns in the common stable-schema case.

        Note: this reads *every* worker file on each call — no assembled table
        is cached — so retrieving N runs individually re-reads the whole sweep N
        times. Acceptable while whole-sweep reads dominate and per-run retrieval
        is occasional.

        fixme: if per-run access becomes common, a future revision may
          take a ``RunId`` or list of ``RunId`` and/or cache the assembled tables.
        """
        record = self._records.get(run_id)
        if record is None:
            raise ScenarioNotFoundException(run_id)
        if record.status == Status.FAILED:
            raise ScenarioFailedException(run_id, record.failure)
        if record.status == Status.ABORTED:
            raise ScenarioAbortedException(run_id, record.failure)
        if record.status != Status.SUCCEEDED:
            raise ScenarioNotReadyException(run_id)

        result: dict[str, pd.DataFrame] = {}
        for output_name, table in self._read_all_outputs().items():
            keep = pc.and_(
                pc.equal(table["scenario_id"], run_id.scenario_id),
                pc.equal(table["replication_id"], run_id.replication_id),
            )
            result[output_name] = table.filter(keep).to_pandas()
        return result

    def _output_names(self) -> list[str]:
        """Names of outputs that have at least one worker file on disk."""
        outputs_dir = self.store_dir / "outputs"
        if not outputs_dir.exists():
            return []
        return sorted(
            p.name
            for p in outputs_dir.iterdir()
            if p.is_dir() and any(p.glob("*.arrow"))
        )

    def _read_all_outputs(self) -> dict[str, pa.Table]:
        """Read every output to a single Arrow table, one per output name.

        Fans over each output directory's worker files. Truncation-tolerant: a
        worker killed mid-batch leaves an unterminated stream, read to its last
        complete batch. Cross-worker schema deviation is reconciled here per
        ``on_schema_conflict``.
        """
        return {name: self._read_one_output(name) for name in self._output_names()}

    def _read_one_output(self, output_name: str) -> pa.Table:
        """Assemble one output's table across all its worker files."""
        output_dir = self.store_dir / "outputs" / output_name
        tables = []
        for path in sorted(output_dir.glob("*.arrow")):
            table = self._read_stream_file(path)
            if table is not None:
                tables.append(table)
        return self._unify(output_name, tables)

    @staticmethod
    def _read_stream_file(path: Path) -> pa.Table | None:
        """Read one Arrow IPC stream file, tolerating a truncated tail.

        Returns the table of all complete batches, or None if the file has no
        readable batches (empty, or torn before the first batch completed).

        A torn tail (a worker killed mid-write) surfaces as one of two
        distinct pyarrow exception types, depending on exactly where the
        truncation lands: ``pa.ArrowInvalid`` when the message itself is
        corrupted/unparsable, or ``OSError`` when the message's header
        parsed fine but declared a body length longer than what's actually
        left in the file (a short read). ``OSError`` is not a broadened
        catch-all here — pyarrow's ``ArrowIOError`` is a direct alias for the
        builtin ``OSError``, not a distinct subclass, so this is the exact
        type pyarrow raises for that second case, confirmed against Arrow's
        own source.
        """
        try:
            with pa.OSFile(str(path), "rb") as source:
                reader = pa_ipc.open_stream(source)
                batches = []
                try:
                    for batch in reader:
                        batches.append(batch)
                except (pa.ArrowInvalid, OSError):
                    # torn tail: everything before it is intact.
                    pass
                if not batches:
                    return None
                return pa.Table.from_batches(batches)
        except (pa.ArrowInvalid, OSError):
            # torn before even the schema/first batch was readable
            return None

    def _unify(self, output_name: str, tables: list[pa.Table]) -> pa.Table:
        """Concatenate worker tables, reconciling schema differences.

        Within a worker, schema is stable (enforced at write). Across workers, it
        can differ (a column present in some runs, absent in others).
        ``promote_options="permissive"`` unifies by null-filling absent columns.
        Under ``on_schema_conflict="raise"``, any cross-worker difference is an
        error instead.
        """
        if not tables:
            return pa.table({c: pa.array([], type=pa.int64()) for c in _ID_COLUMNS})

        first = tables[0].schema
        conflict = any(not t.schema.equals(first) for t in tables[1:])
        if conflict and self.on_schema_conflict == "raise":
            raise ValueError(
                f"cross-worker schema conflict for output {output_name!r} "
                "and on_schema_conflict='raise'"
            )
        if conflict:
            warnings.warn(
                f"unifying differing worker schemas for output "
                f"{output_name!r} (null-filling absent columns)",
                UserWarning,
                stacklevel=2,
            )
        return pa.concat_tables(tables, promote_options="permissive")
