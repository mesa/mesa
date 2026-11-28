"""DiskStore — durable, resumable Store backed by Arrow IPC and JSON manifests.

The root-side counterpart to ``disk_writer``. It owns the store directory,
writes the JSON manifests, records status, and reads outcomes back by fanning
over the per-worker Arrow files.

Layout::

    store_dir/
    ├── store.json          # format version, sessions, provenance (written once)
    ├── scenarios.json      # authoritative, seed-complete (written once)
    ├── status.log          # append-only, root-written, torn-tail-tolerant, session-tagged
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

Durability: when is a run committed?
------------------------------------
A run is committed when its SUCCEEDED line is in ``status.log``. That line
names the session whose worker files hold the run's data, and
``retrieve_output`` reads only the files of that session. Data and status are
written by different processes (worker and root), so a kill can land between
the two. The design makes every such outcome safe without any rollback:

- data written, status line missing: the run reads as PENDING and is run again;
  the orphaned rows sit in a session that no SUCCEEDED line names, so they are
  never read;
- a run that failed half-way through writing its outputs (the ``WRITING``
  failure): same thing, the leftover rows are never read;
- a run repeated in a later session: only the later session's rows are read.

What the status line does not guarantee, unless ``durability="fsync"``, is that
the data survived a crash of the *node* the worker ran on (see ``DiskStore``).
"""

from __future__ import annotations

import uuid
import warnings
import weakref
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as pa_ipc

from mesa.experimental.scenarios import store_metadata
from mesa.experimental.scenarios.disk_writer import (
    DiskStreamWriter,
    _validate_durability,
    _validate_schemas,
)
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
    """A durable Store persisting scenarios, status, and outcomes to disk.

    Not safe with thread-backed executors (``ThreadPoolExecutor``): worker
    threads share one stream writer, which can silently corrupt the output
    files. Use a ``ProcessPoolExecutor``, an MPI executor, or run sequentially
    (``executor=None``).

    Under MPI (``mpi4py.futures.MPIPoolExecutor``, launched with e.g.
    ``mpiexec -n 8 python -m mpi4py.futures script.py``), workers write their
    Arrow files directly, so ``store_dir`` must be on a filesystem that every
    rank can reach (a shared or parallel filesystem, not node-local scratch).
    Only small key-only references travel back to rank 0. ``mpi4py.futures``
    workers exit without closing their Arrow files cleanly; that is harmless,
    because reads tolerate a missing end-of-stream marker.

    Call ``close()`` (or use the store as a context manager) when done, so the
    status log is synced to disk. If you forget, a finalizer does it at garbage
    collection or interpreter exit.
    """

    def __init__(
        self,
        store_dir: str | Path,
        *,
        schemas: dict[str, pa.Schema] | None = None,
        on_schema_conflict: str = "warn",
        durability: str = "flush",
    ):
        """Create a new store rooted at ``store_dir``.

        Only prepares the directory; ``store.json`` is not written until
        ``write_scenarios`` runs, because provenance is derived from the
        ``RunConfiguration`` supplied there (see ``write_scenarios``), not from
        anything known at construction time.

        Args:
            store_dir: root directory of the store. The directory is created if
                       it does not yet exist on disk.
            schemas: optional per-output Arrow schema, keyed by output name.
                Passed straight through to every ``DiskStreamWriter`` (see its
                ``schemas`` parameter). If passed, this will be used instead
                of inferring it from whichever run happens to write it first on
                a given worker — needed as soon as a run can return a zero-row
                frame for that output, since an empty column's inferred type
                does not reliably match the type inferred from a later non-empty
                batch of the same output.
            on_schema_conflict: how the read path resolves outputs whose schemas
                differ across workers — ``"warn"`` unifies with null-fill and
                warns, ``"raise"`` raises. Within-worker deviation is always an
                error, caught earlier by the writer.
            durability: what a worker guarantees about a run's data before it
                reports the run as done. ``"flush"`` (default) hands the data
                to the operating system: it survives the worker or root
                process being killed (e.g. a walltime limit) but not the
                worker's *node* crashing, and on NFS-style filesystems another
                node may not see it until the worker's files are closed.
                ``"fsync"`` additionally forces the data (and the new files'
                directory entries) to stable storage before the run is
                reported, which also covers node loss and makes the data
                visible across nodes, at the cost of a filesystem round trip
                per output per run. Prefer ``"fsync"`` on flaky nodes or
                NFS-style shared filesystems; on Lustre/GPFS and local disks
                ``"flush"`` is normally enough.

        Raises:
            ValueError: if ``on_schema_conflict`` or ``durability`` is invalid, or if any
                schema in ``schemas`` declares a reserved identity column
                (``scenario_id`` or ``replication_id`` — see
                ``disk_writer._validate_schemas``).

        """
        if on_schema_conflict not in ("warn", "raise"):
            raise ValueError(
                f"on_schema_conflict must be 'warn' or 'raise', "
                f"got {on_schema_conflict!r}"
            )
        _validate_durability(durability)
        self.store_dir = Path(store_dir)
        self.on_schema_conflict = on_schema_conflict
        self.durability = durability
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

        # For each SUCCEEDED run, the session whose files hold its data. This
        # is what makes the SUCCEEDED status line the commit point.
        self._commit_session: dict[RunId, str] = {}

        # status.log handle, opened on the first status write and closed by
        # close() or, as a safety net, by the finalizer.
        self._status_log: store_metadata.StatusLog | None = None
        self._finalizer: weakref.finalize | None = None

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
        self.durability = "flush"
        self.schemas = {}
        self.session = None
        self._status_log = None
        self._finalizer = None

        self._records = {}
        self._commit_session = {}
        for scenario in scenarios:
            run_id = RunId(scenario.scenario_id, scenario.replication_id)
            entry = statuses.get(run_id)
            if entry is None:
                status, failure = Status.PENDING, None
            else:
                status, failure = entry.status, entry.failure
                if status == Status.SUCCEEDED:
                    self._commit_session[run_id] = entry.session
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
        return DiskStreamWriter(
            self.store_dir,
            self.session,
            schemas=self.schemas,
            durability=self.durability,
        )

    def write_scenarios(
        self, scenarios: list[Scenario], config: RunConfiguration
    ) -> None:
        """Record the full ensemble before dispatch.

        Writes the authoritative ``scenarios.json`` first, then ``store.json``
        exclusively (failing if the directory already holds one, so a new
        sweep never silently overwrites an existing store), then seeds the
        run set. Every run starts PENDING (the RunRecord default, and absent
        from the status log); terminal states are recorded in place as the
        sweep runs.

        Deliberately scenarios-before-manifest — the reverse of the files'
        read order in ``from_directory``, which reads ``store.json`` first.
        ``scenarios.json`` is the one that can fail on arbitrary user data
        (a scenario parameter ``write_scenarios_manifest`` can't serialize);
        ``store.json`` is Mesa-controlled and essentially never fails.
        Writing the failure-prone one first means a bad parameter raises
        before either file exists on disk, so a retry on the same directory
        — after fixing the data — isn't blocked by a ``store.json`` left
        over from a write that got further than it should have.

        Provenance is derived from ``config.model_class`` — the model that
        will actually execute these scenarios — rather than from a value
        supplied separately at construction time, so recorded provenance can
        never diverge from the config that actually ran.
        """
        store_metadata.write_scenarios_manifest(self.store_dir, scenarios)
        store_metadata.write_store_manifest(
            self.store_dir,
            session=self.session,
            provenance=store_metadata.collect_provenance(
                model_class=config.model_class
            ),
        )

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
        """Write one status transition through to the log and the record.

        The log line carries this store's session. For SUCCEEDED that makes it
        the commit point: it names the session whose files hold the run's data.
        """
        record = self._records.get(run_id)
        if record is None:
            raise ScenarioNotFoundException(run_id)
        if self.session is None:
            raise RuntimeError(
                "this DiskStore has no active session (it was loaded via "
                "from_directory, which is read-only); it cannot record status"
            )
        self._get_status_log().append(run_id, status, failure, session=self.session)
        record.status = status
        record.failure = failure
        if status == Status.SUCCEEDED:
            self._commit_session[run_id] = self.session
        else:
            self._commit_session.pop(run_id, None)

    def _get_status_log(self) -> store_metadata.StatusLog:
        """Return the open status log, opening it (and its finalizer) if needed."""
        if self._status_log is None:
            self._status_log = store_metadata.StatusLog(self.store_dir)
            # The callback is a bound method of the log, not of self, so the
            # finalizer does not keep this store alive.
            self._finalizer = weakref.finalize(self, self._status_log.close)
        return self._status_log

    def close(self) -> None:
        """Sync and close the status log. Safe to call more than once.

        The store stays usable: the next status write reopens the log.
        """
        if self._finalizer is not None:
            self._finalizer()  # runs StatusLog.close once, then deactivates
        self._status_log = None
        self._finalizer = None

    def __enter__(self) -> DiskStore:
        """Return self, for use as a context manager."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the store on leaving the ``with`` block."""
        self.close()

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
        every output directory's worker files *of the session that committed
        the run* (the one named on its SUCCEEDED status line), filtering to this
        ``run_id``. Rows the run left in any other session — from an earlier
        attempt, or from a write that failed half-way — are never read.

        Every output present in that session appears as a key. A frame may be
        empty (zero rows): the run wrote a valid zero-row batch, or — in
        sweeps where not every run produces every output — the run never
        wrote to that output at all. The two are indistinguishable on read
        (the reference is key-only; per-run output membership is not
        recorded). The frame carries the output's *unified* schema (columns
        as reconciled across workers), which equals the run's original
        columns in the common stable-schema case.

        Note: this reads *every* worker file of the session on each call — no
        assembled table is cached — so retrieving N runs individually re-reads
        the whole session N times. Acceptable while whole-sweep reads dominate
        and per-run retrieval is occasional.

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
        session = self._commit_session[run_id]
        for output_name, table in self._read_all_outputs(session).items():
            keep = pc.and_(
                pc.equal(table["scenario_id"], run_id.scenario_id),
                pc.equal(table["replication_id"], run_id.replication_id),
            )
            result[output_name] = table.filter(keep).to_pandas()
        return result

    @staticmethod
    def _session_glob(session: str) -> str:
        """Glob matching the worker files of one session (see ``disk_writer``)."""
        return f"worker-{session}-*.arrow"

    def _output_names(self, session: str) -> list[str]:
        """Names of outputs with at least one worker file of ``session``."""
        outputs_dir = self.store_dir / "outputs"
        if not outputs_dir.exists():
            return []
        pattern = self._session_glob(session)
        return sorted(
            p.name for p in outputs_dir.iterdir() if p.is_dir() and any(p.glob(pattern))
        )

    def _read_all_outputs(self, session: str) -> dict[str, pa.Table]:
        """Read every output of one session to a single Arrow table per output name.

        Fans over each output directory's worker files of ``session``.
        Truncation-tolerant: a worker killed mid-batch leaves an unterminated
        stream, read to its last complete batch. Cross-worker schema deviation
        is reconciled here per ``on_schema_conflict``.
        """
        return {
            name: self._read_one_output(name, session)
            for name in self._output_names(session)
        }

    def _read_one_output(self, output_name: str, session: str) -> pa.Table:
        """Assemble one output's table across the worker files of ``session``."""
        output_dir = self.store_dir / "outputs" / output_name
        tables = []
        for path in sorted(output_dir.glob(self._session_glob(session))):
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
