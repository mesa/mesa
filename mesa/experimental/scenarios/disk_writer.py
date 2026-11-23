"""DiskStore writer — worker-side Arrow IPC streaming.

The writer is re-pickled and handed in with every job (matching run_scenarios'
dispatch via ``store.writer()``), so it CANNOT hold open streams on itself:
each job receives a fresh copy with no streams. Instead, the open streams live
in a module-level per-process registry. The worker PROCESS persists across
jobs even though the writer OBJECT is reconstructed each task, so a stream
opened on a worker's first run is reused by its later runs.

The filename ``{output}/worker-{session}-{host}-{uuid}.arrow`` keeps each
worker's file distinct along every axis that can collide:
  - ``uuid``     a token minted once per ``DiskStreamWriter`` INSTANCE, on
                 first use inside ``_file_for`` (see ``_instance_uuid``) —
                 stored on the instance, not the module. A module-level
                 value, even computed lazily, is not enough: if the SAME
                 process that later forks workers had already itself
                 written a batch (e.g. an earlier sequential/in-process run
                 reusing the same store before a parallel one), that
                 process's cached value would already be set before any
                 fork happens, and every forked child would inherit that
                 identical cached value via the fork-time memory copy — the
                 exact collision this token exists to prevent. An instance
                 attribute sidesteps this entirely: this class is re-pickled
                 and independently unpickled for every job (see the opening
                 paragraph above), and unpickling always produces a
                 genuinely new object with its own ``None`` ``_uuid``, never
                 a shared reference to whatever writer computed one before
                 it — so there is nothing left for a fork (or any other
                 process) to duplicate. Must stay ``None`` until first
                 actually needed; setting it eagerly in ``__init__`` would
                 bake one value into every pickled copy, reproducing the
                 identical collision without even requiring a fork to do it;
  - ``host``     not needed for uniqueness now that the uuid provides it, but
                 kept alongside it so a file can still be eyeballed to the
                 node that produced it — an opaque uuid can't be read that
                 way the way a pid once could. Useful once PR 6 adds MPI
                 across nodes;
  - ``session``  distinguishes invocations (resume) — an Arrow IPC
                 stream cannot be reopened once its end-of-stream marker is
                 written, so a resumed sweep opens NEW files rather than
                 appending to a prior session's.

Data is written where it is produced; only a key-only ``DiskReference`` crosses back
to the root. That is the whole point of a worker-side writer.

The module-level stream registry (``_STREAMS``/``_SINKS``/``_SCHEMAS`` below)
has the same fork hazard as the uuid above, from a different angle: if the
process that later forks workers had already opened real streams itself (the
same already-written-before-forking scenario the per-instance uuid guards
against), every forked child would inherit those entries as already
satisfying ``key in _STREAMS`` — skipping ``_open_stream`` entirely and
writing straight into a stream object whose underlying OS file descriptor is
now shared, unmediated, with the parent process (and any sibling that
inherited the same entry), corrupting the file for everyone. ``_rotate_session``
guards against this by comparing ``os.getpid()`` against ``_REGISTRY_PID``
(module-level, tracking whichever process last legitimately wrote the
registry) on every call: a mismatch means this process did not write what it
just inherited, so every entry is discarded — via
``_discard_inherited_registry``, which deliberately does NOT call
``.close()`` on any of them: closing writes an Arrow IPC end-of-stream
marker, and doing that to a file this process does not actually own (and may
still be legitimately open elsewhere) would corrupt it for whoever does. The
same guard covers process-exit cleanup in ``_close_all_streams``, for a
worker that was forked but never itself wrote anything before exiting.

An output's schema is normally inferred from the first batch a worker writes
for it — fine as long as every run's frame for that output infers the same
Arrow types. A run that returns a zero-row frame breaks this: an empty
column's inferred type doesn't reliably match the type inferred from a later
non-empty batch of the same output, so whichever run happens to write first
fixes a schema the next run's batch may not satisfy (a spurious per-run
WRITING failure). ``DiskStreamWriter``'s optional ``schemas`` parameter
(normally supplied via ``DiskStore(schemas=...)``, which is the only
construction path a user should go through — see ``_validate_schemas``)
sidesteps this: a user-supplied schema is fixed up front, independent of
write order. Outputs with no explicit schema keep the infer-from-first-batch
behavior and remain exposed to this corner case.

For an output with an explicit schema, ``_to_batch`` also reconciles the
frame's actual columns against it before ever calling into pyarrow's own
conversion: a frame missing a declared column is a hard ``WRITING`` failure,
while an undeclared extra column is dropped with a warning (schema is
authoritative). This is a deliberate design choice, not a fallback — it
means the frame handed to pyarrow always matches the declared schema
exactly, so pyarrow's own schema-mismatch error behavior (which may differ
across pyarrow versions and was not something this module wanted to depend
on) is never exercised.

Concurrency preconditions:

- One single-threaded worker process per writer. The registry is a bare dict
  with NO lock. A thread-backed executor (``ThreadPoolExecutor``) would
  violate this — threads share one process and, since threads (unlike
  process-pool workers) share Python objects directly rather than each
  getting an independently unpickled copy, would share the exact same
  writer instance and hence the exact same ``_uuid`` and registry entries,
  all ending up writing into one file per output. Interleaved
  ``write_batch`` calls would corrupt the IPC streams *silently* (no
  exception, unreadable file). To support threaded workers, restore a lock
  guarding every registry access below and give each thread its own stream
  key.
- ``run_scenarios`` is blocking, so a worker only ever serves one session at
  a time. Sessions can change either due to a resume or due to a second call
  to ``run_scenarios``.
"""

from __future__ import annotations

import atexit
import os
import re
import socket
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.ipc

from mesa.experimental.scenarios.store import RunId

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pandas as pd

# Output names become directory names, so they must be safe path segments.
# Reject anything outside this set rather than percent-encoding: the failure is
# loud and per-run, legitimate outcome names are identifier-like, and encoding
# would add a mapping that must round-trip through resume for no real use case.
# Leading dot disallowed (no hidden dirs, no "." / ".."); dot/dash allowed after.
_SAFE_OUTPUT_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*$")

# Appended to every output's batch regardless of whether that output has an
# explicit user schema (see DiskStreamWriter.schemas): identity columns are
# this module's doing, not the model's, so callers never specify them.
# int64 to match what pyarrow already infers today from frame.assign()'s
# plain python ints — this constant must not silently change that dtype for
# outputs that have no explicit schema.
_IDENTITY_SCHEMA = pa.schema(
    [pa.field("scenario_id", pa.int64()), pa.field("replication_id", pa.int64())]
)
_RESERVED_IDENTITY_COLUMNS = frozenset(_IDENTITY_SCHEMA.names)

# Eager and safe: gethostname() returns the same string whether it's called
# fresh in a child process or inherited via a fork-time memory copy, since it
# describes the physical machine, not per-process identity. Do NOT apply the
# same "make it lazy/instance-scoped" fix here that _uuid needed below —
# there is nothing to fix.
_HOST = socket.gethostname()

# NOT a module-level uuid — see the module docstring's ``uuid`` bullet.
# _instance_uuid() on DiskStreamWriter is where this actually lives now.


def _validate_schemas(schemas: dict[str, pa.Schema]) -> None:
    """Reject any output schema that declares a reserved identity column.

    Shared by ``DiskStore.__init__`` (the fail-fast path for real users, who
    only ever construct a ``DiskStore``) and ``DiskStreamWriter.__init__``
    (a safety net for tests that construct a writer in isolation, bypassing
    ``DiskStore`` entirely). Defined once, here, so the two call sites can
    never validate against a different set of reserved names than each other.

    Args:
        schemas: per-output Arrow schema, keyed by output name, as passed to
            either constructor's ``schemas`` parameter.

    Raises:
        ValueError: if any schema declares ``scenario_id`` or
            ``replication_id`` itself — both are appended automatically and
            must not appear in a user-supplied schema.
    """
    for name, schema in schemas.items():
        reserved = set(schema.names) & _RESERVED_IDENTITY_COLUMNS
        if reserved:
            raise ValueError(
                f"schema for output {name!r} declares reserved identity "
                f"column(s) {sorted(reserved)}; these are appended "
                "automatically and must not appear in a user-supplied "
                "schema"
            )


# ---------------------------------------------------------------------------
# Per-worker stream registry.
#
# Module-level => one copy per worker process (each process imports the module
# independently). Keyed by (session, output name) -> the open stream, its
# sink, and the schema fixed on that output's first write. Session is part of
# the key, not just the file name, because the worker PROCESS can outlive a
# single ``run_scenarios()`` invocation (e.g. a persistent executor reused for
# a later sweep or a resume): without it, a writer reconstructed with a new
# session would see the old session's stream already open under the same
# output name and silently keep appending to it instead of opening a new
# file. Populated lazily; reused across a worker's runs for the same session.
# NOT on the writer instance, which is re-pickled per job.
# ---------------------------------------------------------------------------
_StreamKey = tuple[str, str]  # (session, output name)
_STREAMS: dict[_StreamKey, pa.ipc.RecordBatchStreamWriter] = {}
_SINKS: dict[_StreamKey, pa.OSFile] = {}
_SCHEMAS: dict[_StreamKey, pa.Schema] = {}


# The session this worker process is currently serving. A worker serves one
# session at a time (it drains a sweep's jobs before the next sweep's arrive),
# so a writer bearing a different token signals a session transition: the prior
# session is finished on this worker and its streams can be closed.
_CURRENT_SESSION: str | None = None

# pid of whichever process last legitimately wrote the registry above — i.e.
# last ran _rotate_session to completion IN that process, as opposed to
# merely inheriting these dicts via a fork's memory copy. Compared against
# os.getpid() on every write and at process exit; see _rotate_session,
# _close_all_streams, and the module docstring's paragraph on the registry's
# own fork hazard.
_REGISTRY_PID: int | None = None


def _discard_inherited_registry() -> None:
    """Drop every registry entry without closing any of it.

    Used when this process's identity doesn't match whichever process last
    wrote these entries (see the pid check in ``_rotate_session`` and
    ``_close_all_streams``): every entry here was inherited via a fork's
    memory copy, not opened by this process. Their underlying OS
    resources — for ``_STREAMS`` specifically, a live Arrow IPC stream with
    its end-of-stream marker not yet written — may still be genuinely in use
    by the parent process or a sibling that inherited the same file
    descriptor. Calling ``.close()`` (as ``_evict`` does) would write into a
    file this process never opened and does not solely own, corrupting it
    for whichever process actually does. The correct move is to forget these
    references entirely; this process opens its own, independent files from
    here on, via the ordinary lazy-open path in ``_append``/``_open_stream``.
    """
    _STREAMS.clear()
    _SINKS.clear()
    _SCHEMAS.clear()


def _evict(keys: Iterable[_StreamKey]) -> None:
    """Close and drop the registry entries for ``keys``, best-effort.

    Shared by session rotation (evicting a stale session's entries) and
    process-exit cleanup (evicting everything) — both call sites only ever
    reach this after their own pid check confirms the entries being closed
    were genuinely opened by this process, not inherited from a fork (see
    ``_discard_inherited_registry`` for the alternative used otherwise).
    Closing writes each stream's end-of-stream marker; a close that fails
    (or is skipped entirely, at process exit under a hard kill) just leaves
    that file without a clean EOS marker, which the reader's
    truncation-tolerant path already handles — so failures here are
    swallowed rather than raised.
    """
    for key in keys:
        try:  # noqa: SIM105
            _STREAMS.pop(key).close()
        except Exception:  # noqa: S110
            pass
        try:  # noqa: SIM105
            _SINKS.pop(key).close()
        except Exception:  # noqa: S110
            pass
        _SCHEMAS.pop(key, None)


def _rotate_session(session: str) -> None:
    """Close all streams not belonging to ``session`` on a session change.

    Called on every write. Two independent staleness checks run here, in
    order:

    1. Has this process's identity changed since the registry was last
       written BY this process? (``os.getpid()`` vs. ``_REGISTRY_PID``.) A
       mismatch means the registry currently reflects nothing this process
       itself opened — either this is genuinely the first call ever, or
       this process is a fork of one that already had entries open, which
       fork's memory-copy semantics would otherwise let it silently inherit
       and write into. Either way, every entry is discarded (without
       closing — see ``_discard_inherited_registry``) before anything else
       runs.
    2. Has the incoming session changed within THIS process's own writing
       history? When it has, every prior session's stream is closed and
       evicted — this process genuinely owns those files, so closing them
       properly (writing their end-of-stream marker) is correct here,
       unlike case 1.

    This bounds open file descriptors to a single session's outputs even on
    a persistent worker reused across many sweeps or resumes, without any
    root-side coordination: both transitions are detected locally, from
    process identity and the token the writer already carries.
    """
    global _CURRENT_SESSION, _REGISTRY_PID  # noqa: PLW0603
    pid = os.getpid()
    if pid != _REGISTRY_PID:
        _discard_inherited_registry()
        _CURRENT_SESSION = None
        _REGISTRY_PID = pid

    if session == _CURRENT_SESSION:
        return
    _evict([key for key in _STREAMS if key[0] != session])
    _CURRENT_SESSION = session


def _close_all_streams() -> None:
    """Best-effort close of this worker's streams at process exit.

    A worker killed before this runs (SLURM timeout, ``os._exit``) leaves its
    streams with no EOS marker; the reader's truncation-tolerant path
    recovers every complete batch written before the kill, so this close is
    an optimization for clean shutdown, not a correctness requirement.

    Guarded by the same pid check as ``_rotate_session``: a worker that was
    forked but never actually wrote anything (e.g. an idle pool worker never
    dispatched a task) can still exit with inherited-but-never-revalidated
    registry entries. Closing those would be exactly the hazard
    ``_discard_inherited_registry`` exists to avoid — this process never
    opened them and may not be their sole owner.
    """
    if os.getpid() != _REGISTRY_PID:
        _discard_inherited_registry()
        return
    _evict(list(_STREAMS))


atexit.register(_close_all_streams)


@dataclass(frozen=True)
class DiskReference:
    """Key-only reference to a run's persisted outcome.

    Carries no path and no payload: the data lives in this worker's Arrow
    files, and retrieval is a filter on the whole-sweep read by ``run_id``.
    Pickleable and tiny, so returning the data as payload to the root is redundant.
    """

    run_id: RunId

    @property
    def payload(self) -> None:
        """No payload; the outcome is on disk, not carried by the reference."""
        return None


class DiskStreamWriter:
    """Stateless, per-job-pickleable handle that appends run outcomes to disk.

    Carries only configuration — the store directory, the session token, a
    lazily-minted uuid, and an optional per-output schema map. All mutable
    state about open streams and fixed schemas lives in the module-level
    registry, keyed per output and per worker process. Constructed by
    ``DiskStore.writer()`` and re-pickled with every job.

    Normally reached only via ``DiskStore.writer()``, which validates
    ``schemas`` once at ``DiskStore`` construction time. This class validates
    it again in its own ``__init__`` — redundant on that normal path, but
    this class is also constructed directly and in isolation by tests, and
    a bad ``schemas`` dict should fail right there rather than surface later,
    confusingly, from inside ``_to_batch``.
    """

    def __init__(
        self,
        store_dir: Path,
        session: str,
        schemas: dict[str, pa.Schema] | None = None,
    ):
        """Initialize the writer.

        Args:
            store_dir: root directory of the store. ``outputs/`` beneath it
                holds one subdirectory per named output.
            session: token of the current invocation, minted by the root and
                embedded in every file this writer opens.
            schemas: optional per-output Arrow schema, keyed by output name,
                describing that output's columns exactly as returned by
                ``extract_output`` — identity columns are appended
                automatically and must not appear here. When present for an
                output, its stream is fixed to this schema (plus identity
                columns) on first open, instead of inferring one from the
                first batch written — see ``_to_batch``. A frame missing a
                declared column is a hard failure; a frame with an extra,
                undeclared column is warned about once and the column
                dropped (schema is authoritative). Outputs absent here keep
                the existing infer-from-first-batch behavior.

        Raises:
            ValueError: if any schema in ``schemas`` declares a reserved
                identity column — see ``_validate_schemas``.
        """
        self.store_dir = Path(store_dir)
        self.session = session
        self.schemas = schemas or {}
        _validate_schemas(self.schemas)
        # Must stay None here — never set eagerly. See the module
        # docstring's ``uuid`` bullet: an eagerly-set value would be baked
        # into every pickled copy of this writer, defeating the entire
        # point of scoping it to the instance. Minted lazily, on first use,
        # by _instance_uuid().
        self._uuid: str | None = None

    def _instance_uuid(self) -> str:
        """Return this writer instance's uuid, minting it on first call.

        Stored on the instance, not the module — see the module docstring's
        ``uuid`` bullet for why a module-level value, even a lazily-computed
        one, is not sufficient to guarantee uniqueness across forked
        workers.
        """
        if self._uuid is None:
            self._uuid = uuid.uuid4().hex
        return self._uuid

    def _file_for(self, output_name: str) -> Path:
        """Path to this worker's Arrow file for a named output."""
        filename = f"worker-{self.session}-{_HOST}-{self._instance_uuid()}.arrow"
        return self.store_dir / "outputs" / output_name / filename

    def to_reference(
        self, run_id: RunId, outcome: dict[str, pd.DataFrame]
    ) -> DiskReference:
        """Persist a run's outcome and return a key-only reference.

        Validates every named output first, then writes every output — the
        check is atomic across the whole outcome, so a run either contributes a
        full batch to each of its streams or nothing at all, never a partial
        set. Raises on failure; the worker-side ``_safe_call`` converts the
        raised exception into a recorded ``WRITING`` ``FailureInfo`` so one bad
        run never aborts the sweep.

        Assumes extraction's postcondition has already run: every frame in
        ``outcome`` has at least one column. A columnless frame is rejected
        upstream in ``RunConfiguration.__call__`` as an ``EXTRACTING`` failure,
        so it never reaches here. Calling this directly with a columnless frame
        would produce a zero-column batch rather than that clean failure.

        Args:
            run_id: identity of the run being persisted.
            outcome: mapping of output name to tidy DataFrame, straight from
                ``extract_output``. Treated as untrusted data.

        Returns:
            A ``DiskReference`` keyed by ``run_id``.
        """
        _rotate_session(self.session)

        # --- validate + convert all outputs before writing any of them ---
        # (atomic pre-write: a schema-deviating or non-convertible frame in one
        # output must not leave earlier outputs half-written for this run.)
        batches: dict[str, pa.RecordBatch] = {}
        for name, frame in outcome.items():
            self._validate_name(name)
            batches[name] = self._to_batch(run_id, name, frame)

        # schema stability is part of validation: check every output's schema
        # against its fixed per-worker schema before writing any of them.
        for name, batch in batches.items():
            self._check_schema(name, batch.schema)

        # --- all validated; now write ---
        # fixme a write in this loop might fail, leaving partially written output
        #   resume semantics should handle this explicitly
        for name, batch in batches.items():
            self._append(name, batch)

        return DiskReference(run_id)

    @staticmethod
    def _validate_name(name: str) -> None:
        """Reject output names that are unsafe as path segments.

        Raised as a plain ``ValueError``; ``_safe_call`` records it as a
        ``WRITING`` failure. Kept a write-stage concern because the hazard is
        the name becoming a directory, which is this module's doing.
        """
        if not isinstance(name, str) or not _SAFE_OUTPUT_NAME.match(name):
            raise ValueError(
                f"invalid output name {name!r}: must match "
                f"{_SAFE_OUTPUT_NAME.pattern!r} (identifier-like; no path "
                "separators, no leading dot)"
            )

    def _full_schema(self, name: str) -> pa.Schema | None:
        """This output's fixed schema (user schema + identity columns), or None.

        None means no explicit schema was supplied for ``name``: pyarrow
        infers the batch's schema from the frame instead, and the first
        batch a worker writes for this output fixes it going forward
        (existing ``_check_schema``/registry behavior, unchanged).
        """
        user_schema = self.schemas.get(name)
        if user_schema is None:
            return None
        return pa.schema(list(user_schema) + list(_IDENTITY_SCHEMA))

    @staticmethod
    def _conform_to_schema(
        name: str, user_schema: pa.Schema, frame: pd.DataFrame
    ) -> pd.DataFrame:
        """Reconcile a frame's columns against its declared user schema.

        Compares column names only (types are handled by ``from_pandas``
        itself once the columns line up — a present-but-wrong-type column is
        still caught there, by the existing except clause in ``_to_batch``).
        Runs before identity columns are attached, so ``user_schema`` here is
        the declared schema alone, not ``_full_schema``'s identity-appended
        version — identity columns are always added afterward and are never
        something the frame is expected to already carry.

        Args:
            name: output name, for error/warning messages.
            user_schema: this output's declared schema, as supplied by the
                caller (not yet identity-appended).
            frame: the extracted output frame, as returned by
                ``extract_output``.

        Returns:
            ``frame``, unchanged if its columns already match ``user_schema``
            exactly; otherwise a copy with extra columns dropped.

        Raises:
            ValueError: if the frame is missing any column ``user_schema``
                declares. A missing column is always a hard failure — unlike
                an extra column, there is no reasonable value to fill in for
                a required column, so silently proceeding would produce a
                batch that doesn't match its own declared schema.
        """
        declared = set(user_schema.names)
        present = set(frame.columns)

        missing = declared - present
        if missing:
            raise ValueError(
                f"output {name!r} is missing column(s) {sorted(missing)} "
                "required by its declared schema"
            )

        extra = present - declared
        if extra:
            # Message deliberately excludes run_id: warnings.warn's registry
            # dedups per process by (message, category, lineno), so a static
            # message means each worker warns at most once per output per
            # distinct extra-column set, not once per run. A true sweep-wide
            # single warning would need workers to coordinate back to the
            # root, which is more machinery than a courtesy message
            # justifies.
            warnings.warn(
                f"output {name!r} has column(s) {sorted(extra)} not in its "
                "declared schema; dropping them (schema is authoritative)",
                UserWarning,
                stacklevel=3,
            )
            frame = frame[list(user_schema.names)]

        return frame

    def _to_batch(
        self, run_id: RunId, name: str, frame: pd.DataFrame
    ) -> pa.RecordBatch:
        """Convert one output frame to an Arrow batch with identity columns.

        Extraction has already guaranteed the frame has columns (a columnless
        frame is an ``EXTRACTING`` failure in ``RunConfiguration``), so this is
        purely a write concern: reconcile against any declared schema, attach
        identity columns, and convert. A columned zero-row frame is valid —
        the resulting zero-row batch preserves the schema. A frame that will
        not convert to Arrow raises here (re-wrapped for a clearer message
        than pyarrow's raw error) and is recorded as a ``WRITING`` failure by
        the worker-side ``_safe_call``.

        If an explicit schema was supplied for ``name``, the frame's columns
        are reconciled against it first via ``_conform_to_schema`` — a
        missing declared column raises, an extra undeclared column is warned
        about once and dropped. This runs before pyarrow ever sees the frame,
        so the batch handed to ``from_pandas`` always has exactly the
        declared columns; pyarrow's own behavior for a schema/frame mismatch
        (confirmed: a schema field absent from the frame raises, an extra
        frame column absent from the schema is silently ignored) is
        deliberately never exercised, since ``_conform_to_schema`` already
        resolved both cases before this point.

        Identity columns are attached via ``assign``, which shallow-copies the
        frame and only allocates the two new columns — the caller's frame is
        left untouched without paying to deep-copy its existing data. The
        scalars broadcast across the frame's row count, including zero rows.

        For a schema'd output, the batch is then built against the full
        schema (user columns + identity columns) directly rather than left to
        pyarrow's inference: this casts the frame's columns to the declared
        types. ``RecordBatch.from_pandas`` has no ``safe`` parameter (unlike
        ``Table.from_pandas``) — confirmed against Arrow's docs across every
        released version, not a version-specific gap — so casting behavior on
        a lossy conversion (e.g. float64 to a declared int32) is whatever
        pyarrow does internally; it is not pinned or controlled by this
        module. This is still what removes the empty-vs-populated-frame
        schema-mismatch corner case for that output — see the ``schemas``
        parameter on ``__init__``.
        """
        user_schema = self.schemas.get(name)
        if user_schema is not None:
            frame = self._conform_to_schema(name, user_schema, frame)

        frame = frame.assign(
            scenario_id=run_id.scenario_id, replication_id=run_id.replication_id
        )
        try:
            return pa.RecordBatch.from_pandas(
                frame, schema=self._full_schema(name), preserve_index=False
            )
        except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError) as e:
            raise ValueError(
                f"output {name!r} for {run_id} is not Arrow-convertible: {e}"
            ) from e

    def _check_schema(self, name: str, schema: pa.Schema) -> None:
        """Validate a batch's schema against this worker's fixed schema.

        First write to an output fixes its schema; a later batch whose schema
        differs fails the within-worker stability check. Cross-worker schema
        deviation is not checked here — it is reconciled by the reader
        (unify + null-fill + warn). A not-yet-opened output passes: its schema
        is fixed when ``_append`` opens the stream.

        For an output with an explicit user schema, every batch already
        carries that identical fixed schema (``_to_batch``/``_full_schema``
        return the same schema object shape on every call for a given
        output, and ``_conform_to_schema`` guarantees the frame's columns
        match it before conversion), so this check never fires for it in
        practice. It stays in the write path unconditionally rather than
        being skipped for schema'd outputs, so a future change to
        ``_full_schema`` that made it derive something per-call would still
        be caught here instead of silently writing an inconsistent stream.
        """
        fixed = _SCHEMAS.get((self.session, name))
        if fixed is not None and not schema.equals(fixed):
            raise ValueError(
                f"schema for output {name!r} changed within this worker's "
                f"stream.\n expected: {fixed}\n got:      {schema}"
            )

    def _append(self, name: str, batch: pa.RecordBatch) -> None:
        """Append one batch to this worker's stream for ``name``.

        Opens the stream lazily on first write, fixing the per-worker schema to
        this batch's schema. Schema stability was already checked in
        ``_check_schema`` before any write in this run began.
        """
        key = (self.session, name)
        if key not in _STREAMS:
            self._open_stream(name, batch.schema)
        _STREAMS[key].write_batch(batch)

    def _open_stream(self, name: str, schema: pa.Schema) -> None:
        """Open a new IPC stream for ``name``, creating its output directory."""
        key = (self.session, name)
        path = self._file_for(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        sink = pa.OSFile(str(path), "wb")
        _SINKS[key] = sink
        _STREAMS[key] = pa.ipc.new_stream(sink, schema)
        _SCHEMAS[key] = schema
