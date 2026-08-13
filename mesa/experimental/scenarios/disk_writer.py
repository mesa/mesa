"""DiskStore writer — worker-side Arrow IPC streaming.

The writer is re-pickled and handed in with every job (matching run_scenarios'
dispatch via ``store.writer()``), so it CANNOT hold open streams on itself:
each job receives a fresh copy with no streams. Instead, the open streams live
in a module-level per-process registry. The worker PROCESS persists across
jobs even though the writer OBJECT is reconstructed each task, so a stream
opened on a worker's first run is reused by its later runs.

The filename ``{output}/worker-{session}-{host}-{pid}.arrow`` keeps each
worker's file distinct along every axis that can collide:
  - ``pid``      distinguishes workers within one node;
  - ``host``     distinguishes workers across nodes (pid spaces are per-node,
                 so two nodes can share a pid — relevant once PR 6 adds MPI);
  - ``session``  distinguishes invocations (resume) — an Arrow IPC
                 stream cannot be reopened once its end-of-stream marker is
                 written, so a resumed sweep opens NEW files rather than
                 appending to a prior session's.

Data is written where it is produced; only a key-only ``DiskReference`` crosses back
to the root. That is the whole point of a worker-side writer.

fixme: currently schema is created after the first succesfull run on each worker
  these might however  diverge across workers, or if the first call returns an
  empty frame, it might cause problems for subsequent runs. A future PR will
  add an optional schema to RunConfiguration.

Concurrency preconditions:

- One single-threaded worker process per writer. The registry is a bare dict
  with NO lock. A thread-backed executor (``ThreadPoolExecutor``) would
  violate this — all threads share one pid, hence one set of registry
  entries and one file per output, and interleaved ``write_batch`` calls
  would corrupt the IPC streams *silently* (no exception, unreadable file).
  To support threaded workers, restore a lock guarding every registry access
  below and give each thread its own stream key.
- ``run_scenarios`` is blocking, so a worker only ever serves one session at
  a time. Sessions can change either due to a resume or due to a second call
  to ``run_scenarios``.
"""

from __future__ import annotations

import atexit
import os
import re
import socket
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

# Fixed for the lifetime of this worker process; computed once rather than on
# every ``_file_for`` call.
_HOST = socket.gethostname()
_PID = os.getpid()  # fixme: might still be reused if a worker fails and in respawned.


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


def _evict(keys: Iterable[_StreamKey]) -> None:
    """Close and drop the registry entries for ``keys``, best-effort.

    Shared by session rotation (evicting a stale session's entries) and
    process-exit cleanup (evicting everything). Closing writes each stream's
    end-of-stream marker; a close that fails (or is skipped entirely, at
    process exit under a hard kill) just leaves that file without a clean EOS
    marker, which the reader's truncation-tolerant path already handles — so
    failures here are swallowed rather than raised.
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

    Called on every write. When the incoming session differs from the one this
    worker was serving, every prior session's stream is closed and evicted.
    This bounds open file descriptors to a single session's outputs even on a
    persistent worker reused across many sweeps or resumes, without any
    root-side coordination: the transition is detected locally from the token
    the writer already carries.
    """
    global _CURRENT_SESSION  # noqa: PLW0603
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
    """
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

    Carries only configuration — the store directory and the session token.
    All mutable state (open streams, fixed schemas) lives in the module-level
    registry, keyed per output and per worker process. Constructed by
    ``DiskStore.writer()`` and re-pickled with every job.
    """

    def __init__(self, store_dir: Path, session: str):
        """Initialize the writer.

        Args:
            store_dir: root directory of the store. ``outputs/`` beneath it
                holds one subdirectory per named output.
            session: token of the current invocation, minted by the root and
                embedded in every file this writer opens.
        """
        self.store_dir = Path(store_dir)
        self.session = session

    def _file_for(self, output_name: str) -> Path:
        """Path to this worker's Arrow file for a named output."""
        filename = f"worker-{self.session}-{_HOST}-{_PID}.arrow"
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

    def _to_batch(
        self, run_id: RunId, name: str, frame: pd.DataFrame
    ) -> pa.RecordBatch:
        """Convert one output frame to an Arrow batch with identity columns.

        Extraction has already guaranteed the frame has columns (a columnless
        frame is an ``EXTRACTING`` failure in ``RunConfiguration``), so this is
        purely a write concern: attach identity columns and convert. A columned
        zero-row frame is valid — the resulting zero-row batch preserves the
        schema. A frame that will not convert to Arrow raises here (re-wrapped
        for a clearer message than pyarrow's raw error) and is recorded as a
        ``WRITING`` failure by the worker-side ``_safe_call``.

        Identity columns are attached via ``assign``, which shallow-copies the
        frame and only allocates the two new columns — the caller's frame is
        left untouched without paying to deep-copy its existing data. The
        scalars broadcast across the frame's row count, including zero rows.
        """
        frame = frame.assign(
            scenario_id=run_id.scenario_id, replication_id=run_id.replication_id
        )
        try:
            return pa.RecordBatch.from_pandas(frame, preserve_index=False)
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
