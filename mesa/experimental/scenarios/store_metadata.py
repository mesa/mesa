"""Durably store metadata: store manifest, scenario manifest, and status log.

Store-agnostic; reusable by any persistent store implementation. It deals only
with the small root-written files that describe a parameter sweep:

- ``store.json``      written first, read first; carries the version number
                      of the storage format, a list of all sessions (in
                      case of e.g., resume after a timeout), and provenance
                      information about which mesa version and, if available,
                      which version of the model files.
- ``scenarios.json``  the authoritative, seed-complete ensemble of scenarios.
- ``status.log``      A log of the status of each scenario run. Append-only JSON
                      lines, root-written, replayed with tolerance for a torn
                      final line. This log only contains ABORTED, SUCCEEDED,
                      or FAILED. By comparing with scenarios.json, PENDING runs
                      can easily be identified. Every line records the session
                      that produced it. A SUCCEEDED line is the *commit point*
                      of a run: the run's data counts only if it lies in the
                      files of the session named on that line.

"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from mesa.experimental.scenarios.exceptions import FailureInfo, FailureOrigin
from mesa.experimental.scenarios.store import RunId, Status

if TYPE_CHECKING:
    from mesa import Model
    from mesa.experimental.scenarios.scenario import Scenario

#: Bump on any change to file layout, filename conventions, or the schema of
#: the files below. ``DiskStore.from_directory`` refuses other versions.
#: Version 2: every ``status.log`` line records the ``session`` that wrote it.
STORE_FORMAT_VERSION = 2

STORE_MANIFEST = "store.json"
SCENARIOS_MANIFEST = "scenarios.json"
STATUS_LOG = "status.log"


def _json_default(obj: object) -> Any:
    """Convert numpy scalars to JSON-native types; reject everything else."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(
        f"{type(obj).__name__} is not JSON-serializable; scenario parameters "
        "must be JSON-native or numpy scalars"
    )


# ---------------------------------------------------------------------------
# store.json
# ---------------------------------------------------------------------------


def write_store_manifest(
    store_dir: Path,
    *,
    session: str,
    provenance: dict[str, Any] | None = None,
) -> None:
    """Write ``store.json``. Fails if it already exists.

    Args:
        store_dir: root directory of the store.
        session: token of the creating session (first entry of the session list).
        provenance: informational, non-gating metadata to record alongside the
            manifest. Build this via ``collect_provenance`` before calling.

    We dump the manifest to a string first to avoid writing partial data to the
    file if an error occurs during serialization.

    """
    manifest = {
        "store_format_version": STORE_FORMAT_VERSION,
        "sessions": [session],
        "provenance": provenance or {},
    }

    content = json.dumps(
        manifest,
        indent=2,
    )

    path = store_dir / STORE_MANIFEST
    with path.open("x") as f:  # "x": creation is a one-time event; fail loud
        f.write(content)


def read_store_manifest(store_dir: Path) -> dict[str, Any]:
    """Read and return ``store.json`` as a dict."""
    with (store_dir / STORE_MANIFEST).open() as f:
        return json.load(f)


def add_session(store_dir: Path, session: str) -> None:
    """Append a session token to the manifest's ordered session list.

    Args:
        store_dir: root directory of the store.
        session: session token.

    This function is only executed from a root. Appending a session token
    means rewriting the entire file, since we cannot append to a JSON file.
    The rewrite is atomic (write to a temp file, then
    ``os.replace``), so a reader never observes a half-written manifest.

    This is the only read-modify-write in the store; it is safe because
    exactly one root process exists per session.
    """
    manifest = read_store_manifest(store_dir)
    manifest["sessions"].append(session)
    tmp = store_dir / (STORE_MANIFEST + ".tmp")
    with tmp.open("w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, store_dir / STORE_MANIFEST)


def _mesa_version() -> str | None:
    """Private helper function which returns mesa version."""
    try:
        import mesa  # noqa: PLC0415

        return getattr(mesa, "__version__", None)
    except Exception:
        return None


def _model_git_provenance(model_class: type | None) -> dict[str, Any]:
    """Best-effort git commit + dirty flag for the tree containing the model.

    Locates the model class's source file and runs git against its
    directory, so the recorded commit describes the user's model code —
    the code whose provenance a resumer actually cares about — not Mesa's
    installed source. Returns {} if the model isn't in a checkout, its
    source can't be located (e.g. defined in a notebook or REPL), or git
    is unavailable. Never raises.

    This requires that git is available from the command line.

    """
    if model_class is None:
        return {}

    try:
        source_file = inspect.getfile(model_class)
    except (TypeError, OSError):
        # builtin, dynamically generated, or no source on disk
        return {}

    repo = Path(source_file).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        dirty = (
            subprocess.run(
                ["git", "status", "--porcelain"],  # noqa: S607
                cwd=repo,
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            ).stdout.strip()
            != ""
        )
        return {"commit": commit, "dirty": dirty, "source_path": source_file}
    except Exception:
        return {}


def collect_provenance(
    *,
    model_class: type[Model] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble informational, non-gating store provenance.

    Every field is best-effort: anything that cannot be determined is
    omitted rather than raising, because provenance never gates resume —
    it only lets a human judge whether a store's runs came from code they
    trust.
    """
    provenance: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
    }

    if (mesa_version := _mesa_version()) is not None:
        provenance["mesa_version"] = mesa_version

    git = _model_git_provenance(model_class)  # {"commit": ..., "dirty": ...} or {}
    if git:
        provenance["git"] = git

    if extra:
        provenance["extra"] = extra

    return provenance


# ---------------------------------------------------------------------------
# scenarios.json
# ---------------------------------------------------------------------------


def write_scenarios_manifest(store_dir: Path, scenarios: list[Scenario]) -> None:
    """Write the authoritative, seed-complete scenario manifest.

    Args:
        store_dir: root directory of the store.
        scenarios: scenarios to write.

    Serializes each scenario via ``to_dict()``: user parameters plus
    ``scenario_id``, ``replication_id``, ``seed_sequence_entropy``,
    ``seed_sequence_spawn_key``, and ``generator_class``. Together these
    round-trip the SeedSequence bit-exactly.

    The other parameters in a scenario are cast to python native types. So e.g.,
    a np.int64 is cast to a python int.

    We dump the manifest to a string first to avoid writing partial data to the file
    if an error occurs during serialization.

    """
    entries = [scenario.to_dict() for scenario in scenarios]
    content = json.dumps(entries, default=_json_default, indent=2)
    with (store_dir / SCENARIOS_MANIFEST).open("x") as f:
        f.write(content)


def read_scenarios_manifest(
    store_dir: Path, scenario_class: type[Scenario]
) -> list[Scenario]:
    """Reconstruct scenarios from the manifest.

    Args:
        store_dir: root directory of the store.
        scenario_class: the concrete Scenario subclass to instantiate. JSON
            cannot carry the class safely, so the caller supplies it — this
            matches how resume (PR 8) will invoke it.

    Returns:
        Scenario instances with bit-exact SeedSequences and the original
        generator class, in manifest order.
    """
    with (store_dir / SCENARIOS_MANIFEST).open() as f:
        entries = json.load(f)

    scenarios = []
    for entry in entries:
        entropy = entry.pop("seed_sequence_entropy")
        spawn_key = tuple(entry.pop("seed_sequence_spawn_key"))
        generator_class = entry.pop("generator_class")
        scenario_id = entry.pop("scenario_id")
        replication_id = entry.pop("replication_id")

        seed_seq = np.random.SeedSequence(entropy, spawn_key=spawn_key)
        try:
            bg_class = getattr(np.random, generator_class)
        except AttributeError as e:
            # mirrors Scenario.__setstate__
            raise NotImplementedError(
                "only default numpy generators are currently supported"
            ) from e
        # default_rng passes a Generator through unchanged, so the
        # reconstructed scenario keeps this bit generator class rather than
        # being collapsed to PCG64.
        rng = np.random.Generator(bg_class(seed_seq))

        scenarios.append(
            scenario_class(
                rng=rng,
                scenario_id=scenario_id,
                replication_id=replication_id,
                **entry,
            )
        )
    return scenarios


# ---------------------------------------------------------------------------
# status.log
# ---------------------------------------------------------------------------


class StatusEntry(NamedTuple):
    """The last recorded status of one run, as replayed from ``status.log``."""

    status: Status
    failure: FailureInfo | None
    session: str


class StatusLog:
    """Root-side, append-only handle on ``status.log``.

    Keeps one line-buffered file handle open instead of reopening the file for
    every status line, which on a shared filesystem costs a metadata round trip
    per run. Line buffering hands every completed line to the OS immediately,
    so a root killed at any point loses at most the line it was writing (the
    reader tolerates a torn final line).

    Surviving a *node* crash needs an ``fsync``. It happens on the first append
    after ``sync_interval`` seconds have passed, and on ``close``. Losing the
    unsynced tail is safe by design: a run whose status line is missing reads as
    PENDING and is simply run again. Data it wrote is ignored, because a run's
    data only counts in the session named on its SUCCEEDED line.

    Not safe for concurrent writers; only the root writes the log.
    """

    def __init__(self, store_dir: Path, *, sync_interval: float = 5.0):
        """Open (creating if needed) ``status.log`` in ``store_dir`` for appending.

        Args:
            store_dir: root directory of the store.
            sync_interval: minimum seconds between ``fsync`` calls while appending.
        """
        self._file = (store_dir / STATUS_LOG).open("a", buffering=1)
        self._sync_interval = sync_interval
        self._last_sync = time.monotonic()
        self._dirty = False

    def append(
        self,
        run_id: RunId,
        status: Status,
        failure: FailureInfo | None = None,
        *,
        session: str,
    ) -> None:
        """Append one status line.

        Args:
            run_id: run_id of the current run.
            status: status of the current run.
            failure: failure info from the current run.
            session: the session that produced this status. For SUCCEEDED
                this is the session whose files hold the run's data.
        """
        line: dict[str, Any] = {
            "scenario_id": run_id.scenario_id,
            "replication_id": run_id.replication_id,
            "status": status.value,
            "session": session,
        }
        if failure is not None:
            line["failure"] = {
                "origin": failure.origin.value,
                "exception_type": failure.exception_type,
                "message": failure.message,
                "traceback": failure.traceback,
            }
        self._file.write(json.dumps(line) + "\n")
        self._dirty = True
        if time.monotonic() - self._last_sync >= self._sync_interval:
            self.sync()

    def sync(self) -> None:
        """Force everything appended so far to stable storage."""
        if self._file.closed or not self._dirty:
            return
        self._file.flush()
        os.fsync(self._file.fileno())
        self._dirty = False
        self._last_sync = time.monotonic()

    def close(self) -> None:
        """Sync and close the handle. Safe to call more than once."""
        if self._file.closed:
            return
        self.sync()
        self._file.close()

    def __enter__(self) -> StatusLog:
        """Return self, for use as a context manager."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the log on leaving the ``with`` block."""
        self.close()


def read_status(
    store_dir: Path,
) -> dict[RunId, StatusEntry]:
    """Read the status log and return the latest read status for each unique run.

    Args:
        store_dir: root directory of the store.

    Tolerates exactly one torn line: the final one, which a hard kill of the
    root mid-append can leave behind. It is dropped with a warning — the run
    it recorded simply reads as PENDING, which resume handles by re-running.
    A malformed line anywhere *before* the tail is not a torn-tail artifact
    but corruption, and raises.

    Returns:
        Mapping of RunId to its final ``StatusEntry`` (status, failure or
        None, and the session that recorded it). Runs never marked do not
        appear; callers derive PENDING from the scenario manifest minus this
        mapping.
    """
    path = store_dir / STATUS_LOG
    result: dict[RunId, StatusEntry] = {}
    if not path.exists():
        return result

    lines = path.read_text().splitlines()
    last = len(lines) - 1
    for i, raw in enumerate(lines):
        if not raw.strip():
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError as e:
            if i == last:
                warnings.warn(
                    f"dropping torn final line of {STATUS_LOG}",
                    UserWarning,
                    stacklevel=2,
                )
                break
            raise ValueError(
                f"corrupt {STATUS_LOG}: undecodable line {i + 1} of "
                f"{len(lines)} is not a torn tail"
            ) from e

        run_id = RunId(entry["scenario_id"], entry["replication_id"])
        failure = None
        if (f := entry.get("failure")) is not None:
            failure = FailureInfo(
                origin=FailureOrigin(f["origin"]),
                exception_type=f["exception_type"],
                message=f["message"],
                traceback=f["traceback"],
            )
        result[run_id] = StatusEntry(Status(entry["status"]), failure, entry["session"])
    return result
