"""Tests for mesa.experimental.scenarios.store_metadata.

These tests focus on reading and writing the metadata only. No Store class is involved.
"""

import importlib.util
import json
import subprocess
import sys

import numpy as np
import pytest

from mesa.experimental.scenarios import Scenario
from mesa.experimental.scenarios.exceptions import FailureInfo, FailureOrigin
from mesa.experimental.scenarios.store import RunId, Status
from mesa.experimental.scenarios.store_metadata import (
    SCENARIOS_MANIFEST,
    STATUS_LOG,
    STORE_FORMAT_VERSION,
    STORE_MANIFEST,
    add_session,
    append_status,
    collect_provenance,
    read_scenarios_manifest,
    read_status,
    read_store_manifest,
    write_scenarios_manifest,
    write_store_manifest,
)


def _git_available() -> bool:
    """Check that git actually runs, not just that something is on PATH."""
    try:
        subprocess.run(
            ["git", "--version"],  # noqa: S607
            capture_output=True,
            check=True,
            timeout=5,
        )
        return True
    except (OSError, subprocess.SubprocessError):
        return False


requires_git = pytest.mark.skipif(
    not _git_available(), reason="git not available on PATH"
)


def _load_module(path, name):
    """Import a standalone .py file as a real module, registered in sys.modules.

    inspect.getfile (used by _model_git_provenance) resolves a class's source
    file via sys.modules, so a module built via module_from_spec must be
    registered there, or it looks source-less regardless of where its .py
    file actually lives.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ============================================================
# store.json
# ============================================================


def test_write_and_read_store_manifest(tmp_path):
    """store.json round-trips format version, session, and provenance."""
    write_store_manifest(tmp_path, session="s1", provenance={"note": "hello"})

    assert (tmp_path / STORE_MANIFEST).exists()
    manifest = read_store_manifest(tmp_path)
    assert manifest["store_format_version"] == STORE_FORMAT_VERSION
    assert manifest["sessions"] == ["s1"]
    assert manifest["provenance"] == {"note": "hello"}


def test_write_store_manifest_defaults_provenance_to_empty_dict(tmp_path):
    """Omitting provenance records an empty dict, not null or a missing key."""
    write_store_manifest(tmp_path, session="s1")

    manifest = read_store_manifest(tmp_path)
    assert manifest["provenance"] == {}


def test_write_store_manifest_fails_if_already_exists(tmp_path):
    """A second write to the same directory fails loud rather than overwriting."""
    write_store_manifest(tmp_path, session="s1")

    with pytest.raises(FileExistsError):
        write_store_manifest(tmp_path, session="s2")

    # the original manifest is untouched
    assert read_store_manifest(tmp_path)["sessions"] == ["s1"]


def test_add_session_appends_and_preserves_rest_of_manifest(tmp_path):
    """add_session appends to the session list via an atomic rewrite."""
    write_store_manifest(tmp_path, session="s1", provenance={"note": "hello"})

    add_session(tmp_path, "s2")
    add_session(tmp_path, "s3")

    manifest = read_store_manifest(tmp_path)
    assert manifest["sessions"] == ["s1", "s2", "s3"]
    assert manifest["store_format_version"] == STORE_FORMAT_VERSION
    assert manifest["provenance"] == {"note": "hello"}

    # the atomic-rewrite temp file never lingers
    assert not (tmp_path / (STORE_MANIFEST + ".tmp")).exists()


# ============================================================
# collect_provenance
# ============================================================


def test_collect_provenance_includes_created_at_and_mesa_version():
    """created_at and mesa_version are always present; git/extra are not, by default."""
    provenance = collect_provenance()

    assert "created_at" in provenance
    assert "mesa_version" in provenance
    assert "git" not in provenance
    assert "extra" not in provenance


def test_collect_provenance_includes_extra():
    """Extra is passed through verbatim under its own key."""
    provenance = collect_provenance(extra={"foo": "bar"})
    assert provenance["extra"] == {"foo": "bar"}


def test_collect_provenance_no_git_key_for_unlocatable_source():
    """A model class with no locatable source file (e.g. a builtin) records no git block."""
    provenance = collect_provenance(model_class=int)
    assert "git" not in provenance


class _ProvenanceDummyModel:
    """Standalone class living in this test file.

    Used only to give collect_provenance a real, git-tracked source file to
    inspect.
    """


@requires_git
def test_collect_provenance_git_key_present_for_class_in_real_checkout():
    """A model class whose source lives in this repo's checkout records a git block.

    This test file is itself part of the mesa git checkout, so a class
    defined here gives ``inspect.getfile`` a real, tracked source path
    without needing to fabricate a repository.
    """
    provenance = collect_provenance(model_class=_ProvenanceDummyModel)

    assert "git" in provenance
    assert "commit" in provenance["git"]
    assert "source_path" in provenance["git"]
    assert isinstance(provenance["git"]["dirty"], bool)


@requires_git
def test_collect_provenance_no_git_key_outside_any_repo(tmp_path):
    """No git block is recorded when the model's source isn't inside a git checkout."""
    module_path = tmp_path / "standalone_model.py"
    module_path.write_text("class Foo:\n    pass\n")
    try:
        module = _load_module(module_path, "standalone_model")
        provenance = collect_provenance(model_class=module.Foo)
    finally:
        sys.modules.pop("standalone_model", None)

    assert "git" not in provenance


@requires_git
def test_collect_provenance_git_dirty_flag_toggles(tmp_path):
    """Dirty reflects the actual working-tree state of a hermetic, throwaway repo."""
    repo = tmp_path / "repo"
    repo.mkdir()
    module_path = repo / "dummy_model.py"
    module_path.write_text("class Foo:\n    pass\n")

    try:
        # Import before committing: this generates a __pycache__ entry that
        # would otherwise show up as untracked and falsely mark the tree
        # dirty. Committing it up front makes the post-commit baseline
        # genuinely clean.
        module = _load_module(module_path, "dummy_model")

        run = lambda *args: subprocess.run(  # noqa: E731
            args, cwd=repo, check=True, capture_output=True, text=True
        )
        run("git", "init")
        run("git", "config", "user.email", "test@example.com")
        run("git", "config", "user.name", "Test")
        run("git", "add", ".")
        run("git", "commit", "-m", "init")

        provenance = collect_provenance(model_class=module.Foo)
        assert provenance["git"]["dirty"] is False

        module_path.write_text("class Foo:\n    x = 1\n")

        provenance = collect_provenance(model_class=module.Foo)
        assert provenance["git"]["dirty"] is True
    finally:
        sys.modules.pop("dummy_model", None)


# ============================================================
# scenarios.json
# ============================================================


def test_write_and_read_scenarios_manifest_round_trips_parameters(tmp_path):
    """User parameters and identity fields survive a write/read round trip."""
    scenarios = [Scenario(rng=i, density=0.5, label="x") for i in range(3)]

    write_scenarios_manifest(tmp_path, scenarios)
    assert (tmp_path / SCENARIOS_MANIFEST).exists()

    recovered = read_scenarios_manifest(tmp_path, Scenario)

    assert len(recovered) == len(scenarios)
    for original, restored in zip(scenarios, recovered):
        assert restored.scenario_id == original.scenario_id
        assert restored.replication_id == original.replication_id
        assert restored.density == original.density
        assert restored.label == original.label


def test_scenarios_manifest_round_trips_rng_bit_exactly(tmp_path):
    """The reconstructed scenario's rng produces the identical stream as the original."""
    scenario = Scenario(rng=42, scenario_id=0)
    write_scenarios_manifest(tmp_path, [scenario])

    [restored] = read_scenarios_manifest(tmp_path, Scenario)

    assert [scenario.rng.random() for _ in range(5)] == [
        restored.rng.random() for _ in range(5)
    ]


def test_scenarios_manifest_casts_numpy_scalars_to_python_native(tmp_path):
    """Numpy scalar parameters are stored as JSON-native types."""
    scenario = Scenario(rng=1, weight=np.float64(1.5), count=np.int64(3))
    write_scenarios_manifest(tmp_path, [scenario])

    raw = json.loads((tmp_path / SCENARIOS_MANIFEST).read_text())
    assert type(raw[0]["weight"]) is float
    assert type(raw[0]["count"]) is int

    [restored] = read_scenarios_manifest(tmp_path, Scenario)
    assert restored.weight == 1.5
    assert restored.count == 3


def test_write_scenarios_manifest_fails_if_already_exists(tmp_path):
    """A second write to the same directory fails loud rather than overwriting."""
    write_scenarios_manifest(tmp_path, [Scenario(rng=1)])

    with pytest.raises(FileExistsError):
        write_scenarios_manifest(tmp_path, [Scenario(rng=2)])


def test_read_scenarios_manifest_rejects_unsupported_generator_class(tmp_path):
    """An unrecognised generator_class mirrors Scenario.__setstate__'s NotImplementedError."""
    write_scenarios_manifest(tmp_path, [Scenario(rng=1)])

    path = tmp_path / SCENARIOS_MANIFEST
    entries = json.loads(path.read_text())
    entries[0]["generator_class"] = "NotARealBitGenerator"
    path.write_text(json.dumps(entries))

    with pytest.raises(NotImplementedError):
        read_scenarios_manifest(tmp_path, Scenario)


# ============================================================
# status.log
# ============================================================


def test_append_and_read_status_basic(tmp_path):
    """Appended statuses, with and without failure detail, read back correctly."""
    run_ok = RunId(0, -1)
    run_bad = RunId(1, -1)
    failure = FailureInfo(
        origin=FailureOrigin.RUNNING,
        exception_type="RuntimeError",
        message="boom",
        traceback="tb",
    )

    append_status(tmp_path, run_ok, Status.SUCCEEDED)
    append_status(tmp_path, run_bad, Status.FAILED, failure)

    result = read_status(tmp_path)

    assert result[run_ok] == (Status.SUCCEEDED, None)
    status, recovered_failure = result[run_bad]
    assert status == Status.FAILED
    assert recovered_failure == failure


def test_read_status_missing_file_returns_empty_dict(tmp_path):
    """No status.log at all is not an error; it just means nothing is recorded yet."""
    assert read_status(tmp_path) == {}


def test_read_status_last_write_wins(tmp_path):
    """Re-appending a status for the same RunId overrides the earlier entry."""
    run_id = RunId(0, -1)
    append_status(tmp_path, run_id, Status.FAILED)
    append_status(tmp_path, run_id, Status.SUCCEEDED)

    result = read_status(tmp_path)
    assert result[run_id] == (Status.SUCCEEDED, None)


def test_read_status_tolerates_torn_final_line(tmp_path):
    """A truncated last line (simulating a kill mid-append) is dropped with a warning."""
    append_status(tmp_path, RunId(0, -1), Status.SUCCEEDED)
    append_status(tmp_path, RunId(1, -1), Status.FAILED)

    path = tmp_path / STATUS_LOG
    with path.open("a") as f:
        f.write('{"scenario_id": 2, "replication_id": -1, "stat')  # torn, no newline

    with pytest.warns(UserWarning, match="torn"):
        result = read_status(tmp_path)

    assert result[RunId(0, -1)] == (Status.SUCCEEDED, None)
    assert result[RunId(1, -1)][0] == Status.FAILED
    assert RunId(2, -1) not in result


def test_read_status_corrupt_line_not_at_tail_raises(tmp_path):
    """A malformed line before the last one is corruption, not a torn tail — it raises."""
    path = tmp_path / STATUS_LOG
    with path.open("w") as f:
        f.write("not valid json\n")
        f.write(
            json.dumps({"scenario_id": 0, "replication_id": -1, "status": "SUCCEEDED"})
            + "\n"
        )

    with pytest.raises(ValueError, match="corrupt"):
        read_status(tmp_path)


def test_read_status_skips_blank_lines(tmp_path):
    """Blank lines interspersed in the log do not break parsing."""
    append_status(tmp_path, RunId(0, -1), Status.SUCCEEDED)
    with (tmp_path / STATUS_LOG).open("a") as f:
        f.write("\n")
    append_status(tmp_path, RunId(1, -1), Status.FAILED)

    result = read_status(tmp_path)
    assert set(result) == {RunId(0, -1), RunId(1, -1)}
