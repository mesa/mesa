"""Tests for DiskStreamWriter, DiskReference, and disk_writer module helpers."""

import pickle

import pandas as pd
import pyarrow as pa
import pyarrow.ipc
import pytest

from mesa.experimental.scenarios import disk_writer
from mesa.experimental.scenarios.disk_writer import (
    DiskReference,
    DiskStreamWriter,
    _validate_schemas,
)
from mesa.experimental.scenarios.store import RunId

# ============================================================
# _validate_name
# ============================================================


@pytest.mark.parametrize("name", ["results", "results_v2", "results-2", "a.b", "A1"])
def test_validate_name_accepts_identifier_like_names(name):
    """Identifier-like output names, with dot/dash allowed after the first char, pass."""
    DiskStreamWriter._validate_name(name)  # does not raise


@pytest.mark.parametrize("name", ["../escape", "a/b", ".hidden", "", 123])
def test_validate_name_rejects_unsafe_names(name):
    """Path separators, a leading dot, an empty name, and non-strings are all rejected."""
    with pytest.raises(ValueError, match="invalid output name"):
        DiskStreamWriter._validate_name(name)


def test_to_reference_rejects_unsafe_output_name(tmp_path):
    """An unsafe output name in the outcome dict is rejected before anything is written."""
    writer = DiskStreamWriter(tmp_path, session="s1")

    with pytest.raises(ValueError, match="invalid output name"):
        writer.to_reference(RunId(0, -1), {"../escape": pd.DataFrame({"x": [1]})})

    assert not (tmp_path / "outputs").exists()


# ============================================================
# _validate_schemas / reserved identity columns
# ============================================================


def test_validate_schemas_rejects_scenario_id_column():
    """A schema declaring scenario_id itself is rejected."""
    schemas = {"results": pa.schema([pa.field("scenario_id", pa.int64())])}
    with pytest.raises(ValueError, match="scenario_id"):
        _validate_schemas(schemas)


def test_validate_schemas_rejects_replication_id_column():
    """A schema declaring replication_id itself is rejected."""
    schemas = {"results": pa.schema([pa.field("replication_id", pa.int64())])}
    with pytest.raises(ValueError, match="replication_id"):
        _validate_schemas(schemas)


def test_validate_schemas_accepts_schema_without_reserved_columns():
    """A schema declaring only the model's own outcome columns passes."""
    schemas = {"results": pa.schema([pa.field("value", pa.float64())])}
    _validate_schemas(schemas)  # does not raise


def test_diskstreamwriter_init_validates_schemas(tmp_path):
    """DiskStreamWriter.__init__ runs the same validation as _validate_schemas directly.

    This is the safety net for tests (and any other caller) that construct a
    writer in isolation, bypassing DiskStore's own fail-fast validation.
    """
    bad_schema = {"results": pa.schema([pa.field("scenario_id", pa.int64())])}
    with pytest.raises(ValueError, match="scenario_id"):
        DiskStreamWriter(tmp_path, session="s1", schemas=bad_schema)


# ============================================================
# _file_for
# ============================================================


def test_file_for_matches_documented_pattern(tmp_path):
    """_file_for produces worker-{session}-{host}-{uuid}.arrow under outputs/{name}."""
    writer = DiskStreamWriter(tmp_path, session="sess1")

    path = writer._file_for("results")

    assert path.parent == tmp_path / "outputs" / "results"
    assert path.name.startswith("worker-sess1-")
    assert path.suffix == ".arrow"


def test_file_for_differs_by_session(tmp_path):
    """Two writers with different sessions produce different filenames for the same output.

    Host and uuid are fixed for the whole test process, so this isolates the
    session axis specifically.
    """
    writer_a = DiskStreamWriter(tmp_path, session="sess-a")
    writer_b = DiskStreamWriter(tmp_path, session="sess-b")

    assert writer_a._file_for("results") != writer_b._file_for("results")


# ============================================================
# The empty-vs-populated-frame schema corner case, and its fix
# ============================================================


def test_unscheduled_output_schema_mismatch_between_empty_and_populated_frame(tmp_path):
    """Documents the corner case explicit schemas exist to avoid.

    Without an explicit schema, whichever run writes first to an output
    fixes its inferred schema. An empty int64 column and a later populated
    string column are chosen specifically because they are guaranteed to
    infer as different Arrow types, regardless of any version-specific
    empty-column inference quirk (e.g. an empty object column inferring as
    Arrow's null type) — the point under test is the mismatch itself, not
    any particular inference behavior.
    """
    writer = DiskStreamWriter(tmp_path, session="s1")
    empty = pd.DataFrame({"value": pd.array([], dtype="int64")})
    populated = pd.DataFrame({"value": ["a", "b"]})

    writer.to_reference(RunId(0, -1), {"results": empty})
    with pytest.raises(ValueError, match="schema"):
        writer.to_reference(RunId(1, -1), {"results": populated})


def test_explicit_schema_avoids_empty_vs_populated_mismatch(tmp_path):
    """With an explicit schema, the same empty-then-populated sequence succeeds.

    Same shape as the corner-case test above (an empty frame written before
    a populated one for the same output) — this is the fix for it.
    """
    schema = {"results": pa.schema([pa.field("value", pa.string())])}
    writer = DiskStreamWriter(tmp_path, session="s1", schemas=schema)
    empty = pd.DataFrame({"value": pd.array([], dtype="object")})
    populated = pd.DataFrame({"value": ["a", "b"]})

    ref0 = writer.to_reference(RunId(0, -1), {"results": empty})
    ref1 = writer.to_reference(RunId(1, -1), {"results": populated})

    assert ref0.run_id == RunId(0, -1)
    assert ref1.run_id == RunId(1, -1)


# ============================================================
# Missing / extra columns against a declared schema
# ============================================================


def test_missing_declared_column_raises(tmp_path):
    """A frame missing a column its declared schema requires is a hard failure."""
    schema = {
        "results": pa.schema(
            [pa.field("value", pa.float64()), pa.field("extra", pa.int64())]
        )
    }
    writer = DiskStreamWriter(tmp_path, session="s1", schemas=schema)
    frame = pd.DataFrame({"value": [1.0, 2.0]})  # missing "extra"

    with pytest.raises(ValueError, match="missing column"):
        writer.to_reference(RunId(0, -1), {"results": frame})


def test_extra_undeclared_column_warns_and_is_dropped(tmp_path):
    """An extra column not in the declared schema is warned about once and dropped.

    Reads the file back directly (rather than trusting no exception was
    raised) to confirm the column is actually absent from what's on disk,
    not merely that persisting didn't error.
    """
    schema = {"results": pa.schema([pa.field("value", pa.float64())])}
    writer = DiskStreamWriter(tmp_path, session="s1", schemas=schema)
    frame = pd.DataFrame({"value": [1.0, 2.0], "debug_col": ["x", "y"]})

    with pytest.warns(UserWarning, match="debug_col"):
        ref = writer.to_reference(RunId(0, -1), {"results": frame})

    path = writer._file_for("results")
    with pa.OSFile(str(path), "rb") as source:
        table = pa.ipc.open_stream(source).read_all()
    assert "debug_col" not in table.column_names
    assert "value" in table.column_names
    assert ref.run_id == RunId(0, -1)


def test_to_reference_is_atomic_across_outputs_on_validation_failure(tmp_path):
    """A failure converting one output prevents writing any output for that run.

    to_reference validates+converts every named output before writing any of
    them, so a run either contributes a full batch to every stream or none
    at all — confirmed here by a perfectly valid sibling output ("good")
    never reaching disk when "bad" fails.
    """
    schema = {"bad": pa.schema([pa.field("required", pa.float64())])}
    writer = DiskStreamWriter(tmp_path, session="s1", schemas=schema)
    outcome = {
        "good": pd.DataFrame({"x": [1]}),
        "bad": pd.DataFrame({"x": [1]}),  # missing the declared "required" column
    }

    with pytest.raises(ValueError, match="missing column"):
        writer.to_reference(RunId(0, -1), outcome)

    assert not (tmp_path / "outputs" / "good").exists()


# ============================================================
# Session rotation
# ============================================================


def test_session_rotation_closes_prior_session_stream_and_opens_new_file(tmp_path):
    """A writer constructed with a new session closes the prior session's open stream.

    Simulates a worker process reused across two run_scenarios() invocations
    (or a resume): the module-level registry is per-process, so a session
    change must be detected and acted on, not just carried in the writer's
    own state.
    """
    frame = pd.DataFrame({"value": [1.0]})

    writer_a = DiskStreamWriter(tmp_path, session="sess-a")
    writer_a.to_reference(RunId(0, -1), {"results": frame})
    path_a = writer_a._file_for("results")
    assert path_a.exists()

    writer_b = DiskStreamWriter(tmp_path, session="sess-b")
    writer_b.to_reference(RunId(0, -1), {"results": frame})
    path_b = writer_b._file_for("results")

    assert path_b != path_a
    assert path_b.exists()
    # session A's registry entry was evicted (closed), not left dangling
    assert ("sess-a", "results") not in disk_writer._STREAMS
    assert ("sess-b", "results") in disk_writer._STREAMS

    with pa.OSFile(str(path_a), "rb") as source:
        table = pa.ipc.open_stream(source).read_all()
    assert table.num_rows == 1


# ============================================================
# Picklability (needed for parallel execution)
# ============================================================


def test_disk_stream_writer_is_picklable(tmp_path):
    """The writer handed to workers round-trips through pickle, schemas included."""
    schema = pa.schema([pa.field("value", pa.int64())])
    writer = DiskStreamWriter(tmp_path, session="s1", schemas={"results": schema})

    restored = pickle.loads(pickle.dumps(writer))  # noqa: S301

    assert restored.store_dir == writer.store_dir
    assert restored.session == writer.session
    assert restored.schemas["results"].equals(schema)


def test_disk_reference_is_picklable():
    """References cross back from worker to root; they must pickle."""
    ref = DiskReference(RunId(1, 0))
    restored = pickle.loads(pickle.dumps(ref))  # noqa: S301
    assert restored.run_id == RunId(1, 0)
    assert restored.payload is None
