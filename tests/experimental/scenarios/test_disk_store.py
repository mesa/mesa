"""Tests for DiskStore: manifests, status tracking, and outcome read-back."""

import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc
import pytest
from conftest import _VariableRowsModel

from mesa.experimental.scenarios import (
    DiskStore,
    RunConfiguration,
    Scenario,
    ScenarioAbortedException,
    ScenarioFailedException,
    ScenarioNotFoundException,
    ScenarioNotReadyException,
    run_scenarios,
    store_metadata,
)
from mesa.experimental.scenarios.disk_writer import DiskReference
from mesa.experimental.scenarios.exceptions import FailureInfo, FailureOrigin
from mesa.experimental.scenarios.store import RunId, Status


def _write_worker_file(directory: Path, filename: str, table: pa.Table) -> None:
    """Write a standalone Arrow IPC stream file, simulating one worker's output.

    DiskStreamWriter can't produce two distinct files for the same session
    within a single test process (host/uuid are fixed once per process), so
    a second "worker" is constructed directly at the file level instead.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    with (
        pa.OSFile(str(path), "wb") as sink,
        pa.ipc.new_stream(sink, table.schema) as ipc_writer,
    ):
        ipc_writer.write_table(table)


# ============================================================
# Construction / validation
# ============================================================


def test_creates_store_directory_and_outputs_subdir(tmp_path):
    """Constructing a DiskStore creates store_dir and its outputs/ subdirectory."""
    store_dir = tmp_path / "nested" / "store"
    DiskStore(store_dir)

    assert store_dir.is_dir()
    assert (store_dir / "outputs").is_dir()


def test_rejects_invalid_on_schema_conflict(tmp_path):
    """on_schema_conflict must be 'warn' or 'raise'."""
    with pytest.raises(ValueError, match="on_schema_conflict"):
        DiskStore(tmp_path, on_schema_conflict="ignore")


def test_rejects_schema_declaring_reserved_identity_column(tmp_path):
    """Schemas validation is wired through to disk_writer._validate_schemas."""
    bad_schema = {"results": pa.schema([pa.field("scenario_id", pa.int64())])}
    with pytest.raises(ValueError, match="scenario_id"):
        DiskStore(tmp_path, schemas=bad_schema)


# ============================================================
# write_scenarios
# ============================================================


def test_write_scenarios_writes_manifests_and_records_scenarios(
    disk_store, scenario_list, basic_config
):
    """write_scenarios writes store.json and scenarios.json, and records the scenarios."""
    disk_store.write_scenarios(scenario_list, basic_config)

    assert (disk_store.store_dir / "store.json").exists()
    assert (disk_store.store_dir / "scenarios.json").exists()
    recovered = disk_store.read_scenarios()
    assert len(recovered) == len(scenario_list)
    assert {s.scenario_id for s in recovered} == {s.scenario_id for s in scenario_list}


def test_write_scenarios_fails_if_store_json_already_exists(
    disk_store, scenario_list, basic_config
):
    """A second write_scenarios call to the same store directory fails loud."""
    disk_store.write_scenarios(scenario_list, basic_config)

    with pytest.raises(FileExistsError):
        disk_store.write_scenarios(scenario_list, basic_config)


def test_write_scenarios_records_provenance(disk_store, scenario_list, basic_config):
    """Provenance in store.json is populated via config.model_class."""
    disk_store.write_scenarios(scenario_list, basic_config)

    manifest = store_metadata.read_store_manifest(disk_store.store_dir)
    assert "mesa_version" in manifest["provenance"]
    assert "created_at" in manifest["provenance"]


# ============================================================
# Status marking
# ============================================================


def test_initial_status_is_pending(populated_disk_store):
    """Every recorded scenario starts PENDING."""
    store, scenarios = populated_disk_store
    for scenario in scenarios:
        run_id = RunId(scenario.scenario_id, scenario.replication_id)
        assert store.check_status(run_id) == Status.PENDING


def test_mark_succeeded_writes_through_to_status_log(populated_disk_store):
    """mark_succeeded updates both the in-process record and status.log."""
    store, scenarios = populated_disk_store
    run_id = RunId(scenarios[0].scenario_id, scenarios[0].replication_id)
    ref = store.writer().to_reference(run_id, {"results": pd.DataFrame({"x": [1]})})

    store.mark_succeeded(ref)

    assert store.check_status(run_id) == Status.SUCCEEDED
    logged = store_metadata.read_status(store.store_dir)
    assert logged[run_id] == (Status.SUCCEEDED, None)


def test_mark_failed_writes_through_to_status_log(populated_disk_store):
    """mark_failed records the failure on both the record and status.log."""
    store, scenarios = populated_disk_store
    run_id = RunId(scenarios[0].scenario_id, scenarios[0].replication_id)
    failure = FailureInfo(FailureOrigin.RUNNING, "RuntimeError", "boom", "tb")

    store.mark_failed(run_id, failure)

    assert store.check_status(run_id) == Status.FAILED
    logged_status, logged_failure = store_metadata.read_status(store.store_dir)[run_id]
    assert logged_status == Status.FAILED
    assert logged_failure == failure


def test_mark_aborted_writes_through_to_status_log(populated_disk_store):
    """mark_aborted records the failure on both the record and status.log."""
    store, scenarios = populated_disk_store
    run_id = RunId(scenarios[0].scenario_id, scenarios[0].replication_id)
    failure = FailureInfo(FailureOrigin.ABORTED, "BrokenProcessPool", "pool died", "tb")

    store.mark_aborted(run_id, failure)

    assert store.check_status(run_id) == Status.ABORTED
    logged_status, logged_failure = store_metadata.read_status(store.store_dir)[run_id]
    assert logged_status == Status.ABORTED
    assert logged_failure == failure


def test_mark_succeeded_unknown_run_id_raises(disk_store):
    """Marking succeeded for a run never registered via write_scenarios raises."""
    with pytest.raises(ScenarioNotFoundException):
        disk_store.mark_succeeded(DiskReference(RunId(999, -1)))


def test_check_status_unknown_run_id_raises(disk_store):
    """check_status on an unregistered run raises."""
    with pytest.raises(ScenarioNotFoundException):
        disk_store.check_status(RunId(999, -1))


# ============================================================
# status() dataframe / filter methods
# ============================================================


def test_status_dataframe_reflects_current_state(populated_disk_store):
    """status() returns one row per run, with the current status."""
    store, scenarios = populated_disk_store
    s0, s1, s2 = scenarios
    store.mark_succeeded(
        store.writer().to_reference(RunId(s0.scenario_id, s0.replication_id), {})
    )
    store.mark_failed(
        RunId(s1.scenario_id, s1.replication_id),
        FailureInfo(FailureOrigin.RUNNING, "E", "m", ""),
    )

    df = store.status()

    assert df.loc[(s0.scenario_id, s0.replication_id), "status"] == "SUCCEEDED"
    assert df.loc[(s1.scenario_id, s1.replication_id), "status"] == "FAILED"
    assert df.loc[(s2.scenario_id, s2.replication_id), "status"] == "PENDING"


def test_filter_methods_partition_by_status(populated_disk_store):
    """succeeded/failed/pending/aborted correctly partition the run set."""
    store, scenarios = populated_disk_store
    s0, s1, s2 = scenarios
    run_id_0 = RunId(s0.scenario_id, s0.replication_id)
    run_id_1 = RunId(s1.scenario_id, s1.replication_id)
    run_id_2 = RunId(s2.scenario_id, s2.replication_id)

    store.mark_succeeded(store.writer().to_reference(run_id_0, {}))
    store.mark_failed(run_id_1, FailureInfo(FailureOrigin.RUNNING, "E", "m", ""))

    assert set(store.succeeded()) == {run_id_0}
    assert set(store.failed()) == {run_id_1}
    assert set(store.pending()) == {run_id_2}
    assert set(store.aborted()) == set()


# ============================================================
# retrieve_output: status gating and read-back correctness
# ============================================================


def test_retrieve_output_pending_raises(populated_disk_store):
    """A pending run's output is not ready yet."""
    store, scenarios = populated_disk_store
    run_id = RunId(scenarios[0].scenario_id, scenarios[0].replication_id)
    with pytest.raises(ScenarioNotReadyException) as exc_info:
        store.retrieve_output(run_id)
    assert exc_info.value.run_id == run_id


def test_retrieve_output_failed_raises_with_failure_detail(populated_disk_store):
    """A failed run's output request raises with the recorded failure attached."""
    store, scenarios = populated_disk_store
    run_id = RunId(scenarios[0].scenario_id, scenarios[0].replication_id)
    failure = FailureInfo(FailureOrigin.RUNNING, "RuntimeError", "boom", "tb")
    store.mark_failed(run_id, failure)

    with pytest.raises(ScenarioFailedException) as exc_info:
        store.retrieve_output(run_id)
    assert exc_info.value.run_id == run_id
    assert exc_info.value.failure is failure


def test_retrieve_output_aborted_raises_with_failure_detail(populated_disk_store):
    """An aborted run's output request raises with the recorded failure attached."""
    store, scenarios = populated_disk_store
    run_id = RunId(scenarios[0].scenario_id, scenarios[0].replication_id)
    failure = FailureInfo(FailureOrigin.ABORTED, "BrokenProcessPool", "pool died", "tb")
    store.mark_aborted(run_id, failure)

    with pytest.raises(ScenarioAbortedException) as exc_info:
        store.retrieve_output(run_id)
    assert exc_info.value.run_id == run_id
    assert exc_info.value.failure is failure


def test_retrieve_output_unknown_run_id_raises(disk_store):
    """Requesting output for a run never registered raises."""
    with pytest.raises(ScenarioNotFoundException):
        disk_store.retrieve_output(RunId(999, -1))


def test_retrieve_output_returns_persisted_data(populated_disk_store):
    """A succeeded run's output reads back the exact data that was written."""
    store, scenarios = populated_disk_store
    run_id = RunId(scenarios[0].scenario_id, scenarios[0].replication_id)
    writer = store.writer()
    outcome = {"results": pd.DataFrame({"value": [1.0, 2.0, 3.0]})}

    ref = writer.to_reference(run_id, outcome)
    store.mark_succeeded(ref)

    output = store.retrieve_output(run_id)
    assert list(output["results"]["value"]) == [1.0, 2.0, 3.0]


def test_retrieve_output_filters_to_the_requested_run(populated_disk_store):
    """retrieve_output returns only rows belonging to the requested run_id.

    Both runs share one worker file (same session, same process), so this
    exercises the scenario_id/replication_id filter, not just single-run
    read-back.
    """
    store, scenarios = populated_disk_store
    s0, s1, _ = scenarios
    run_id_0 = RunId(s0.scenario_id, s0.replication_id)
    run_id_1 = RunId(s1.scenario_id, s1.replication_id)
    writer = store.writer()

    store.mark_succeeded(
        writer.to_reference(run_id_0, {"results": pd.DataFrame({"value": [1.0]})})
    )
    store.mark_succeeded(
        writer.to_reference(run_id_1, {"results": pd.DataFrame({"value": [2.0]})})
    )

    assert list(store.retrieve_output(run_id_0)["results"]["value"]) == [1.0]
    assert list(store.retrieve_output(run_id_1)["results"]["value"]) == [2.0]


def test_retrieve_output_zero_row_output_reads_back_empty(populated_disk_store):
    """A run that wrote a valid zero-row batch reads back as an empty (not missing) frame."""
    store, scenarios = populated_disk_store
    run_id = RunId(scenarios[0].scenario_id, scenarios[0].replication_id)
    writer = store.writer()
    empty = pd.DataFrame({"value": pd.array([], dtype="int64")})

    store.mark_succeeded(writer.to_reference(run_id, {"results": empty}))

    output = store.retrieve_output(run_id)
    assert "results" in output
    assert len(output["results"]) == 0
    assert "value" in output["results"].columns


# ============================================================
# from_directory
# ============================================================


def test_from_directory_round_trips_scenarios_and_status(
    tmp_path, scenario_list, basic_config
):
    """from_directory reconstructs the run set from the manifests and status log."""
    store = DiskStore(tmp_path)
    store.write_scenarios(scenario_list, basic_config)
    writer = store.writer()
    s0, s1, _ = scenario_list
    run_id_0 = RunId(s0.scenario_id, s0.replication_id)
    run_id_1 = RunId(s1.scenario_id, s1.replication_id)
    store.mark_succeeded(writer.to_reference(run_id_0, {}))
    store.mark_failed(run_id_1, FailureInfo(FailureOrigin.RUNNING, "E", "m", ""))

    reloaded = DiskStore.from_directory(tmp_path, Scenario)

    assert reloaded.check_status(run_id_0) == Status.SUCCEEDED
    assert reloaded.check_status(run_id_1) == Status.FAILED
    assert len(reloaded.read_scenarios()) == len(scenario_list)


def test_from_directory_writer_raises(tmp_path, scenario_list, basic_config):
    """writer() is unavailable on a store loaded via from_directory (read-only, no session)."""
    store = DiskStore(tmp_path)
    store.write_scenarios(scenario_list, basic_config)

    reloaded = DiskStore.from_directory(tmp_path, Scenario)

    assert reloaded.session is None
    with pytest.raises(RuntimeError, match="from_directory"):
        reloaded.writer()


def test_from_directory_rejects_invalid_on_schema_conflict(
    tmp_path, scenario_list, basic_config
):
    """on_schema_conflict is validated on from_directory too, not just __init__."""
    store = DiskStore(tmp_path)
    store.write_scenarios(scenario_list, basic_config)

    with pytest.raises(ValueError, match="on_schema_conflict"):
        DiskStore.from_directory(tmp_path, Scenario, on_schema_conflict="ignore")


def test_from_directory_rejects_mismatched_store_format_version(
    tmp_path, scenario_list, basic_config
):
    """A store.json format version different from the running code's raises."""
    store = DiskStore(tmp_path)
    store.write_scenarios(scenario_list, basic_config)

    manifest_path = tmp_path / "store.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["store_format_version"] = 999
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="format version"):
        DiskStore.from_directory(tmp_path, Scenario)


# ============================================================
# Truncated file tolerance
# ============================================================


def test_read_stream_file_tolerates_truncated_tail(populated_disk_store):
    """A worker file truncated mid-write (simulating a hard kill) doesn't raise.

    Whether the truncation lands inside the batch body or the schema
    preamble, _read_stream_file must recover gracefully (None or a partial
    table) rather than propagate the underlying ArrowInvalid.
    """
    store, scenarios = populated_disk_store
    run_id = RunId(scenarios[0].scenario_id, scenarios[0].replication_id)
    writer = store.writer()
    store.mark_succeeded(
        writer.to_reference(run_id, {"results": pd.DataFrame({"value": [1.0, 2.0]})})
    )

    path = writer._file_for("results")
    original = path.read_bytes()
    assert len(original) > 16, "test assumption: file has enough bytes to truncate"
    path.write_bytes(original[:-16])

    table = DiskStore._read_stream_file(path)  # must not raise
    assert table is None or table.num_rows <= 2


def test_read_stream_file_empty_file_returns_none(tmp_path):
    """An empty file (no schema, no batches) reads back as None, not an error."""
    path = tmp_path / "empty.arrow"
    path.write_bytes(b"")

    assert DiskStore._read_stream_file(path) is None


# ============================================================
# Cross-worker schema conflict (_unify)
# ============================================================


def test_unify_warns_and_null_fills_on_cross_worker_schema_conflict(tmp_path):
    """Two worker files with different schemas for the same output unify under 'warn'."""
    store = DiskStore(tmp_path, on_schema_conflict="warn")
    output_dir = store.store_dir / "outputs" / "results"

    table_a = pa.table({"scenario_id": [0], "replication_id": [-1], "value": [1.0]})
    table_b = pa.table(
        {"scenario_id": [1], "replication_id": [-1], "value": [2.0], "extra": [3]}
    )
    _write_worker_file(output_dir, "worker-a.arrow", table_a)
    _write_worker_file(output_dir, "worker-b.arrow", table_b)

    with pytest.warns(UserWarning, match="unifying"):
        table = store._read_one_output("results")

    assert table.num_rows == 2
    assert "extra" in table.column_names
    mask = pc.equal(table["scenario_id"], 0)
    assert table.filter(mask)["extra"][0].as_py() is None


def test_unify_raises_on_cross_worker_schema_conflict_when_configured(tmp_path):
    """The same conflict raises instead of warning under on_schema_conflict='raise'."""
    store = DiskStore(tmp_path, on_schema_conflict="raise")
    output_dir = store.store_dir / "outputs" / "results"

    table_a = pa.table({"scenario_id": [0], "replication_id": [-1], "value": [1.0]})
    table_b = pa.table(
        {"scenario_id": [1], "replication_id": [-1], "value": [2.0], "extra": [3]}
    )
    _write_worker_file(output_dir, "worker-a.arrow", table_a)
    _write_worker_file(output_dir, "worker-b.arrow", table_b)

    with pytest.raises(ValueError, match="schema conflict"):
        store._read_one_output("results")


def test_unify_no_conflict_when_worker_schemas_match(tmp_path):
    """Two worker files with identical schemas concatenate without any warning."""
    store = DiskStore(tmp_path)
    output_dir = store.store_dir / "outputs" / "results"

    table_a = pa.table({"scenario_id": [0], "replication_id": [-1], "value": [1.0]})
    table_b = pa.table({"scenario_id": [1], "replication_id": [-1], "value": [2.0]})
    _write_worker_file(output_dir, "worker-a.arrow", table_a)
    _write_worker_file(output_dir, "worker-b.arrow", table_b)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        table = store._read_one_output("results")

    assert table.num_rows == 2


# ============================================================
# Real parallel execution: the schema corner case, fixed and unfixed
# ============================================================


def test_run_scenarios_disk_store_schema_survives_real_parallel_execution(tmp_path):
    """End-to-end: DiskStore + explicit schema + a real multi-worker pool.

    Two scenarios produce a zero-row and a five-row 'results' frame
    respectively — exactly the empty-vs-populated corner case the schemas
    parameter exists to avoid — and must succeed regardless of which worker
    (and therefore which write order) each scenario lands on.
    """
    scenarios = [Scenario(n_rows=0), Scenario(n_rows=5)]
    store = DiskStore(
        tmp_path, schemas={"results": pa.schema([pa.field("value", pa.int64())])}
    )

    with ProcessPoolExecutor(max_workers=2) as ex:
        run_scenarios(
            scenarios,
            RunConfiguration(_VariableRowsModel, until=1),
            store=store,
            executor=ex,
            progress=False,
        )

    assert len(store.succeeded()) == 2
    assert len(store.failed()) == 0

    empty_output = store.retrieve_output(RunId(scenarios[0].scenario_id, -1))
    full_output = store.retrieve_output(RunId(scenarios[1].scenario_id, -1))
    assert len(empty_output["results"]) == 0
    assert list(full_output["results"]["value"]) == [0, 1, 2, 3, 4]


def test_run_scenarios_disk_store_without_schema_hits_corner_case(tmp_path):
    """Without an explicit schema, the same corner case is a real, reproducible failure.

    max_workers=1 forces both scenarios onto the same worker process, in
    submission order (see test_run_scenarios_partial_abort in test_runner.py
    for the same guarantee used elsewhere), so the zero-row scenario's batch
    deterministically fixes a schema the five-row scenario's batch cannot
    satisfy. This documents the bug the schemas parameter exists to fix — if
    this test starts failing (i.e. both runs start succeeding), the fix may
    have started silently applying without an explicit schema, and this
    test's assumptions need revisiting.
    """
    scenarios = [Scenario(n_rows=0), Scenario(n_rows=5)]
    store = DiskStore(tmp_path)

    with ProcessPoolExecutor(max_workers=1) as ex:
        run_scenarios(
            scenarios,
            RunConfiguration(_VariableRowsModel, until=1),
            store=store,
            executor=ex,
            progress=False,
        )

    assert len(store.succeeded()) == 1
    assert len(store.failed()) == 1
    [failure_record] = store.failed().values()
    assert failure_record.failure.origin == FailureOrigin.WRITING
