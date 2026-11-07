"""Tests for InMemoryStore and the store-side exceptions it raises."""

import pickle

import numpy as np
import pandas as pd
import pytest

from mesa.experimental.scenarios import (
    ScenarioAbortedException,
    ScenarioFailedException,
    ScenarioNotFoundException,
    ScenarioNotReadyException,
)
from mesa.experimental.scenarios.exceptions import FailureInfo, FailureOrigin
from mesa.experimental.scenarios.store import (
    InMemoryReference,
    InMemoryStore,
    InMemoryWriter,
    RunId,
    Status,
)


def test_store_write_and_read_scenarios(scenario_list, basic_config):
    """Store and read scenarios."""
    store = InMemoryStore()
    store.write_scenarios(scenario_list, basic_config)
    recovered = store.read_scenarios()
    assert len(recovered) == len(scenario_list)
    assert {s.scenario_id for s in recovered} == {s.scenario_id for s in scenario_list}


def test_store_initial_status_is_pending(populated_store):
    """Test the initial status of the store."""
    store, scenarios = populated_store
    for scenario in scenarios:
        run_id = RunId(scenario.scenario_id, scenario.replication_id)
        assert store.check_status(run_id) == Status.PENDING


def test_store_mark_succeeded(populated_store):
    """Test the marked status of the store for successes."""
    store, scenarios = populated_store
    scenario = scenarios[0]
    run_id = RunId(scenario.scenario_id, scenario.replication_id)
    writer = store.writer()
    outcome = {"results": pd.DataFrame({"x": [1]})}
    ref = writer.to_reference(run_id, outcome)
    store.mark_succeeded(ref)

    assert store.check_status(run_id) == Status.SUCCEEDED
    output = store.retrieve_output(run_id)
    assert "results" in output


def test_store_retrieve_output_pending_raises(populated_store):
    """Test the retrieve output of the store while status is pending."""
    store, scenarios = populated_store
    run_id = RunId(scenarios[0].scenario_id, scenarios[0].replication_id)
    with pytest.raises(ScenarioNotReadyException) as exc_info:
        store.retrieve_output(run_id)
    assert exc_info.value.run_id == run_id


def test_store_mark_failed(populated_store):
    """Test the marked status of the store for failures."""
    store, scenarios = populated_store
    scenario = scenarios[0]
    run_id = RunId(scenario.scenario_id, scenario.replication_id)
    failure = FailureInfo(
        origin=FailureOrigin.RUNNING,
        exception_type="RuntimeError",
        message="boom",
        traceback="...",
    )
    store.mark_failed(run_id, failure)

    assert store.check_status(run_id) == Status.FAILED
    with pytest.raises(ScenarioFailedException) as exc_info:
        store.retrieve_output(run_id)
    assert exc_info.value.run_id == run_id
    assert exc_info.value.failure is failure


def test_store_mark_aborted(populated_store):
    """Test storing marked aborted scenarios."""
    store, scenarios = populated_store
    run_id = RunId(scenarios[0].scenario_id, scenarios[0].replication_id)
    failure = FailureInfo(FailureOrigin.ABORTED, "BrokenProcessPool", "pool died", "tb")
    store.mark_aborted(run_id, failure)

    assert store.check_status(run_id) == Status.ABORTED
    with pytest.raises(ScenarioAbortedException) as exc_info:
        store.retrieve_output(run_id)
    assert exc_info.value.run_id == run_id
    assert exc_info.value.failure is failure


def test_store_unknown_run_id_raises():
    """Test the unknown run_id raises exception."""
    store = InMemoryStore()
    with pytest.raises(ScenarioNotFoundException) as exc_info:
        store.check_status(RunId(999, -1))
    assert exc_info.value.run_id == RunId(999, -1)


def test_store_status_dataframe(populated_store):
    """Test the status dataframe store."""
    store, scenarios = populated_store
    df = store.status()
    assert list(df.columns) == ["status"]
    assert len(df) == len(scenarios)
    assert (df["status"] == "PENDING").all()


def test_store_status_dataframe_mixed_case(populated_store):
    """Test the status dataframe store with a mix of statuses."""
    store, scenarios = populated_store
    writer = store.writer()

    s0, s1, s2 = scenarios
    run_id_0 = RunId(s0.scenario_id, s0.replication_id)
    run_id_1 = RunId(s1.scenario_id, s1.replication_id)

    store.mark_succeeded(writer.to_reference(run_id_0, {}))
    store.mark_failed(run_id_1, FailureInfo(FailureOrigin.RUNNING, "E", "m", ""))

    df = store.status()
    # pandas converts None replication_id to NaN in the MultiIndex, so look up by scenario_id
    assert df.index.get_level_values("replication_id").dtype == np.int64
    assert df.loc[(s0.scenario_id, s0.replication_id), "status"] == "SUCCEEDED"
    assert df.loc[(s1.scenario_id, s1.replication_id), "status"] == "FAILED"
    assert df.loc[(s2.scenario_id, s2.replication_id), "status"] == "PENDING"


def test_store_filter_methods(populated_store):
    """Test the extraction methods on InMemoryStore for succeeded, failed, and pending."""
    store, scenarios = populated_store
    writer = store.writer()

    s0, s1, s2 = scenarios
    run_id_0 = RunId(s0.scenario_id, s0.replication_id)
    run_id_1 = RunId(s1.scenario_id, s1.replication_id)
    run_id_2 = RunId(s2.scenario_id, s2.replication_id)

    store.mark_succeeded(writer.to_reference(run_id_0, {}))
    store.mark_failed(run_id_1, FailureInfo(FailureOrigin.RUNNING, "E", "m", ""))

    assert set(store.succeeded()) == {run_id_0}
    assert set(store.failed()) == {run_id_1}
    assert set(store.pending()) == {run_id_2}
    assert set(store.aborted()) == set()


def test_store_aborted_filter(populated_store):
    """Test the aborted() filter on InMemoryStore."""
    store, scenarios = populated_store
    s0, _, s2 = scenarios
    run_id_0 = RunId(s0.scenario_id, s0.replication_id)
    run_id_2 = RunId(s2.scenario_id, s2.replication_id)

    store.mark_aborted(
        run_id_0,
        FailureInfo(FailureOrigin.ABORTED, "BrokenProcessPool", "pool died", "tb"),
    )

    assert set(store.aborted()) == {run_id_0}
    assert run_id_0 not in store.pending()
    assert run_id_2 in store.pending()


# ============================================================
# Exception constructors
# ============================================================


@pytest.mark.parametrize(
    "exc_class, kwargs",
    [
        (ScenarioNotFoundException, {}),
        (ScenarioNotFoundException, {"run_id": RunId(1, -1)}),
        (ScenarioNotReadyException, {}),
        (ScenarioNotReadyException, {"run_id": RunId(2, 0)}),
        (ScenarioFailedException, {}),
        (ScenarioFailedException, {"run_id": RunId(3, 1)}),
        (
            ScenarioFailedException,
            {
                "run_id": RunId(4, 2),
                "failure": FailureInfo(
                    FailureOrigin.RUNNING, "RuntimeError", "boom", "tb"
                ),
            },
        ),
        (ScenarioAbortedException, {}),
        (ScenarioAbortedException, {"run_id": RunId(5, 0)}),
        (
            ScenarioAbortedException,
            {
                "run_id": RunId(6, 1),
                "failure": FailureInfo(
                    FailureOrigin.ABORTED, "BrokenProcessPool", "pool died", "tb"
                ),
            },
        ),
    ],
)
def test_exception_constructors(exc_class, kwargs):
    """Test exception constructors."""
    exc = exc_class(**kwargs)
    assert str(exc)
    assert exc.run_id == kwargs.get("run_id")


def test_scenario_failed_exception_message_includes_failure_detail():
    """Test scenario failed exception message including failure detail."""
    failure = FailureInfo(
        origin=FailureOrigin.EXTRACTING,
        exception_type="KeyError",
        message="missing key",
        traceback="...",
    )
    exc = ScenarioFailedException(run_id=RunId(5, 0), failure=failure)
    assert "extracting" in str(exc)
    assert "KeyError" in str(exc)
    assert "missing key" in str(exc)
    assert exc.failure is failure


def test_scenario_aborted_exception_message_includes_failure_detail():
    """Test scenario aborted exception message including failure detail."""
    failure = FailureInfo(
        origin=FailureOrigin.ABORTED,
        exception_type="BrokenProcessPool",
        message="worker died",
        traceback="...",
    )
    exc = ScenarioAbortedException(run_id=RunId(5, 0), failure=failure)
    assert "aborted" in str(exc)
    assert "BrokenProcessPool" in str(exc)
    assert "worker died" in str(exc)
    assert exc.failure is failure


# ============================================================
# Picklability (needed for parallel execution)
# ============================================================


def test_writer_is_picklable():
    """The writer handed to workers round-trips through pickle."""
    writer = InMemoryStore().writer()
    restored = pickle.loads(pickle.dumps(writer))  # noqa: S301
    assert isinstance(restored, InMemoryWriter)


def test_inmemory_reference_is_picklable():
    """References cross back from worker to root; they must pickle."""
    ref = InMemoryReference(RunId(1, 0), {"results": pd.DataFrame({"x": [1]})})
    restored = pickle.loads(pickle.dumps(ref))  # noqa: S301
    assert restored.run_id == RunId(1, 0)
    assert "results" in restored.payload
