"""Tests for RunConfiguration, _safe_call, and run_scenarios."""

import pickle
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import pytest
from conftest import (
    _ConditionalConfig,
    _ConditionalKillModel,
    _DummyModel,
    _ExtractionFailModel,
    _FailingWriter,
    _InstantiationFailModel,
    _RunFailModel,
    _WorkerKillModel,
)

from mesa import Model
from mesa.experimental.data_collection import DataRecorder
from mesa.experimental.scenarios import RunConfiguration, Scenario, run_scenarios
from mesa.experimental.scenarios.exceptions import FailureInfo, FailureOrigin
from mesa.experimental.scenarios.runner import _safe_call
from mesa.experimental.scenarios.store import InMemoryStore, InMemoryWriter, RunId


def test_run_configuration(mocker):
    """Tests for RunConfiguration."""
    dummy_recorder = mocker.Mock(spec=DataRecorder)

    class DummyModel(Model):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.data_recorder = dummy_recorder()

            # setting up the mock
            self.data_recorder.get_table_dataframe.return_value = pd.DataFrame(
                columns=["a", "b"]
            )
            self.data_recorder.get_all_dataframes.return_value = {
                "a": pd.DataFrame(columns=["a", "b"]),
                "b": pd.DataFrame(columns=["a", "b"]),
            }

    until = 10
    configuration = RunConfiguration(DummyModel, until=until)
    assert configuration.model_class is DummyModel

    with pytest.raises(TypeError):
        RunConfiguration(DummyModel(), until=until)
    with pytest.raises(TypeError):
        RunConfiguration(object, until=until)
    with pytest.raises(TypeError):
        RunConfiguration(DummyModel, until="some string")
    with pytest.raises(ValueError):
        RunConfiguration(DummyModel, until=-until)

    configuration = RunConfiguration(DummyModel, until=until, outcomes="a")
    assert configuration.outcomes == ["a"]

    configuration = RunConfiguration(DummyModel, until=until)
    scenario = Scenario()
    model = configuration.instantiate_model(scenario)
    assert model.scenario == scenario

    configuration.run_model(model)
    assert model.time == until

    output = configuration.extract_output(model)
    assert "a" in output
    assert "b" in output

    configuration = RunConfiguration(DummyModel, until=until, outcomes="a")
    output = configuration(scenario)
    assert "a" in output
    assert "b" not in output


# ============================================================
# _safe_call
# ============================================================


def test_safe_call_success(basic_config):
    """Test the success branch of safe call."""
    scenario = Scenario(x=1)
    ref, failure = _safe_call(basic_config, scenario, InMemoryWriter())

    assert failure is None
    assert ref is not None
    assert ref.run_id == RunId(scenario.scenario_id, scenario.replication_id)
    assert "results" in ref.payload


def test_safe_call_instantiation_failure():
    """Test the instantiation failure branch of safe call."""
    config = RunConfiguration(_InstantiationFailModel, until=5)
    ref, failure = _safe_call(config, Scenario(), InMemoryWriter())

    assert ref is None
    assert failure.origin == FailureOrigin.INSTANTIATING
    assert failure.exception_type == "RuntimeError"
    assert "cannot instantiate" in failure.message
    assert failure.traceback


def test_safe_call_run_failure():
    """Test the run failure branch of safe call."""
    config = RunConfiguration(_RunFailModel, until=5)
    ref, failure = _safe_call(config, Scenario(), InMemoryWriter())

    assert ref is None
    assert failure.origin == FailureOrigin.RUNNING
    assert failure.exception_type == "RuntimeError"
    assert "run failed" in failure.message


def test_safe_call_extraction_failure():
    """Test the extraction failure branch of safe call."""
    config = RunConfiguration(_ExtractionFailModel, until=5)
    ref, failure = _safe_call(config, Scenario(), InMemoryWriter())

    assert ref is None
    assert failure.origin == FailureOrigin.EXTRACTING
    assert failure.exception_type == "RuntimeError"
    assert "extraction failed" in failure.message


def test_safe_call_writer_failure(basic_config):
    """Test the failure branch of safe call."""
    ref, failure = _safe_call(basic_config, Scenario(), _FailingWriter())

    assert ref is None
    assert failure.origin == FailureOrigin.WRITING
    assert failure.exception_type == "OSError"
    assert "disk full" in failure.message


# ============================================================
# run_scenarios integration
# ============================================================


def test_run_scenarios_all_succeed(maybe_executor):
    """Test the successful branch of run_scenarios."""
    scenarios = [Scenario(x=i) for i in range(4)]
    store = run_scenarios(
        scenarios,
        RunConfiguration(_DummyModel, until=3),
        progress=False,
        executor=maybe_executor,
    )

    assert len(store.succeeded()) == 4
    assert len(store.failed()) == 0
    assert len(store.pending()) == 0

    for scenario in scenarios:
        output = store.retrieve_output(
            RunId(scenario.scenario_id, scenario.replication_id)
        )
        assert "results" in output


def test_run_scenarios_partial_failure(maybe_executor):
    """A failing run becomes a recorded FailureInfo while the rest succeed.

    Parametrized over sequential and process backends: under the process pool
    this proves a worker-raised failure pickles home and attributes to the
    correct RunId, not just that the sequential path records it.
    """
    scenarios = [Scenario(x=i, should_fail=(i == 1)) for i in range(3)]
    store = run_scenarios(
        scenarios,
        _ConditionalConfig(_DummyModel, until=3),
        progress=False,
        executor=maybe_executor,
    )

    assert len(store.succeeded()) == 2
    assert len(store.failed()) == 1

    failed_id = RunId(scenarios[1].scenario_id, scenarios[1].replication_id)
    assert failed_id in store.failed()
    assert store.failed()[failed_id].failure.origin == FailureOrigin.RUNNING


def test_run_scenarios_uses_provided_store():
    """Test run_scenarios for user specified store."""
    custom_store = InMemoryStore()
    returned = run_scenarios(
        [Scenario(x=0)],
        RunConfiguration(_DummyModel, until=1),
        store=custom_store,
        progress=False,
    )
    assert returned is custom_store


def test_run_scenarios_empty_input():
    """Test run_scenarios for empty input."""
    store = run_scenarios([], RunConfiguration(_DummyModel, until=1), progress=False)
    assert len(store.pending()) == 0
    assert len(store.succeeded()) == 0
    assert len(store.failed()) == 0


def test_run_scenarios_aborts_on_broken_pool():
    """A hard worker death aborts every unrun scenario and does not raise.

    Every _WorkerKillModel calls os._exit on run, so no run can complete: the
    pool breaks on the first future and all four runs flip PENDING -> ABORTED.
    Asserting the exact partition (4 aborted, 0 of everything else) distinguishes
    correct bulk-marking from a partial-marking bug that == 4 catches but > 0
    would not.
    """
    scenarios = [Scenario(x=i) for i in range(4)]
    with ProcessPoolExecutor(max_workers=2) as ex:
        store = run_scenarios(
            scenarios,
            RunConfiguration(_WorkerKillModel, until=3),
            executor=ex,
            progress=False,
        )

    # returns, does not raise
    assert len(store.aborted()) == 4
    assert len(store.succeeded()) == 0
    assert len(store.failed()) == 0
    assert len(store.pending()) == 0

    for rec in store.aborted().values():
        assert rec.failure.origin is FailureOrigin.ABORTED


def test_run_scenarios_partial_abort():
    """Runs that finish before the pool breaks are SUCCEEDED; the rest become ABORTED.

    With max_workers=1 the pool processes tasks in submission order. Scenario 0
    completes successfully before scenario 1 starts. Scenario 1 kills the worker,
    so its future surfaces as BrokenExecutor before _record can mark it — leaving
    scenarios 1, 2, and 3 all PENDING when the outer handler fires. All three
    become ABORTED.
    """
    scenarios = [
        Scenario(x=0, should_kill=False),
        Scenario(x=1, should_kill=True),
        Scenario(x=2, should_kill=False),
        Scenario(x=3, should_kill=False),
    ]
    with ProcessPoolExecutor(max_workers=1) as ex:
        store = run_scenarios(
            scenarios,
            RunConfiguration(_ConditionalKillModel, until=3),
            executor=ex,
            progress=False,
        )

    assert len(store.succeeded()) == 1
    assert len(store.aborted()) == 3
    assert len(store.failed()) == 0
    assert len(store.pending()) == 0

    succeeded_id = RunId(scenarios[0].scenario_id, scenarios[0].replication_id)
    assert succeeded_id in store.succeeded()
    for rec in store.aborted().values():
        assert rec.failure.origin is FailureOrigin.ABORTED


def test_run_scenarios_handles_result_transport_error():
    """A non-BrokenExecutor error from future.result() records the run as failed and continues.

    This exercises the inner except-Exception branch in the executor path, which
    handles pickling failures or CancelledError on the return trip from a worker.
    A pre-failed Future simulates that condition without spawning real processes.
    """
    from concurrent.futures import Future  # noqa: PLC0415

    class _TransportErrorExecutor:
        def submit(self, fn, *args, **kwargs):
            f = Future()
            f.set_exception(RuntimeError("simulated transport error"))
            return f

    scenarios = [Scenario(x=i) for i in range(2)]
    store = run_scenarios(
        scenarios,
        RunConfiguration(_DummyModel, until=3),
        executor=_TransportErrorExecutor(),
        progress=False,
    )

    assert len(store.failed()) == 2
    assert len(store.succeeded()) == 0
    assert len(store.pending()) == 0
    assert len(store.aborted()) == 0
    for rec in store.failed().values():
        assert rec.failure.origin == FailureOrigin.WRITING
        assert rec.failure.exception_type == "RuntimeError"
        assert "transport error" in rec.failure.message


# ============================================================
# Picklability (needed for parallel execution)
# ============================================================
# These objects are pickled to/from workers once a process-pool executor
# lands. FailureInfo in particular is a primitives-only dataclass precisely
# so it pickles (a live exception's traceback object would not). Verifying
# the round-trip here keeps a regression visible on the named object rather
# than surfacing later as a PicklingError deep in the parallel machinery.


def test_run_configuration_is_picklable():
    """RunConfiguration round-trips through pickle (sent to workers)."""
    config = RunConfiguration(
        _DummyModel, until=5, model_kwargs={"w": 3}, outcomes=["a"]
    )
    restored = pickle.loads(pickle.dumps(config))  # noqa: S301
    assert restored.model_class is _DummyModel
    assert restored.until == 5
    assert restored.model_kwargs == {"w": 3}
    assert restored.outcomes == ["a"]


def test_failure_info_is_picklable():
    """FailureInfo crosses back from worker to root; primitives-only by design."""
    fi = FailureInfo(
        origin=FailureOrigin.RUNNING,
        exception_type="RuntimeError",
        message="boom",
        traceback="tb",
    )
    restored = pickle.loads(pickle.dumps(fi))  # noqa: S301
    assert restored.origin is FailureOrigin.RUNNING
    assert restored.exception_type == "RuntimeError"
    assert restored.message == "boom"
