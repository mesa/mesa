"""Tests for mesa.experimental.scenarios."""

import pickle

import numpy as np
import pandas as pd
import pytest
import scipy.stats.qmc as qmc

from mesa import Agent, Model
from mesa.experimental.data_collection import DataRecorder
from mesa.experimental.scenarios import (
    RunConfiguration,
    Scenario,
    ScenarioFailedException,
    ScenarioNotFoundException,
    ScenarioNotReadyException,
    rescale_samples,
    run_scenarios,
)
from mesa.experimental.scenarios.exceptions import FailureInfo, FailureOrigin
from mesa.experimental.scenarios.runner import _safe_call
from mesa.experimental.scenarios.store import (
    InMemoryReference,
    InMemoryStore,
    InMemoryWriter,
    RunId,
    Status,
)


def test_scenario():
    """Test Scenario and ModelWithScenario class."""
    Scenario._reset_counter()

    scenario = Scenario(a=1, b=2, c=3, rng=42)
    assert scenario.scenario_id == 0
    assert scenario.a == 1
    assert len(scenario) == 3
    assert isinstance(scenario.rng, np.random.Generator)

    d = scenario.to_dict()
    assert d["a"] == 1
    assert d["scenario_id"] == 0
    assert d["replication_id"] is None

    with pytest.raises(TypeError):
        scenario.c = 4

    with pytest.raises(TypeError):
        del scenario.c

    scenario = Scenario(a=1, b=2, c=3, rng=42)
    assert scenario.scenario_id == 1

    model = Model(scenario=scenario)
    assert model.scenario is scenario

    # When no scenario is passed, the auto-created scenario shares the model's Generator
    model = Model()
    assert model.scenario.rng is model.rng

    # Passing a pre-built Generator is forwarded as-is
    gen = np.random.default_rng(42)
    scenario = Scenario(rng=gen)
    model = Model(scenario=scenario)
    assert model.rng is gen


def test_scenario_serialization():
    """Test that scenarios can be pickled/unpickled."""
    scenario = Scenario(a=1, rng=42)

    pickled = pickle.dumps(scenario)
    unpickled = pickle.loads(pickled)  # noqa: S301
    assert unpickled.a == scenario.a
    assert unpickled.scenario_id == scenario.scenario_id
    assert unpickled.replication_id == scenario.replication_id
    assert unpickled.initial_rng_state == scenario.initial_rng_state

    scenario = Scenario(a=1, rng=np.random.default_rng(42))

    pickled = pickle.dumps(scenario)
    unpickled = pickle.loads(pickled)  # noqa: S301
    assert unpickled.a == scenario.a
    assert unpickled.scenario_id == scenario.scenario_id


def test_agent_scenario_property():
    """Test that agents can access scenario via property."""
    scenario = Scenario(test_param=100, another_param="test", rng=42)
    model = Model(scenario=scenario)
    agent = Agent(model)

    # Agent should have access to scenario
    assert agent.scenario is model.scenario
    assert agent.scenario.test_param == 100
    assert agent.scenario.another_param == "test"

    # Verify it's the same object, not a copy
    assert agent.scenario is agent.model.scenario


def test_scenario_subclassing():
    """Test that Scenario can be subclassed with type-hinted attributes."""

    class MyScenario(Scenario):
        density: float = 0.8
        vision: int = 7
        movement: bool = True

    # Test class-level defaults are picked up
    scenario = MyScenario(rng=42)
    assert scenario.density == 0.8
    assert scenario.vision == 7
    assert scenario.movement is True
    assert isinstance(scenario.rng, np.random.Generator)

    # Test overriding defaults
    scenario = MyScenario(rng=42, density=0.5, vision=10)
    assert scenario.density == 0.5
    assert scenario.vision == 10
    assert scenario.movement is True  # Not overridden, still default


def test_scenario_subclass_with_model():
    """Test that scenario subclasses work correctly with Model."""

    class TestScenario(Scenario):
        citizen_density: float = 0.7
        cop_vision: int = 7

    # Create scenario and pass to model
    scenario = TestScenario(rng=42, citizen_density=0.8)
    model = Model(scenario=scenario)

    # Verify model has correct scenario type
    assert isinstance(model.scenario, TestScenario)
    assert model.scenario.citizen_density == 0.8
    assert model.scenario.cop_vision == 7


def test_scenario_frozen():
    """Test that scenario parameters cannot be modified after initialisation."""

    class MyScenario(Scenario):
        counter: int = 0

    scenario = MyScenario(rng=42)
    assert scenario.counter == 0

    with pytest.raises(TypeError):
        scenario.counter = 5

    with pytest.raises(TypeError):
        del scenario.counter

    # Two scenarios created from the same defaults are independent
    scenario1 = MyScenario(rng=42)
    scenario2 = MyScenario(rng=43, counter=5)
    assert scenario1.counter == 0
    assert scenario2.counter == 5


def test_scenario_spawn_replications():
    """Test that replicate() produces correctly seeded copies."""

    class MyScenario(Scenario):
        density: float = 0.8

    base = MyScenario(rng=42, scenario_id=3)
    replicas = base.spawn_replications(5)

    assert len(replicas) == 5
    for i, r in enumerate(replicas):
        assert r.replication_id == i
        assert r.scenario_id == 3
        assert r.density == 0.8
        assert (
            r.initial_rng_state != base.initial_rng_state
        )  # derived seed, not the same

    # Seeds are deterministic: same base produces same replicas
    base2 = MyScenario(rng=42, scenario_id=3)
    replicas2 = base2.spawn_replications(5)
    for r1, r2 in zip(replicas, replicas2):
        assert r1.initial_rng_state == r2.initial_rng_state, (
            "generators are not the same"
        )

    # Replicas are also frozen
    with pytest.raises(TypeError):
        replicas[0].density = 0.5

    # SeedSequence rng works and is reproducible
    base_1 = MyScenario(rng=np.random.SeedSequence(42))
    base_2 = MyScenario(rng=np.random.SeedSequence(42))
    replicas_ss1 = base_1.spawn_replications(3)
    replicas_ss2 = base_2.spawn_replications(3)
    for r1, r2 in zip(replicas_ss1, replicas_ss2):
        assert r1.initial_rng_state == r2.initial_rng_state, (
            "generators are not the same"
        )


def test_scenario_from():
    """Test that scenario generation from numpy/pandas dataframe."""
    # we don't directly test from_dataframe because its called by from_numpy.
    # create a 100X3 LHS sample on unit interval
    d = 3
    n = 100
    parameter_names = ["a", "b", "c"]
    samples = qmc.LatinHypercube(d).random(n)

    # check scenario generation
    scenarios = Scenario.from_ndarray(samples, parameter_names=parameter_names, rng=42)
    assert len(scenarios) == n
    assert len(scenarios[0]) == d

    for scenario in scenarios:
        values = samples[scenario.scenario_id, :]
        for i, entry in enumerate(parameter_names):
            assert values[i] == getattr(scenario, entry)

    # check replication creation
    replications = 10
    scenarios = Scenario.from_ndarray(
        samples, parameter_names=parameter_names, rng=42, replications=replications
    )
    assert len(scenarios) == n * replications
    assert len(scenarios[0]) == d

    for j, scenario in enumerate(scenarios[0:10]):
        assert scenario.replication_id == j
        values = samples[scenario.scenario_id, :]
        for i, entry in enumerate(parameter_names):
            assert values[i] == getattr(scenario, entry)

    # check if parameter names matches number of columns in numpy array
    with pytest.raises(ValueError):
        Scenario.from_ndarray(samples, parameter_names=[], rng=42)


def test_rescale_basic():
    """Test basic rescaling from unit interval to parameter ranges."""
    samples = np.array([[0.0, 0.5, 1.0], [0.25, 0.75, 0.5]])

    ranges = np.array([[0, 10], [10, 20], [-1, 1]])

    scaled = rescale_samples(samples, ranges)

    expected = np.array([[0, 15, 1], [2.5, 17.5, 0]])

    assert np.allclose(scaled, expected)


def test_rescale_shape_preserved():
    """Rescaling should preserve the (n, d) shape of samples."""
    samples = np.random.random((50, 4))
    ranges = np.array([[0, 1], [10, 20], [-5, 5], [100, 200]])

    scaled = rescale_samples(samples, ranges)

    assert scaled.shape == samples.shape


def test_rescale_negative_ranges():
    """Rescale should correctly handle negative parameter ranges."""
    samples = np.array([[0.0, 1.0], [0.5, 0.25]])

    ranges = np.array([[-10, -2], [-5, 5]])

    scaled = rescale_samples(samples, ranges)

    expected = np.array([[-10, 5], [-6, -2.5]])

    assert np.allclose(scaled, expected)


def test_rescale_single_dimension():
    """Rescale should work for a single parameter dimension."""
    samples = np.array([[0.0], [0.5], [1.0]])
    ranges = np.array([[10, 20]])

    scaled = rescale_samples(samples, ranges)

    expected = np.array([[10], [15], [20]])

    assert np.allclose(scaled, expected)


def test_rescale_dimension_mismatch():
    """Rescale should raise error if dimensions do not match."""
    samples = np.random.random((10, 3))
    ranges = np.array([[0, 1], [10, 20]])  # only 2 ranges

    with pytest.raises(ValueError):
        rescale_samples(samples, ranges)


def test_rescale_large_sample():
    """Rescale should work correctly for larger experiment matrices."""
    samples = np.random.random((1000, 5))
    ranges = np.array(
        [
            [0, 10],
            [10, 20],
            [-5, 5],
            [100, 200],
            [0, 1],
        ]
    )

    scaled = rescale_samples(samples, ranges)

    assert scaled.shape == samples.shape
    assert np.all(scaled[:, 0] >= 0)
    assert np.all(scaled[:, 0] <= 10)


def test_rescale_bounds_mapping():
    """0 should map to min and 1 should map to max of each range."""
    samples = np.array([[0.0, 1.0]])
    ranges = np.array([[5, 10], [-2, 2]])

    scaled = rescale_samples(samples, ranges)

    expected = np.array([[5, 2]])
    assert np.allclose(scaled, expected)


def test_rescale_inplace():
    """Check that inplace=True modifies the original array."""
    samples = np.array([[0.0, 1.0]])
    ranges = np.array([[0, 10], [0, 10]])

    rescale_samples(samples, ranges, inplace=True)

    assert np.allclose(samples, np.array([[0, 10]]))


def test_from_ndarray_returns_subclass():
    """from_ndarray called on a subclass should return instances of that subclass."""

    class MyScenario(Scenario):
        x: float = 0.5

    samples = np.array([[0.1], [0.2]])
    scenarios = MyScenario.from_ndarray(samples, ["x"], rng=42)

    assert isinstance(scenarios[0], MyScenario)


def test_run_configuration(mocker):
    """Tests for RunConfiguration."""
    dummy_recorder = mocker.Mock(spec=DataRecorder)

    class DummyModel(Model):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.data_recorder = dummy_recorder()

            # setting up the mock
            self.data_recorder.get_table_dataframe.return_value = pd.DataFrame()
            self.data_recorder.get_all_dataframes.return_value = {
                "a": pd.DataFrame(),
                "b": pd.DataFrame(),
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
# Shared helpers for store / runner tests
# ============================================================


class _DummyRecorder:
    def get_all_dataframes(self):
        return {"results": pd.DataFrame({"x": [1]})}

    def get_table_dataframe(self, key):
        return pd.DataFrame({"x": [1]})


class _DummyModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_recorder = _DummyRecorder()


class _InstantiationFailModel(Model):
    def __init__(self, *args, **kwargs):
        raise RuntimeError("cannot instantiate")


class _RunFailModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_recorder = _DummyRecorder()

    def run_until(self, until):
        raise RuntimeError("run failed")


class _FailingRecorder:
    def get_all_dataframes(self):
        raise RuntimeError("extraction failed")


class _ExtractionFailModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_recorder = _FailingRecorder()


class _FailingWriter:
    def to_reference(self, run_id, outcome):
        raise OSError("disk full")


@pytest.fixture
def basic_config():
    """Basic scenario configuration."""
    return RunConfiguration(_DummyModel, until=5)


@pytest.fixture
def scenario_list():
    """Scenario list."""
    Scenario._reset_counter()
    return [Scenario(x=i) for i in range(3)]


@pytest.fixture
def populated_store(scenario_list):
    """Populated InMemoryStore."""
    store = InMemoryStore()
    store.write_scenarios(scenario_list)
    return store, scenario_list


# ============================================================
# InMemoryStore
# ============================================================


def test_store_write_and_read_scenarios(scenario_list):
    """Store and read scenarios."""
    store = InMemoryStore()
    store.write_scenarios(scenario_list)
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


def test_store_unknown_run_id_raises():
    """Test the unknown run_id raises exception."""
    store = InMemoryStore()
    with pytest.raises(ScenarioNotFoundException) as exc_info:
        store.check_status(RunId(999, None))
    assert exc_info.value.run_id == RunId(999, None)


def test_store_status_dataframe(populated_store):
    """Test the status dataframe store."""
    store, scenarios = populated_store
    df = store.status()
    assert list(df.columns) == ["status"]
    assert len(df) == len(scenarios)
    assert (df["status"] == "PENDING").all()


def test_store_status_dataframe_mixed_case(populated_store):
    """Test the status dataframe store with a mix of successes and failures."""
    store, scenarios = populated_store
    writer = store.writer()

    s0, s1, s2 = scenarios
    run_id_0 = RunId(s0.scenario_id, s0.replication_id)
    run_id_1 = RunId(s1.scenario_id, s1.replication_id)

    store.mark_succeeded(writer.to_reference(run_id_0, {}))
    store.mark_failed(run_id_1, FailureInfo(FailureOrigin.RUNNING, "E", "m", ""))

    df = store.status()
    # pandas converts None replication_id to NaN in the MultiIndex, so look up by scenario_id
    by_id = dict(zip(df.reset_index()["scenario_id"], df["status"]))
    assert by_id[s0.scenario_id] == "SUCCEEDED"
    assert by_id[s1.scenario_id] == "FAILED"
    assert by_id[s2.scenario_id] == "PENDING"


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


def test_run_scenarios_all_succeed():
    """Test the successful branch of run_scenarios."""
    Scenario._reset_counter()
    scenarios = [Scenario(x=i) for i in range(4)]
    store = run_scenarios(
        scenarios, RunConfiguration(_DummyModel, until=3), progress=False
    )

    assert len(store.succeeded()) == 4
    assert len(store.failed()) == 0
    assert len(store.pending()) == 0

    for scenario in scenarios:
        output = store.retrieve_output(
            RunId(scenario.scenario_id, scenario.replication_id)
        )
        assert "results" in output


def test_run_scenarios_partial_failure():
    """Test run_scenarios with a mix of successes and failures."""
    Scenario._reset_counter()
    scenarios = [Scenario(x=i, should_fail=(i == 1)) for i in range(3)]

    class _ConditionalConfig(RunConfiguration):
        def run_model(self, model):
            if getattr(model.scenario, "should_fail", False):
                raise RuntimeError("intentional")
            super().run_model(model)

    store = run_scenarios(
        scenarios, _ConditionalConfig(_DummyModel, until=3), progress=False
    )

    assert len(store.succeeded()) == 2
    assert len(store.failed()) == 1

    failed_id = RunId(scenarios[1].scenario_id, scenarios[1].replication_id)
    assert failed_id in store.failed()
    assert store.failed()[failed_id].failure.origin == FailureOrigin.RUNNING


def test_run_scenarios_uses_provided_store():
    """Test run_scenarios for user specified store."""
    Scenario._reset_counter()
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


# ============================================================
# Exception constructors
# ============================================================


@pytest.mark.parametrize(
    "exc_class, kwargs",
    [
        (ScenarioNotFoundException, {}),
        (ScenarioNotFoundException, {"run_id": RunId(1, None)}),
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