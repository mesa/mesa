"""Shared fixtures and dummy Model/Store helpers for mesa.experimental.scenarios tests."""

from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import pytest

from mesa import Model
from mesa.experimental.scenarios import (
    DiskStore,
    RunConfiguration,
    Scenario,
    disk_writer,
)
from mesa.experimental.scenarios.store import InMemoryStore


@pytest.fixture(autouse=True)
def _reset_scenario_ids():
    Scenario._ids.clear()


@pytest.fixture(autouse=True)
def _reset_disk_writer_registry():
    """disk_writer's stream registry is a module-level global, not per-store.

    Keyed only by (session, output name) — not store_dir — so it must be
    reset before and after every test, or two tests reusing the same
    session string (or a DiskStreamWriter left with an open stream) would
    silently leak state into each other. _REGISTRY_PID is reset alongside
    _CURRENT_SESSION for the same reason: leaving a stale value from a prior
    test would make the next test's first _rotate_session call believe the
    registry was inherited from a different process (harmless here, since
    the dicts are already empty either way, but the fixture should actually
    reset everything it documents itself as resetting).
    """
    disk_writer._evict(list(disk_writer._STREAMS))
    disk_writer._CURRENT_SESSION = None
    disk_writer._REGISTRY_PID = None
    yield
    disk_writer._evict(list(disk_writer._STREAMS))
    disk_writer._CURRENT_SESSION = None
    disk_writer._REGISTRY_PID = None


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


class _WorkerKillModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_recorder = _DummyRecorder()

    def run_until(self, until):
        import os  # noqa: PLC0415

        os._exit(1)  # hard-kills the worker; not catchable by _safe_call


class _ConditionalKillModel(Model):
    """Kills the worker for any scenario with should_kill=True; otherwise runs normally."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_recorder = _DummyRecorder()

    def run_until(self, until):
        if getattr(self.scenario, "should_kill", False):
            import os  # noqa: PLC0415

            os._exit(1)
        super().run_until(until)


class _ConditionalConfig(RunConfiguration):
    """Fails the run for any scenario flagged should_fail.

    Module-level (not a local class inside a test) so it pickles to process
    workers when the parallel backend runs it.
    """

    def run_model(self, model):
        if getattr(model.scenario, "should_fail", False):
            raise RuntimeError("intentional")
        super().run_model(model)


class _VariableRowsRecorder:
    """Recorder whose 'results' table has as many rows as it's constructed with."""

    def __init__(self, n_rows):
        self._n_rows = n_rows

    def get_all_dataframes(self):
        return {"results": pd.DataFrame({"value": list(range(self._n_rows))})}

    def get_table_dataframe(self, key):
        return self.get_all_dataframes()[key]


class _VariableRowsModel(Model):
    """A model whose 'results' output row count is set by scenario.n_rows.

    n_rows=0 exercises the empty-vs-populated-frame schema corner case that
    DiskStore's schemas parameter exists to avoid. Module-level so it
    pickles to process workers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_recorder = _VariableRowsRecorder(getattr(self.scenario, "n_rows", 1))


@pytest.fixture
def basic_config():
    """Basic scenario configuration."""
    return RunConfiguration(_DummyModel, until=5)


@pytest.fixture
def scenario_list():
    """Scenario list."""
    return [Scenario(x=i) for i in range(3)]


@pytest.fixture
def populated_store(scenario_list, basic_config):
    """Populated InMemoryStore."""
    store = InMemoryStore()
    store.write_scenarios(scenario_list, basic_config)
    return store, scenario_list


@pytest.fixture
def disk_store(tmp_path):
    """A fresh DiskStore rooted at a subdirectory of tmp_path."""
    return DiskStore(tmp_path / "store")


@pytest.fixture
def populated_disk_store(disk_store, scenario_list, basic_config):
    """A DiskStore with scenarios recorded (all PENDING), mirroring populated_store."""
    disk_store.write_scenarios(scenario_list, basic_config)
    return disk_store, scenario_list


@pytest.fixture(params=["sequential", "process"])
def maybe_executor(request):
    """Fixture controlling the executor to use."""
    if request.param == "sequential":
        yield None
    else:
        with ProcessPoolExecutor(max_workers=2) as ex:
            yield ex
