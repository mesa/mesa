"""Shared fixtures and dummy Model/Store helpers for mesa.experimental.scenarios tests."""

from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import pytest

from mesa import Model
from mesa.experimental.scenarios import RunConfiguration, Scenario
from mesa.experimental.scenarios.store import InMemoryStore


@pytest.fixture(autouse=True)
def _reset_scenario_ids():
    Scenario._ids.clear()


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


@pytest.fixture(params=["sequential", "process"])
def maybe_executor(request):
    """Fixture controlling the executor to use."""
    if request.param == "sequential":
        yield None
    else:
        with ProcessPoolExecutor(max_workers=2) as ex:
            yield ex
