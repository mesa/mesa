"""Tests for mesa.experimental.scenarios."""

import pickle

import numpy as np
import pytest

from mesa import Model
from mesa.experimental.scenarios import Scenario


def test_scenario():
    """Test Scenario and ModelWithScenario class."""
    Scenario._reset_counter()

    scenario = Scenario(a=1, b=2, c=3, rng=42)
    assert scenario._scenario_id == 0
    assert scenario.model is None
    assert scenario.a == 1
    assert len(scenario) == 4

    values = {"a": 1, "b": 2, "c": 3, "rng": 42}
    for k, v in scenario:
        assert values[k] == v
    assert scenario.to_dict() == {
        "a": 1,
        "b": 2,
        "c": 3,
        "rng": 42,
        "model": None,
        "_scenario_id": 0,
    }

    scenario.c = 4
    assert scenario.c == 4

    del scenario.c
    with pytest.raises(AttributeError):
        _ = scenario.c

    scenario = Scenario(**values)
    assert scenario._scenario_id == 1

    model = Model(scenario=scenario)
    model.running = True
    assert model.scenario.model is model

    with pytest.raises(ValueError):
        scenario.a = 5

    model = Model()
    assert model.scenario.rng is model._seed

    gen = np.random.default_rng(42)
    scenario = Scenario(rng=gen)
    model = Model(scenario=scenario)
    # Should work without error
    assert model.rng is not None
    assert (
        model.rng is gen
    )  # fixme we might want to spawn a generator (in essence a copy)


def test_scenario_serialization():
    """Test that scenarios can be pickled/unpickled."""
    scenario = Scenario(a=1, rng=42)

    pickled = pickle.dumps(scenario)
    unpickled = pickle.loads(pickled)  # noqa: S301
    assert unpickled.a == scenario.a
    assert unpickled._scenario_id == scenario._scenario_id

    scenario = Scenario(a=1, rng=np.random.default_rng(42))

    pickled = pickle.dumps(scenario)
    unpickled = pickle.loads(pickled)  # noqa: S301
    assert unpickled.a == scenario.a
    assert unpickled._scenario_id == scenario._scenario_id
