"""Tests for mesa.experimental.scenarios."""

import numpy as np
import pytest

from mesa.experimental.scenarios import ModelWithScenario, Scenario


def test_scenario():
    """Test Scenario and ModelWithScenario class."""
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

    model = ModelWithScenario(scenario=scenario)
    model.running = True
    assert model.scenario.model is model

    with pytest.raises(ValueError):
        scenario.a = 5

    model = ModelWithScenario()
    assert model.scenario.rng is None

    gen = np.random.default_rng(42)
    scenario = Scenario(rng=gen)
    model = ModelWithScenario(scenario=scenario)
    # Should work without error
    assert model.rng is not None


def test_scenario_serialization():
    """Test that scenarios can be pickled/unpickled."""
    import pickle

    scenario = Scenario(a=1, rng=42)

    pickled = pickle.dumps(scenario)
    unpickled = pickle.loads(pickled)
    assert unpickled.a == scenario.a
    assert unpickled._scenario_id == scenario._scenario_id

    scenario = Scenario(a=1, rng=np.random.default_rng(42))

    pickled = pickle.dumps(scenario)
    unpickled = pickle.loads(pickled)
    assert unpickled.a == scenario.a
    assert unpickled._scenario_id == scenario._scenario_id
