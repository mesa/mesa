"""Tests for mesa.experimental.scenarios."""

import pytest

from mesa.experimental.scenarios import ModelWithScenario, Scenario


def test_scenario():
    """Test Scenario and ModelWithScenario class."""
    scenario = Scenario(a=1, b=2, c=3, rng=42)
    assert scenario.scenario_id == 0
    assert scenario.model is None
    assert scenario.a == 1
    assert len(scenario) == 4

    values = {"a": 1, "b": 2, "c": 3, "rng": 42}
    for k, v in scenario.items():
        assert values[k] == v
    assert scenario.to_dict() == {
        "a": 1,
        "b": 2,
        "c": 3,
        "rng": 42,
        "model": None,
        "scenario_id": 0,
    }

    scenario.c = 4
    assert scenario.c == 4

    del scenario.c
    with pytest.raises(AttributeError):
        _ = scenario.c

    scenario = Scenario(**values)
    assert scenario.scenario_id == 1

    model = ModelWithScenario(scenario=scenario)
    model.running = True
    assert model.scenario.model is model

    with pytest.raises(ValueError):
        scenario.a = 5

    model = ModelWithScenario()
    assert model.scenario.rng is None
