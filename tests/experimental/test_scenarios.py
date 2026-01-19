"""Tests for mesa.experimental.scenarios."""

import pickle

import numpy as np
import pytest

from mesa import Agent, Model
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
    assert scenario.rng == 42

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


def test_scenario_fresh_instance_per_model():
    """Test that each model gets a fresh scenario instance (no shared state)."""

    class MyScenario(Scenario):
        counter: int = 0

    # Create first model
    scenario1 = MyScenario(rng=42)
    model1 = Model(scenario=scenario1)
    model1.running = True

    # Create second model with fresh scenario
    scenario2 = MyScenario(rng=43)
    model2 = Model(scenario=scenario2)
    model2.running = False  # Not running yet

    # Should not raise error - different scenario instances
    scenario2.counter = 5
    assert scenario2.counter == 5
    assert scenario1.counter == 0
