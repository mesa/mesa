"""Tests for mesa.experimental.scenarios."""

import pickle

import numpy as np
import pytest
import scipy.stats.qmc as qmc

from mesa import Agent, Model
from mesa.experimental.scenarios import Scenario


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
