"""Tests for model.py."""

import numpy as np
import pytest

from mesa.agent import Agent, AgentSet
from mesa.model import Model


def test_model_set_up():
    """Test Model initialization."""
    model = Model()
    assert model.running is True
    assert model.time == 0.0

    model.step()
    assert model.time == 1.0


def test_model_time_increment():
    """Test that time increments correctly with steps."""
    model = Model()

    for i in range(5):
        model.step()
        assert model.time == float(i + 1)


def test_running():
    """Test Model is running."""

    class TestModel(Model):
        def step(self):
            """Stop at step 10."""
            if self.time == 10:
                self.running = False

    model = TestModel()
    model.run_model()
    assert model.time == 10.0


def test_step_continues_after_exception():
    """An exception in step() reaches the caller, and later steps still run."""

    class FailOnce(Model):
        def __init__(self):
            super().__init__()
            self.steps_run = 0

        def step(self):
            self.steps_run += 1
            if self.steps_run == 2:
                raise RuntimeError("boom")

    model = FailOnce()
    model.step()
    with pytest.raises(RuntimeError, match="boom"):
        model.step()
    for _ in range(3):
        model.step()

    assert model.steps_run == 5
    assert model.time == 5.0


def test_rng(rng=23):
    """Test initialization of model with specific seed."""
    model = Model(rng=rng)
    assert model.scenario.seed_sequence.entropy == rng
    model2 = Model(rng=rng + 1)
    assert model2.scenario.seed_sequence.entropy == rng + 1
    assert model.scenario.seed_sequence.entropy == rng

    assert Model(rng=42).random.random() == Model(rng=42).random.random()
    assert np.all(
        Model(rng=42).rng.random(
            10,
        )
        == Model(rng=42).rng.random(
            10,
        )
    )


def test_agent_types():
    """Test Model.agent_types property."""

    class TestAgent(Agent):
        pass

    model = Model()
    test_agent = TestAgent(model)
    assert test_agent in model.agents
    assert type(test_agent) in model.agent_types


def test_agents_by_type():
    """Test getting agents by type from Model."""

    class Wolf(Agent):
        pass

    class Sheep(Agent):
        pass

    model = Model()
    wolf = Wolf(model)
    sheep = Sheep(model)

    assert model.agents_by_type[Wolf] == AgentSet([wolf], random=model.random)
    assert model.agents_by_type[Sheep] == AgentSet([sheep], random=model.random)
    assert len(model.agents_by_type) == 2


def test_agent_remove():
    """Test removing all agents from the model."""

    class TestAgent(Agent):
        pass

    model = Model()
    for _ in range(100):
        TestAgent(model)
    assert len(model.agents) == 100

    model.remove_all_agents()
    assert len(model.agents) == 0


def test_remove_all_agents_keeps_datasets_open():
    """A model can replace its whole population each step and keep recording."""

    class Creature(Agent):
        def __init__(self, model):
            super().__init__(model)
            self.fitness = 1.0

    class Cell(Agent):
        def __init__(self, model):
            super().__init__(model)
            self.energy = 1.0

    model = Model(rng=42)
    registry = model.data_registry
    registry.track_model(model, "stats", fields=["time"])
    Creature.create_agents(model, 5)
    registry.track_agents(
        model.agents_by_type[Creature], "creatures", fields=["fitness"]
    )
    registry.track_agents_numpy(Cell, "cells", fields=["energy"])
    Cell.create_agents(model, 5)

    for _ in range(3):
        model.remove_all_agents()
        Creature.create_agents(model, 5)
        Cell.create_agents(model, 5)

    assert len(model.agents) == 10
    assert registry["stats"].data == {"time": 0.0}
    assert len(registry["creatures"].data) == 5
    assert len(registry["cells"].data) == 5


def test_agents_by_type_keeps_empty_bucket():
    """Removing all agents of a type keeps its empty bucket in agents_by_type.

    The per-type AgentSet is retained (as an empty set) after the last agent of
    that type is removed, and the type stays listed in agent_types. Models such
    as WolfSheep rely on this: they index agents_by_type[SomeType] every step and
    expect an empty set rather than a KeyError once that type goes extinct.
    """

    class Predator(Agent):
        pass

    class Prey(Agent):
        pass

    model = Model()
    for _ in range(3):
        Predator(model)
    for _ in range(2):
        Prey(model)

    for agent in list(model.agents_by_type[Prey]):
        agent.remove()

    assert Prey in model.agent_types
    assert len(model.agents_by_type[Prey]) == 0
    assert len(model.agents_by_type[Predator]) == 3
