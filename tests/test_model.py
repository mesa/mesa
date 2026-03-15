"""Tests for model.py."""

import numpy as np

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


def test_rng(rng=23):
    """Test initialization of model with specific seed."""
    model = Model(rng=rng)
    assert (
        model.scenario.initial_rng_state
        == np.random.default_rng(rng).bit_generator.state
    )
    model2 = Model(rng=rng + 1)
    assert (
        model2.scenario.initial_rng_state
        == np.random.default_rng(rng + 1).bit_generator.state
    )
    assert (
        model.scenario.initial_rng_state
        == np.random.default_rng(rng).bit_generator.state
    )

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
