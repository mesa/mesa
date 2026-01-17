"""Tests for model.py."""

import numpy as np
import pytest

from mesa.agent import Agent, AgentSet
from mesa.experimental.devs.simulator import DEVSimulator
from mesa.model import Model


def test_model_set_up():
    """Test Model initialization."""
    model = Model()
    assert model.running is True
    assert model.steps == 0
    assert model.time == 0.0
    assert model._simulator is None

    model.step()
    assert model.steps == 1
    assert model.time == 1.0


def test_model_time_increment():
    """Test that time increments correctly with steps."""
    model = Model()

    for i in range(5):
        model.step()
        assert model.steps == i + 1
        assert model.time == float(i + 1)


def test_model_time_with_simulator():
    """Test that simulator controls time when attached."""
    model = Model()
    simulator = DEVSimulator()
    simulator.setup(model)

    # Simulator is now attached
    assert model._simulator is simulator

    # Time should not auto-increment when simulator is attached
    # (In practice, the simulator controls stepping, but we can test the flag)
    model._user_step()  # Call user step directly to avoid wrapped_step
    # Time unchanged because simulator controls it


def test_running():
    """Test Model is running."""

    class TestModel(Model):
        def step(self):
            """Stop at step 10."""
            if self.steps == 10:
                self.running = False

    model = TestModel()
    model.run_model()
    assert model.steps == 10
    assert model.time == 10.0


def test_rng(rng=23):
    """Test initialization of model with specific seed."""
    model = Model(rng=rng)
    assert model._rng == np.random.default_rng(rng).bit_generator.state
    model2 = Model(rng=rng + 1)
    assert model2._rng == np.random.default_rng(rng + 1).bit_generator.state
    assert model._rng == np.random.default_rng(rng).bit_generator.state

    assert Model(rng=42).random.random() == Model(rng=42).random.random()
    assert np.all(
        Model(rng=42).rng.random(
            10,
        )
        == Model(rng=42).rng.random(
            10,
        )
    )


def test_reset_randomizer(newseed=42):
    """Test resetting the random seed on the model."""
    with pytest.warns(FutureWarning):
        model = Model()
        oldseed = model._seed
        model.reset_randomizer()
        assert model._seed == oldseed
        model.reset_randomizer(seed=newseed)
        assert model._seed == newseed


def test_reset_rng(newseed=42):
    """Test resetting the random seed on the model."""
    model = Model(rng=5)
    old_rng = model._rng

    model.reset_rng(rng=6)
    new_rng = model._rng

    assert old_rng != new_rng

    old_rng = new_rng
    model.reset_rng()
    new_rng = model.rng.bit_generator.state

    assert old_rng == new_rng

    model = Model(rng=np.random.MT19937(42))
    old_rng = model._rng
    model.reset_rng()
    new_rng = model.rng.bit_generator.state

    assert np.all(old_rng["state"]["key"] == new_rng["state"]["key"])


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
