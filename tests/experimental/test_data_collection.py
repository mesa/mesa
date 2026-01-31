import pytest
import numpy as np
from mesa.experimental.data_collection import (
    DataRegistry,
    AgentDataSet,
    NumpyAgentDataSet,
    ModelDataSet,
    TableDataSet,
)

from mesa import Model, Agent


def test_data_registry():
    """Test DataRegistry."""
    registry = DataRegistry()


def test_agent_dataset():
    """Test AgentDataSet."""

    class MyAgent(Agent):
        def __init__(self, model, value):
            super().__init__(model)
            self.test = value

    class MyModel(Model):
        def __init__(self, rng=42, n=100):
            super().__init__(rng=rng)

            MyAgent.create_agents(
                self,
                n,
                self.rng.random(
                    size=n,
                ),
            )

    n = 100
    model = MyModel(n=n)
    dataset = AgentDataSet("test", model.agents, "test")

    values = dataset.data
    assert len(values) == n

    single_field = values[0]
    assert "unique_id" in single_field
    assert "test" in single_field

    dataset.close()
    assert dataset._closed
    with pytest.raises(RuntimeError):
        _ = dataset.data
    dataset.close()


def test_numpy_agent_dataset():
    """Test NumpyAgentDataSet."""

    class MyAgent(Agent):
        def __init__(self, model, value):
            super().__init__(model)
            self.test = value

    class MyModel(Model):
        def __init__(self, rng=42, n=100):
            super().__init__(rng=rng)

            self.dataset = NumpyAgentDataSet("test", MyAgent, "test", dtype=float)
            self.data_registry.add_dataset(self.dataset)
            MyAgent.create_agents(
                self,
                n,
                self.rng.random(
                    size=n,
                ),
            )

    n = 150
    model = MyModel(n=n)
    dataset = model.dataset

    values = dataset.data
    assert values.shape == (n, 1)

    assert values[0, 0] == model.agents.to_list()[0].test

    dataset.close()
    assert dataset._closed
    with pytest.raises(RuntimeError):
        _ = dataset.data
    dataset.close()


def test_model_dataset():
    """Test ModelDataSet."""
    class MyAgent(Agent):
        def __init__(self, model, value):
            super().__init__(model)
            self.test = value

    class MyModel(Model):

        @property
        def mean_value(self):
            data = self.agents.get("test")
            return np.mean(data)

        def __init__(self, rng=42, n=100):
            super().__init__(rng=rng)
            self.data_registry.track_model(self, "model_data", "mean_value", )
            MyAgent.create_agents(
                self,
                n,
                self.rng.random(
                    size=n,
                ),
            )

    model = MyModel(n=100)
    data = model.data_registry["model_data"].data

    assert len(data) == 1

    model.data_registry.close()
    assert model.data_registry["model_data"]._closed
    with pytest.raises(RuntimeError):
        _ = model.data_registry["model_data"].data


def test_data_dataset():
    """Test ModelDataSet."""
    dataset = TableDataSet("test", "test")