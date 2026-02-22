"""Tests for experimental datasets."""

import numpy as np
import pytest

from mesa import Agent, Model
from mesa.experimental.data_collection import (
    AgentDataSet,
    DataRegistry,
    ModelDataSet,
    NumpyAgentDataSet,
    TableDataSet,
)
from mesa.experimental.data_collection.dataset import DataSet


def test_data_registry():
    """Test DataRegistry."""
    registry = DataRegistry()
    dataset = TableDataSet("test", fields="field")

    registry.add_dataset(dataset)

    assert "test" in registry
    assert "nonexistent" not in registry
    assert registry["test"] is dataset
    assert registry.get("test") is dataset
    assert registry.get("nonexistent") is None
    assert registry["test"] is dataset

    with pytest.raises(KeyError):
        _ = registry["nonexistent"]

    with pytest.raises(
        RuntimeError, match=f"Dataset '{dataset.name}' already registered"
    ):
        registry.add_dataset(dataset)


def test_data_registry_create_dataset():
    """Test DataRegistry.create_dataset."""
    registry = DataRegistry()
    dataset = registry.create_dataset(TableDataSet, "table", fields=["a", "b"])

    assert "table" in registry
    assert registry["table"] is dataset
    assert dataset.fields == ["a", "b"]


def test_data_registry_iteration():
    """Test DataRegistry __iter__."""
    registry = DataRegistry()
    ds1 = TableDataSet("a", fields="f1")
    ds2 = TableDataSet("b", fields="f2")
    registry.add_dataset(ds1)
    registry.add_dataset(ds2)

    datasets = list(registry)
    assert len(datasets) == 2
    assert ds1 in datasets
    assert ds2 in datasets


def test_data_registry_track():
    """Test DataRegistry.track_agents convenience method."""

    class MyAgent(Agent):
        def __init__(self, model, value):
            super().__init__(model)
            self.wealth = value

    class MyModel(Model):
        @property
        def summed_wealth(self):
            return self.agents.agg("wealth", sum)

        def __init__(self, rng=42):
            super().__init__(rng=rng)
            MyAgent.create_agents(
                self,
                10,
                self.rng.random(
                    10,
                )
                * 100,
            )

    model = MyModel()
    registry = DataRegistry()
    agent_dataset = registry.track_agents(model.agents, "agent_data", fields="wealth")
    model_dataset = registry.track_model(model, "model_data", fields="summed_wealth")

    assert "agent_data" in registry
    assert "model_data" in registry

    assert len(agent_dataset.data) == 10
    assert len(model_dataset.data) == 1


def test_data_registry_close_all():
    """Test DataRegistry.close() closes all datasets."""
    registry = DataRegistry()
    ds1 = TableDataSet("a", fields="f1")
    ds2 = TableDataSet("b", fields="f2")
    registry.add_dataset(ds1)
    registry.add_dataset(ds2)

    registry.close()

    assert ds1.rows is None
    assert ds2.rows is None


def test_agent_dataset():
    """Test AgentDataSet."""

    class MyAgent(Agent):
        def __init__(self, model, value):
            super().__init__(model)
            self.test = value
            self.second_attribute = value * self.random.random()

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
    dataset = AgentDataSet("test", model.agents, fields="test")
    assert isinstance(dataset, DataSet)

    values = dataset.data
    assert len(values) == n

    single_agent = values[0]
    assert "unique_id" in single_agent
    assert "test" in single_agent

    dataset.close()
    assert dataset._closed
    with pytest.raises(RuntimeError):
        _ = dataset.data
    dataset.close()

    dataset = AgentDataSet("test", model.agents, fields=["test", "second_attribute"])
    values = dataset.data
    assert len(values) == n

    single_agent = values[0]
    assert "unique_id" in single_agent
    assert "test" in single_agent
    assert "second_attribute" in single_agent

    dataset.close()
    with pytest.raises(RuntimeError):
        _ = dataset.data
    assert dataset.agents is None


def test_numpy_agent_dataset():
    """Test NumpyAgentDataSet."""

    class MyAgent(Agent):
        def __init__(self, model, value):
            super().__init__(model)
            self.test = value
            self.first_attribute = value * self.random.random()
            self.second_attribute = value * self.random.random()

    class MyModel(Model):
        def __init__(self, rng=42, n=100):
            super().__init__(rng=rng)
            self.data_registry.track_agents_numpy(
                MyAgent, "my_data", fields=["first_attribute", "second_attribute"]
            )
            self.dataset = NumpyAgentDataSet("test", MyAgent, fields="test", dtype=int)
            self.data_registry.add_dataset(self.dataset)
            MyAgent.create_agents(
                self,
                n,
                self.rng.integers(
                    0,
                    10,
                    size=n,
                ),
            )

    n = 150
    model = MyModel(n=n)
    agents = model.agents.to_list()
    dataset = model.dataset

    assert isinstance(dataset, DataSet)

    values = dataset.data
    assert values.shape == (n, 1)

    for agent in agents:
        index = agent.__dict__[dataset._index_in_table]
        assert dataset._agent_ids[index] == agent.unique_id
        assert dataset.data[index, 0] == agent.test

    my_data = model.data_registry.get("my_data")
    values = my_data.data
    assert values.shape == (n, 2)

    active = dataset.active_agents
    assert len(active) == len(agents)
    assert set(active) == set(agents)

    # let's get a copy of the data and mutate it to ensure it's truly a copy
    a = my_data.data_copy
    a[:, 0] = -1.0
    assert np.all(my_data.data[:, 0] != -1)

    dataset.close()
    assert dataset._closed
    with pytest.raises(RuntimeError):
        _ = dataset.data

    with pytest.raises(RuntimeError):
        _ = dataset.active_agents

    with pytest.raises(RuntimeError):
        _ = dataset.add_agent(MyAgent(model, 2))

    with pytest.raises(RuntimeError):
        dataset.remove_agent(MyAgent(model, 2))

    dataset.close()

    with pytest.raises(ValueError, match="please pass one or more fields to collect"):
        NumpyAgentDataSet("test", MyAgent)


def test_numpy_agent_dataset_remove_agent():
    """Test NumpyAgentDataSet.remove_agent with swap-with-last semantics."""

    class MyAgent(Agent):
        def __init__(self, model, value):
            super().__init__(model)
            self.value = value

    class MyModel(Model):
        def __init__(self):
            super().__init__()
            self.dataset = NumpyAgentDataSet(
                "test", MyAgent, fields="value", dtype=float
            )
            self.data_registry.add_dataset(self.dataset)

            MyAgent.create_agents(self, 5, [0.0, 1.0, 2.0, 3.0, 4.0])

    model = MyModel()
    dataset = model.dataset
    agents = model.agents.to_list()

    assert len(dataset) == 5

    # Remove agent at index 1 (value=1.0)
    # The last agent (value=4.0) should be swapped into its position
    agent_to_remove = agents[1]
    last_agent = agents[4]

    dataset.remove_agent(agent_to_remove)

    for agent in agents:
        if agent is agent_to_remove:
            continue
        index = agent.__dict__[dataset._index_in_table]
        assert dataset._agent_ids[index] == agent.unique_id
        assert dataset.data[index, 0] == agent.value

    assert len(dataset) == 4
    # The last agent should now be at index 1
    assert dataset.data[1, 0] == 4.0
    # Verify the agent's internal index was updated
    assert last_agent.__dict__[dataset._index_in_table] == 1

    with pytest.raises(ValueError):
        dataset.remove_agent(agent_to_remove)

    model = MyModel()
    dataset = model.dataset
    agents = model.agents.to_list()

    # Remove the last agent
    dataset.remove_agent(agents[4])

    assert len(dataset) == 4
    assert dataset.data[0, 0] == 0.0
    assert dataset.data[1, 0] == 1.0


def test_numpy_agent_dataset_expand_storage():
    """Test NumpyAgentDataSet auto-expansion when exceeding initial capacity."""

    class MyAgent(Agent):
        def __init__(self, model, value):
            super().__init__(model)
            self.value = value

    class MyModel(Model):
        def __init__(self):
            super().__init__()
            # Start with small capacity
            self.dataset = NumpyAgentDataSet(
                "test", MyAgent, fields="value", n=5, dtype=float
            )
            self.data_registry.add_dataset(self.dataset)
            # Add more agents than initial capacity
            for i in range(20):
                MyAgent(self, value=float(i))

    model = MyModel()
    dataset = model.dataset

    assert len(dataset) == 20
    assert dataset._agent_data.shape[0] >= 20
    # Verify all data is correct after expansion
    for i in range(20):
        assert dataset.data[i, 0] == float(i)


def test_numpy_agent_dataset_property_cleanup_on_close():
    """Test that properties are removed from agent class on close."""

    class MyAgent(Agent):
        pass

    class MyModel(Model):
        def __init__(self):
            super().__init__()
            self.dataset = NumpyAgentDataSet(
                "test", MyAgent, fields="value", dtype=float
            )
            self.data_registry.add_dataset(self.dataset)

    model = MyModel()

    # Property should exist
    assert hasattr(MyAgent, "value")
    assert isinstance(getattr(MyAgent.__class__, "value", None), property) or hasattr(
        MyAgent, "value"
    )

    model.dataset.close()

    # Property should be removed
    assert not hasattr(MyAgent, "value")


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

        @property
        def std_value(self):
            data = self.agents.get("test")
            return np.std(data)

        def __init__(self, rng=42, n=100):
            super().__init__(rng=rng)
            self.data_registry.track_model(
                self,
                "model_data",
                fields="mean_value",
            )
            MyAgent.create_agents(
                self,
                n,
                self.rng.random(
                    size=n,
                ),
            )

    model = MyModel(n=100)
    data = model.data_registry["model_data"].data
    assert isinstance(model.data_registry["model_data"], DataSet)

    assert len(data) == 1

    model.data_registry.close()
    assert model.data_registry["model_data"]._closed
    with pytest.raises(RuntimeError):
        _ = model.data_registry["model_data"].data

    dataset = ModelDataSet("test", model, fields=["mean_value", "std_value"])
    data = dataset.data
    assert len(data) == 2
    dataset.close()
    assert dataset.model is None


def test_table_dataset():
    """Test TableDataSet."""
    dataset = TableDataSet("test", fields="test")
    assert isinstance(dataset, DataSet)
    assert dataset.fields == ["test"]

    dataset = TableDataSet("test", fields=["a", "b", "c"])
    assert dataset.fields == ["a", "b", "c"]

    dataset.add_row({"a": 1, "b": 2, "c": 99})
    dataset.add_row({"a": 3, "b": 4, "c": 5})

    with pytest.raises(ValueError):
        dataset.add_row({"a": 3, "b": 4, "c": 5, "extra": "value"})

    with pytest.raises(ValueError):
        dataset.add_row({"a": 3, "b": 4})

    assert len(dataset.data) == 2
    assert dataset.data[0] == {"a": 1, "b": 2, "c": 99}
    assert dataset.data[1] == {"a": 3, "b": 4, "c": 5}

    dataset.close()

    with pytest.raises(RuntimeError, match="has been closed"):
        dataset.add_row({"field": 2})

    with pytest.raises(RuntimeError, match="has been closed"):
        _ = dataset.data


def test_agent_dataset_dirty_flag():
    """Test optional manual dirty flag caching behavior in AgentDataSet."""

    class MyAgent(Agent):
        def __init__(self, model, value):
            super().__init__(model)
            self.wealth = value

    class MyModel(Model):
        def __init__(self):
            super().__init__()
            MyAgent.create_agents(self, 5, [1, 2, 3, 4, 5])

    model = MyModel()

    # Default behavior (no cache)
    dataset = AgentDataSet("wealth", model.agents, fields="wealth")

    first = dataset.data
    second = dataset.data

    assert first is not second
    agent = model.agents.to_list()[0]
    agent.wealth = 999

    third = dataset.data
    assert third[0]["wealth"] == 999

    dataset.close()

    # Opt-in dirty flag caching behavior
    dataset = AgentDataSet(
        "wealth_cached",
        model.agents,
        fields="wealth",
        use_dirty_flag=True,
    )

    first = dataset.data
    second = dataset.data
    assert first is second

    agent.wealth = 1234

    third = dataset.data
    assert third is first
    assert third[0]["wealth"] != 1234

    dataset.set_dirty_flag()
    fourth = dataset.data

    assert fourth is not first
    assert fourth[0]["wealth"] == 1234

    dataset.close()
    with pytest.raises(RuntimeError):
        dataset.set_dirty_flag()
