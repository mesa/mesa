"""Agent.py related tests."""

import numpy as np
import pandas as pd
import pytest

from mesa.agent import Agent
from mesa.model import Model


class AgentTest(Agent):
    """Agent class for testing."""

    def get_unique_identifier(self):
        """Return unique identifier for this agent."""
        return self.unique_id


def test_agent_removal():
    """Test agent removal."""
    model = Model()
    agent = AgentTest(model)
    # Check if the agent is added
    assert agent in model.agents

    agent.remove()
    # Check if the agent is removed
    assert agent not in model.agents


def test_agent_rng():
    """Test whether agent.random and agent.rng are equal to model.random and model.rng."""
    model = Model(rng=42)
    agent = Agent(model)
    assert agent.random is model.random
    assert agent.rng is model.rng


def test_agent_create():
    """Test create agent factory method."""
    # Fast Path (No args/kwargs)
    model = Model()
    n = 10
    fast_agents = Agent.create_agents(model, n)

    assert len(fast_agents) == n
    assert all(isinstance(a, Agent) for a in fast_agents)
    assert all(a.model is model for a in fast_agents)

    # Standard Path (With args/kwargs)
    class TestAgent(Agent):
        def __init__(self, model, attr, def_attr, a=0, b=0):
            super().__init__(model)
            self.some_attribute = attr
            self.some_default_value = def_attr
            self.a = a
            self.b = b

    model = Model(rng=42)
    n = 10
    some_attribute = model.rng.random(n)
    a = tuple([model.random.random() for _ in range(n)])
    TestAgent.create_agents(model, n, some_attribute, 5, a=a, b=7)

    for agent, value, a_i in zip(model.agents, some_attribute, a):
        assert agent.some_attribute == value
        assert agent.some_default_value == 5
        assert agent.a == a_i
        assert agent.b == 7


def test_agent_create_with_pandas():
    """Test create_agents with pandas Series to improve coverage."""

    class TestAgent(Agent):
        def __init__(self, model, series_attr=None, kw_series_attr=None):
            super().__init__(model)
            self.series_attr = series_attr
            self.kw_series_attr = kw_series_attr

    model = Model()
    n = 5

    # Test pandas Series as positional argument (should hit pandas detection logic)
    series_data = pd.Series([10, 20, 30, 40, 50])
    agents = TestAgent.create_agents(model, n, series_data)
    for i, agent in enumerate(agents):
        assert agent.series_attr == series_data.iloc[i]

    # Test pandas Series as keyword argument
    kw_series_data = pd.Series([100, 200, 300, 400, 500])
    agents = TestAgent.create_agents(model, n, kw_series_attr=kw_series_data)
    for i, agent in enumerate(agents):
        assert agent.kw_series_attr == kw_series_data.iloc[i]

    # Test pandas Series with length mismatch
    short_series = pd.Series([1, 2])  # length 2, but n=5
    agents = TestAgent.create_agents(model, n, short_series)
    for agent in agents:
        # Should repeat the entire series, not individual elements
        assert agent.series_attr.equals(short_series)


def test_agent_from_dataframe():
    """Test create_agents from a pandas DataFrame."""

    class TestAgent(Agent):
        def __init__(
            self,
            model,
            value=None,
            list_attr=None,
            tuple_attr=None,
            df_value=None,
            extra_attr=None,
        ):
            super().__init__(model)
            self.value = value
            self.list_attr = list_attr
            self.tuple_attr = tuple_attr
            self.df_value = df_value
            self.extra_attr = extra_attr

    model = Model()
    n = 5
    data = {
        "value": range(n),
        "list_attr": [[i] for i in range(n)],
        "df_value": [f"df_{i}" for i in range(n)],
        "tuple_attr": [(1, 2)] * n,
    }
    df = pd.DataFrame(data)

    # Test with constant (non-sequence) override via **kwargs
    agents = TestAgent.from_dataframe(model, df, extra_attr=5)

    assert len(agents) == n
    for i, agent in enumerate(agents):
        assert agent.value == i
        assert agent.list_attr == [i]
        assert agent.df_value == f"df_{i}"
        assert agent.tuple_attr == (1, 2)
        assert agent.extra_attr == 5

    # Test that passing a sequence in kwargs raises TypeError
    for bad in ([1, 2, 3], (1, 2, 3), np.array([1, 2, 3]), pd.Series([1, 2, 3])):
        with pytest.raises(TypeError, match="does not support sequence data in kwargs"):
            TestAgent.from_dataframe(model, df, list_attr=bad)

    # kwargs should override DataFrame columns on key collision
    agents = TestAgent.from_dataframe(model, df, value=999)
    assert all(a.value == 999 for a in agents)

    # empty DataFrame should create an empty AgentSet
    empty_df = pd.DataFrame(columns=list(df.columns))
    agents = TestAgent.from_dataframe(model, empty_df, extra_attr=5)
    assert len(agents) == 0

    # DataFrame index should be ignored
    df_with_index = df.copy()
    df_with_index.index = range(100, 100 + n)
    agents = TestAgent.from_dataframe(model, df_with_index, extra_attr=5)
    assert [a.value for a in agents] == list(range(n))
