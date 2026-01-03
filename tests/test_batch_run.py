"""Test Batchrunner."""

import pytest

import mesa
from mesa.agent import Agent
from mesa.batchrunner import _make_model_kwargs
from mesa.datacollection import DataCollector
from mesa.model import Model


def test_make_model_kwargs():  # noqa: D103
    assert _make_model_kwargs({"a": 3, "b": 5}) == [{"a": 3, "b": 5}]
    assert _make_model_kwargs({"a": 3, "b": range(3)}) == [
        {"a": 3, "b": 0},
        {"a": 3, "b": 1},
        {"a": 3, "b": 2},
    ]
    assert _make_model_kwargs({"a": range(2), "b": range(2)}) == [
        {"a": 0, "b": 0},
        {"a": 0, "b": 1},
        {"a": 1, "b": 0},
        {"a": 1, "b": 1},
    ]
    # If the value is a single string, do not iterate over it.
    assert _make_model_kwargs({"a": "value"}) == [{"a": "value"}]


def test_batch_run_with_params_with_empty_content():
    """Test handling of empty iterables in model kwargs."""
    # If "a" is a single value and "b" is an empty list (should raise error for the empty list)
    parameters_with_empty_list = {
        "a": 3,
        "b": [],
    }

    try:
        _make_model_kwargs(parameters_with_empty_list)
        raise AssertionError(
            "Expected ValueError for empty iterable but no error was raised."
        )
    except ValueError as e:
        assert "contains an empty iterable" in str(e)

    # If "a" is a iterable and "b" is an empty list (should still raise error)
    parameters_with_empty_b = {
        "a": [1, 2],
        "b": [],
    }

    try:
        _make_model_kwargs(parameters_with_empty_b)
        raise AssertionError(
            "Expected ValueError for empty iterable but no error was raised."
        )
    except ValueError as e:
        assert "contains an empty iterable" in str(e)


class MockAgent(Agent):
    """Minimalistic agent implementation for testing purposes."""

    def __init__(self, model, val):
        """Initialize a MockAgent.

        Args:
            model: a model instance
            val: a value for attribute
        """
        super().__init__(model)
        self.val = val
        self.local = 0

    def step(self):  # noqa: D102
        self.val += 1
        self.local += 0.25


class MockModel(Model):
    """Minimalistic model for testing purposes."""

    def __init__(
        self,
        variable_model_param=None,
        variable_agent_param=None,
        fixed_model_param=None,
        enable_agent_reporters=True,
        n_agents=3,
        seed=None,
        **kwargs,
    ):
        """Initialize a MockModel.

        Args:
            variable_model_param: variable model parameters
            variable_agent_param: variable agent parameters
            fixed_model_param: fixed model parameters
            enable_agent_reporters: whether to enable agent reporters
            n_agents: number of agents
            seed : random seed
            kwargs: keyword arguments
        """
        super().__init__(seed=seed, **kwargs)
        self.variable_model_param = variable_model_param
        self.variable_agent_param = variable_agent_param
        self.fixed_model_param = fixed_model_param
        self.n_agents = n_agents
        if enable_agent_reporters:
            agent_reporters = {"agent_id": "unique_id", "agent_local": "local"}
        else:
            agent_reporters = None
        self.datacollector = DataCollector(
            model_reporters={"reported_model_param": self.get_local_model_param},
            agent_reporters=agent_reporters,
        )
        self.running = True
        self.init_agents()

    def init_agents(self):
        """Initialize agents."""
        if self.variable_agent_param is None:
            agent_val = 1
        else:
            agent_val = self.variable_agent_param
        for _ in range(self.n_agents):
            MockAgent(self, agent_val)

    def get_local_model_param(self):  # noqa: D102
        return 42

    def step(self):  # noqa: D102
        self.agents.do("step")
        self.datacollector.collect(self)


def test_batch_run():  # noqa: D103
    result = mesa.batch_run(MockModel, {}, number_processes=2, rng=42)
    assert result == [
        {
            "RunId": 0,
            "iteration": 0,
            "Step": 1000,
            "reported_model_param": 42,
            "AgentID": 1,
            "agent_id": 1,
            "agent_local": 250.0,
            "seed": 42,
        },
        {
            "RunId": 0,
            "iteration": 0,
            "Step": 1000,
            "reported_model_param": 42,
            "AgentID": 2,
            "agent_id": 2,
            "agent_local": 250.0,
            "seed": 42,
        },
        {
            "RunId": 0,
            "iteration": 0,
            "Step": 1000,
            "reported_model_param": 42,
            "AgentID": 3,
            "agent_id": 3,
            "agent_local": 250.0,
            "seed": 42,
        },
    ]

    result = mesa.batch_run(MockModel, {}, number_processes=2, rng=[None])
    assert result == [
        {
            "RunId": 0,
            "iteration": 0,
            "Step": 1000,
            "reported_model_param": 42,
            "AgentID": 1,
            "agent_id": 1,
            "agent_local": 250.0,
            "seed": None,
        },
        {
            "RunId": 0,
            "iteration": 0,
            "Step": 1000,
            "reported_model_param": 42,
            "AgentID": 2,
            "agent_id": 2,
            "agent_local": 250.0,
            "seed": None,
        },
        {
            "RunId": 0,
            "iteration": 0,
            "Step": 1000,
            "reported_model_param": 42,
            "AgentID": 3,
            "agent_id": 3,
            "agent_local": 250.0,
            "seed": None,
        },
    ]

    result = mesa.batch_run(MockModel, {}, number_processes=2, rng=[42, 31415])

    # we use 2 processes, so we are not guaranteed the order of the return
    result = sorted(result, key=lambda x: (x["RunId"], x["AgentID"]))

    assert result == [
        {
            "RunId": 0,
            "iteration": 0,
            "Step": 1000,
            "reported_model_param": 42,
            "AgentID": 1,
            "agent_id": 1,
            "agent_local": 250.0,
            "seed": 42,
        },
        {
            "RunId": 0,
            "iteration": 0,
            "Step": 1000,
            "reported_model_param": 42,
            "AgentID": 2,
            "agent_id": 2,
            "agent_local": 250.0,
            "seed": 42,
        },
        {
            "RunId": 0,
            "iteration": 0,
            "Step": 1000,
            "reported_model_param": 42,
            "AgentID": 3,
            "agent_id": 3,
            "agent_local": 250.0,
            "seed": 42,
        },
        {
            "RunId": 1,
            "iteration": 1,
            "Step": 1000,
            "reported_model_param": 42,
            "AgentID": 1,
            "agent_id": 1,
            "agent_local": 250.0,
            "seed": 31415,
        },
        {
            "RunId": 1,
            "iteration": 1,
            "Step": 1000,
            "reported_model_param": 42,
            "AgentID": 2,
            "agent_id": 2,
            "agent_local": 250.0,
            "seed": 31415,
        },
        {
            "RunId": 1,
            "iteration": 1,
            "Step": 1000,
            "reported_model_param": 42,
            "AgentID": 3,
            "agent_id": 3,
            "agent_local": 250.0,
            "seed": 31415,
        },
    ]

    with pytest.raises(ValueError):
        mesa.batch_run(MockModel, {}, number_processes=2, rng=42, iterations=1)


def test_batch_run_with_params():  # noqa: D103
    mesa.batch_run(
        MockModel,
        {
            "variable_model_param": range(3),
            "variable_agent_param": range(3),
        },
        number_processes=2,
    )


def test_batch_run_no_agent_reporters():  # noqa: D103
    result = mesa.batch_run(
        MockModel, {"enable_agent_reporters": False}, number_processes=2
    )
    print(result)
    assert result == [
        {
            "RunId": 0,
            "iteration": 0,
            "Step": 1000,
            "enable_agent_reporters": False,
            "reported_model_param": 42,
            "seed": None,
        }
    ]


def test_batch_run_single_core():  # noqa: D103
    mesa.batch_run(MockModel, {}, number_processes=1, rng=[None] * 6)


def test_batch_run_unhashable_param():  # noqa: D103
    result = mesa.batch_run(
        MockModel,
        {
            "n_agents": 2,
            "variable_model_param": [{"key": "value"}],
        },
        rng=[None, None],
    )
    template = {
        "Step": 1000,
        "reported_model_param": 42,
        "agent_local": 250.0,
        "n_agents": 2,
        "variable_model_param": {"key": "value"},
        "seed": None,
    }

    assert result == [
        {
            "RunId": 0,
            "iteration": 0,
            "AgentID": 1,
            "agent_id": 1,
            **template,
        },
        {
            "RunId": 0,
            "iteration": 0,
            "AgentID": 2,
            "agent_id": 2,
            **template,
        },
        {
            "RunId": 1,
            "iteration": 1,
            "AgentID": 1,
            "agent_id": 1,
            **template,
        },
        {
            "RunId": 1,
            "iteration": 1,
            "AgentID": 2,
            "agent_id": 2,
            **template,
        },
    ]


def test_iterations_deprecation_warning():
    """Test that using iterations parameter raises DeprecationWarning."""
    with pytest.warns(DeprecationWarning, match="iterations.*deprecated.*rng"):
        mesa.batch_run(MockModel, {}, number_processes=1, iterations=1)
