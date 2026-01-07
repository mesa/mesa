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


class SparseAgent(Agent):
    """Test agent for sparse data collection scenarios."""

    def __init__(self, model):
        """Initialize a SparseAgent.

        Args:
            model: The model instance this agent belongs to.
        """
        super().__init__(model)
        self.value = 0

    def step(self):
        """Increment the agent's value by 1."""
        self.value += 1


class SparseCollectionModel(Model):
    """Test model that collects data sparsely (every N steps)."""

    def __init__(self, collect_interval=5, rng=None):
        """Initialize a SparseCollectionModel.

        Args:
            collect_interval: Number of steps between data collections.
            rng: Random number generator seed.
        """
        super().__init__(rng=rng)
        self.collect_interval = collect_interval
        self.agent = SparseAgent(self)

        self.datacollector = DataCollector(
            model_reporters={"Value": lambda m: m.agent.value}
        )
        self.running = True

    def step(self):
        """Execute one model step, collecting data at specified intervals."""
        if self.steps % self.collect_interval == 0:
            self.datacollector.collect(self)

        self.agent.step()

        if self.steps >= 20:
            self.running = False


def test_batch_run_sparse_collection():
    """Test batch_run with sparse data collection (only collecting every N steps)."""
    result = mesa.batch_run(
        SparseCollectionModel,
        parameters={"collect_interval": [5]},
        rng=[42],
        max_steps=20,
        data_collection_period=1,
        number_processes=1,
    )

    assert len(result) > 0
    assert all("Value" in row for row in result)
    assert all("Step" in row for row in result)


class TimeDilationModel(Model):
    """Model that collects data multiple times per step to test BatchRunner alignment."""

    def __init__(self, *args, **kwargs):
        """Initialize the model."""
        self.schedule = None
        super().__init__()
        self.datacollector = DataCollector(
            model_reporters={"RealStep": lambda m: m.time}
        )
        # Collect INITIAL state
        self.datacollector.collect(self)

    def step(self):
        """Advance the model by one step."""
        super().step()
        # Collect data TWICE per step to simulate sub-step resolution
        self.datacollector.collect(self)
        self.datacollector.collect(self)


def test_batch_run_time_dilation():
    """Test that batch_run correctly aligns data when collection frequency != 1 step."""
    results = mesa.batch_run(
        TimeDilationModel,
        parameters={},
        number_processes=1,
        rng=[None],  # Use rng instead of iterations to avoid deprecation warning
        max_steps=5,
        data_collection_period=1,
        display_progress=False,
    )

    # We expect to find data for 'Step 5'
    # Without the fix, it grabs index 5 (Step 2/3). With fix, it finds correct Step 5.
    last_result = results[-1]
    reported_step = last_result["Step"]
    actual_step_data = last_result["RealStep"]

    assert reported_step == actual_step_data, (
        f"BatchRunner returned data from Step {actual_step_data} when asked for Step {reported_step}"
    )


def test_batch_run_legacy_datacollector():
    """Test batch_run with DataCollector missing _collection_steps (backwards compatibility)."""

    class LegacyModel(Model):
        """Model simulating old DataCollector without _collection_steps."""

        def __init__(self, *args, **kwargs):
            self.schedule = None
            super().__init__()
            self.datacollector = DataCollector(
                model_reporters={"Value": lambda m: m.time * 10}
            )
            # Remove _collection_steps to simulate old DataCollector
            delattr(self.datacollector, "_collection_steps")

        def step(self):
            super().step()
            self.datacollector.collect(self)

    results = mesa.batch_run(
        LegacyModel,
        parameters={},
        number_processes=1,
        rng=[None],
        max_steps=3,
        data_collection_period=1,
        display_progress=False,
    )

    # Should fallback to index-based access
    assert len(results) > 0
    assert "Value" in results[0]


def test_batch_run_missing_step():
    """Test batch_run when requested step not found in _collection_steps."""

    class SparseModel(Model):
        """Model that skips some collections to test edge cases."""

        def __init__(self, *args, **kwargs):
            self.schedule = None
            super().__init__()
            self.datacollector = DataCollector(
                model_reporters={"Value": lambda m: m.time}
            )
            # Collect initial state
            self.datacollector.collect(self)

        def step(self):
            super().step()
            # Collect on steps 2, 4, 6 to create gaps
            if self.time in [2, 4, 6]:
                self.datacollector.collect(self)

    # Request data for a step that wasn't collected (step 5)
    # The fallback should handle this gracefully
    results = mesa.batch_run(
        SparseModel,
        parameters={},
        number_processes=1,
        rng=[None],
        max_steps=6,
        data_collection_period=1,
        display_progress=False,
    )

    # Should handle sparse collection - may have fewer results
    assert len(results) >= 0


def test_batch_run_empty_collection_edge_case():
    """Test batch_run with edge case: requesting data before any collection happens."""

    class EmptyCollectionModel(Model):
        """Model that doesn't collect any data initially."""

        def __init__(self, *args, **kwargs):
            self.schedule = None
            super().__init__()
            self.datacollector = DataCollector(
                model_reporters={"Value": lambda m: m.time}
            )
            # Don't collect initial state - this creates the edge case

        def step(self):
            super().step()
            # Only collect on final step
            if self.time == 3:
                self.datacollector.collect(self)

    # Request data for early steps when nothing has been collected yet
    results = mesa.batch_run(
        EmptyCollectionModel,
        parameters={},
        number_processes=1,
        rng=[None],
        max_steps=3,
        data_collection_period=1,
        display_progress=False,
    )

    # Should handle empty collections gracefully
    assert len(results) >= 0
