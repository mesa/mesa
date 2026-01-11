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


def test_batch_run_legacy():
    """Covers the legacy fallback."""

    class LegacyModel(Model):
        def __init__(self, *args, **kwargs):
            self.schedule = None
            super().__init__()
            self.datacollector = DataCollector(
                model_reporters={"Step": lambda m: m.steps},
                agent_reporters={"Dummy": lambda a: 1},
            )
            # FORCE LEGACY: Delete _collection_steps attribute manually
            delattr(self.datacollector, "_collection_steps")

            # Ensure there is at least one agent
            MockAgent(self, 1)

        def step(self):
            super().step()
            self.datacollector.collect(self)

    # Logic to hit the line:
    # Max steps = 6 (Indices: 0, 1, 2, 3, 4, 5)
    # Period = 2
    # range(0, 6, 2) generates -> [0, 2, 4]
    # The last model step is 5.
    # steps[-1] (4) != model.steps-1 (5).
    # This forces the code to execute: steps.append(5)
    results = mesa.batch_run(
        LegacyModel,
        parameters={},
        number_processes=1,
        rng=[None],
        max_steps=6,
        data_collection_period=2,
        display_progress=False,
    )

    steps_captured = [r["Step"] for r in results]

    # Expect [1, 2, 4, 5] instead of [0, 2, 4, 5]
    # Why? Step 0 was not collected. Legacy fallback logic defaults to
    # the first available step (Step 1) when the requested Step (0) is missing.
    assert steps_captured == [1, 2, 4, 5]
    # Ensure last step is present
    assert 5 in steps_captured


def test_batch_run_coverage_cases():
    """Covers all the cases related to data_collection_period.

    - case -1: Only collect at the end of the run.
    - case 1: Collect every step.
    - case _: Collect every N steps (default).
    """
    # Cover 'case -1:' (End Only)
    results_case_end = mesa.batch_run(
        MockModel,
        parameters={},
        number_processes=1,
        rng=[None],
        max_steps=5,
        data_collection_period=-1,  # Triggers 'case -1:'
        display_progress=False,
    )

    # Use set() to deduplicate agent rows
    captured_steps_end = sorted({r["Step"] for r in results_case_end})

    assert captured_steps_end == [5]

    # Cover 'case 1:' (Every Step)
    results_case_1 = mesa.batch_run(
        MockModel,
        parameters={},
        number_processes=1,
        rng=[None],
        max_steps=5,
        data_collection_period=1,  # Triggers 'case 1:'
        display_progress=False,
    )

    assert len(results_case_1) > 0

    assert results_case_1[0]["Step"] == 1

    # Cover 'case _:' (Default)
    results_case_default = mesa.batch_run(
        MockModel,
        parameters={},
        number_processes=1,
        rng=[None],
        max_steps=5,
        data_collection_period=2,  # Triggers 'case _:'
        display_progress=False,
    )

    # Use set() to deduplicate because MockModel returns 3 rows per step (one per agent)
    captured_steps = sorted({r["Step"] for r in results_case_default})

    # Start at 1, step by 2 -> [1, 3, 5]
    assert captured_steps == [1, 3, 5]


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

    # Verify no "ghost" steps were created.
    # Expected steps: 5, 10, 15, 20 (Total 4 rows).
    assert len(result) == 4, f"Expected 4 rows for sparse collection, got {len(result)}"
    steps_captured = sorted([row["Step"] for row in result])
    assert steps_captured == [5, 10, 15, 20]


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

    # Verify ALL sub-step data is captured
    # 1 Initial + (2 collections per step * 5 steps) = 11 rows total
    assert len(results) == 11, f"Data Loss! Expected 11 rows, got {len(results)}"

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

    # Init(1) + Steps 2,4,6(3) = 4 total rows
    # Step 5 should NOT be present.
    assert len(results) == 4, f"Expected 4 rows, got {len(results)}"
    steps_found = [r["Step"] for r in results]
    assert 5 not in steps_found, "Ghost data found for Step 5"


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
    # Only one collection occurred at step 3.
    assert len(results) == 1, f"Expected 1 row, got {len(results)}"
    assert results[0]["Step"] == 3


def test_batch_run_agenttype_reporters():
    """Test batch_run with agenttype_reporters to ensure agent type data is collected."""

    class TypedAgent(Agent):
        """Agent with a value attribute for testing."""

        def __init__(self, model, agent_value):
            super().__init__(model)
            self.agent_value = agent_value

        def step(self):
            self.agent_value += 1

    class AgenttypeModel(Model):
        """Model with agenttype_reporters."""

        def __init__(self, n_agents=5, seed=None):
            super().__init__(seed=seed)
            self.n_agents = n_agents
            self.datacollector = DataCollector(
                model_reporters={"total_agents": lambda m: len(m.agents)},
                agenttype_reporters={TypedAgent: {"value": "agent_value"}},
            )
            for i in range(n_agents):
                TypedAgent(self, agent_value=i)

        def step(self):
            self.agents.do("step")
            self.datacollector.collect(self)

    results = mesa.batch_run(
        AgenttypeModel,
        parameters={"n_agents": [3, 5]},
        number_processes=1,
        rng=[None],
        max_steps=10,
        data_collection_period=-1,
        display_progress=False,
    )

    # Verify results structure
    assert len(results) > 0, "No results returned from batch_run"

    # Check that we have data from agenttype_reporters
    result_keys = results[0].keys()
    assert "value" in result_keys, "agenttype_reporters data not collected"
    assert "AgentID" in result_keys, "AgentID not in results"
    assert "total_agents" in result_keys, "model_reporters data not collected"

    # Verify we have the right number of rows (one per agent per run)
    # 2 parameter combinations * 1 iteration * (3 + 5) agents = 8 rows
    assert len(results) == 8, f"Expected 8 rows, got {len(results)}"

    for result in results:
        assert "value" in result, "Missing 'value' field in result"
        assert result["value"] >= 0, "Invalid agent value"


def test_batch_run_agenttype_and_agent_reporters():
    """Test batch_run with both agent_reporters and agenttype_reporters."""

    class MixedAgent(Agent):
        """Agent for testing mixed reporters."""

        def __init__(self, model, wealth):
            super().__init__(model)
            self.wealth = wealth
            self.steps = 0

        def step(self):
            self.wealth += 1
            self.steps += 1

    class MixedReportersModel(Model):
        """Model with both agent_reporters and agenttype_reporters."""

        def __init__(self, n_agents=3, seed=None):
            super().__init__(seed=seed)
            self.n_agents = n_agents
            self.datacollector = DataCollector(
                model_reporters={"agent_count": lambda m: len(m.agents)},
                agent_reporters={"wealth": "wealth"},
                agenttype_reporters={MixedAgent: {"type_steps": "steps"}},
            )
            for i in range(n_agents):
                MixedAgent(self, wealth=i * 10)

        def step(self):
            self.agents.do("step")
            self.datacollector.collect(self)

    results = mesa.batch_run(
        MixedReportersModel,
        parameters={"n_agents": [2]},
        number_processes=1,
        rng=[None],
        max_steps=5,
        data_collection_period=-1,
        display_progress=False,
    )

    # When both reporters are used, we should have data from both
    # Each agent appears twice: once for agent_reporters, once for agenttype_reporters
    # 1 param combo * 1 iteration * 2 agents * 2 (agent + agenttype) = 4 rows
    assert len(results) == 4, f"Expected 4 rows, got {len(results)}"

    wealth_count = sum(1 for r in results if "wealth" in r and r["wealth"] is not None)
    type_steps_count = sum(
        1 for r in results if "type_steps" in r and r["type_steps"] is not None
    )

    assert wealth_count > 0, "agent_reporters data not collected"
    assert type_steps_count > 0, "agenttype_reporters data not collected"
