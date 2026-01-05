"""Test the DataCollector."""

import unittest

import pandas as pd

from mesa import Agent, Model
from mesa.datacollection import DataCollector


class MockAgent(Agent):
    """Minimalistic agent for testing purposes."""

    def __init__(self, model, val=0):  # noqa: D107
        super().__init__(model)
        self.val = val
        self.val2 = val

    def step(self):  # D103
        """Increment vals by 1."""
        self.val += 1
        self.val2 += 1

    def double_val(self):  # noqa: D102
        return self.val * 2

    def write_final_values(self):  # D103
        """Write the final value to the appropriate table."""
        row = {"agent_id": self.unique_id, "final_value": self.val}
        self.model.datacollector.add_table_row("Final_Values", row)


class MockAgentA(MockAgent):
    """Agent subclass A for testing agent-type-specific reporters."""

    def __init__(self, model, val=0):  # noqa: D107
        super().__init__(model, val)
        self.type_a_val = val * 2

    def step(self):  # noqa: D102
        super().step()
        self.type_a_val = self.val * 2


class MockAgentB(MockAgent):
    """Agent subclass B for testing agent-type-specific reporters."""

    def __init__(self, model, val=0):  # noqa: D107
        super().__init__(model, val)
        self.type_b_val = val * 3

    def step(self):  # noqa: D102
        super().step()
        self.type_b_val = self.val * 3


def agent_function_with_params(agent, multiplier, offset):  # noqa: D103
    return (agent.val * multiplier) + offset


class MockModel(Model):
    """Minimalistic model for testing purposes."""

    def __init__(self):  # noqa: D107
        super().__init__()
        self.model_val = 100

        self.n = 10
        for i in range(1, self.n + 1):
            MockAgent(self, val=i)
        self.datacollector = DataCollector(
            model_reporters={
                "total_agents": lambda m: len(m.agents),
                "model_value": "model_val",
                "model_calc_comp": [self.test_model_calc_comp, [3, 4]],
                "model_calc_fail": [self.test_model_calc_comp, [12, 0]],
            },
            agent_reporters={
                "value": lambda a: a.val,
                "value2": "val2",
                "double_value": MockAgent.double_val,
                "value_with_params": [agent_function_with_params, [2, 3]],
            },
            tables={"Final_Values": ["agent_id", "final_value"]},
        )

    def test_model_calc_comp(self, input1, input2):  # noqa: D102
        if input2 > 0:
            return (self.model_val * input1) / input2
        else:
            assert ValueError
            return None

    def step(self):  # noqa: D102
        self.agents.do("step")
        self.datacollector.collect(self)


class MockModelWithAgentTypes(Model):
    """Model for testing agent-type-specific reporters."""

    def __init__(self):  # noqa: D107
        super().__init__()
        self.model_val = 100

        for i in range(10):
            if i % 2 == 0:
                MockAgentA(self, val=i)
            else:
                MockAgentB(self, val=i)

        self.datacollector = DataCollector(
            model_reporters={"total_agents": lambda m: len(m.agents)},
            agent_reporters={"value": lambda a: a.val},
            agenttype_reporters={
                MockAgentA: {"type_a_val": lambda a: a.type_a_val},
                MockAgentB: {"type_b_val": lambda a: a.type_b_val},
            },
        )

    def step(self):  # noqa: D102
        self.agents.do("step")
        self.datacollector.collect(self)


class TestDataCollector(unittest.TestCase):
    """Tests for DataCollector."""

    def setUp(self):
        """Create the model and run it a set number of steps."""
        self.model = MockModel()
        self.model.datacollector.collect(self.model)
        for i in range(7):
            if i == 4:
                self.model.agents[3].remove()
            self.model.step()

        # Write to table:
        for agent in self.model.agents:
            agent.write_final_values()

    def step_assertion(self, model_var):  # noqa: D102
        for element in model_var:
            if model_var.index(element) < 4:
                assert element == 10
            else:
                assert element == 9

    def test_model_vars(self):
        """Test model-level variable collection."""
        data_collector = self.model.datacollector
        assert "total_agents" in data_collector.model_vars
        assert "model_value" in data_collector.model_vars
        assert "model_calc_comp" in data_collector.model_vars
        assert "model_calc_fail" in data_collector.model_vars
        length = 8
        assert len(data_collector.model_vars["total_agents"]) == length
        assert len(data_collector.model_vars["model_value"]) == length
        assert len(data_collector.model_vars["model_calc_comp"]) == length
        self.step_assertion(data_collector.model_vars["total_agents"])
        for element in data_collector.model_vars["model_value"]:
            assert element == 100
        for element in data_collector.model_vars["model_calc_comp"]:
            assert element == 75
        for element in data_collector.model_vars["model_calc_fail"]:
            assert element is None

    def test_agent_records(self):
        """Test agent-level variable collection."""
        data_collector = self.model.datacollector
        agent_table = data_collector.get_agent_vars_dataframe()

        assert "double_value" in list(agent_table.columns)
        assert "value_with_params" in list(agent_table.columns)

        # Check the double_value column
        for (step, agent_id), value in agent_table["double_value"].items():
            expected_value = (step + agent_id) * 2
            self.assertEqual(value, expected_value)

        # Check the value_with_params column
        for (step, agent_id), value in agent_table["value_with_params"].items():
            expected_value = ((step + agent_id) * 2) + 3
            self.assertEqual(value, expected_value)

        assert len(data_collector._agent_records) == 8
        for step, records in data_collector._agent_records.items():
            if step < 5:
                assert len(records) == 10
            else:
                assert len(records) == 9

            for values in records:
                assert len(values) == 6

        assert "value" in list(agent_table.columns)
        assert "value2" in list(agent_table.columns)
        assert "value3" not in list(agent_table.columns)

        with self.assertRaises(KeyError):
            data_collector._agent_records[8]

    def test_table_rows(self):
        """Test table collection."""
        data_collector = self.model.datacollector
        assert len(data_collector.tables["Final_Values"]) == 2
        assert "agent_id" in data_collector.tables["Final_Values"]
        assert "final_value" in data_collector.tables["Final_Values"]
        for _key, data in data_collector.tables["Final_Values"].items():
            assert len(data) == 9

        with self.assertRaises(Exception):
            data_collector.add_table_row("error_table", {})

        with self.assertRaises(Exception):
            data_collector.add_table_row("Final_Values", {"final_value": 10})

    def test_table_ignore_missing(self):
        """Test table collection with ignore_missing=True."""
        data_collector = self.model.datacollector
        # ["agent_id", "final_value"]

        row = {"agent_id": 999}
        data_collector.add_table_row("Final_Values", row, ignore_missing=True)

        table_df = data_collector.get_table_dataframe("Final_Values")
        # the last row should have 999 and None
        last_row = table_df.iloc[-1]
        self.assertEqual(last_row["agent_id"], 999)

        self.assertTrue(pd.isna(last_row["final_value"]))

    def test_exports(self):
        """Test DataFrame exports."""
        data_collector = self.model.datacollector
        model_vars = data_collector.get_model_vars_dataframe()
        agent_vars = data_collector.get_agent_vars_dataframe()
        table_df = data_collector.get_table_dataframe("Final_Values")
        assert model_vars.shape == (8, 4)
        assert agent_vars.shape == (77, 4)
        assert table_df.shape == (9, 2)

        with self.assertRaises(Exception):
            table_df = data_collector.get_table_dataframe("not a real table")


class TestDataCollectorWithAgentTypes(unittest.TestCase):
    """Tests for DataCollector with agent-type-specific reporters."""

    def setUp(self):
        """Create the model and run it a set number of steps."""
        self.model = MockModelWithAgentTypes()
        for _ in range(5):
            self.model.step()

    def test_agenttype_vars(self):
        """Test agent-type-specific variable collection."""
        data_collector = self.model.datacollector

        # Test MockAgentA data
        agent_a_data = data_collector.get_agenttype_vars_dataframe(MockAgentA)
        self.assertIn("type_a_val", agent_a_data.columns)
        self.assertEqual(len(agent_a_data), 25)  # 5 agents * 5 steps
        for (step, agent_id), value in agent_a_data["type_a_val"].items():
            expected_value = (agent_id - 1) * 2 + step * 2
            self.assertEqual(value, expected_value)

        # Test MockAgentB data
        agent_b_data = data_collector.get_agenttype_vars_dataframe(MockAgentB)
        self.assertIn("type_b_val", agent_b_data.columns)
        self.assertEqual(len(agent_b_data), 25)  # 5 agents * 5 steps
        for (step, agent_id), value in agent_b_data["type_b_val"].items():
            expected_value = (agent_id - 1) * 3 + step * 3
            self.assertEqual(value, expected_value)

    def test_agenttype_and_agent_vars(self):
        """Test that agent-type-specific and general agent variables are collected correctly."""
        data_collector = self.model.datacollector

        agent_vars = data_collector.get_agent_vars_dataframe()
        agent_a_vars = data_collector.get_agenttype_vars_dataframe(MockAgentA)
        agent_b_vars = data_collector.get_agenttype_vars_dataframe(MockAgentB)

        # Check that general agent variables are present for all agents
        self.assertIn("value", agent_vars.columns)

        # Check that agent-type-specific variables are only present in their respective dataframes
        self.assertIn("type_a_val", agent_a_vars.columns)
        self.assertNotIn("type_a_val", agent_b_vars.columns)
        self.assertIn("type_b_val", agent_b_vars.columns)
        self.assertNotIn("type_b_val", agent_a_vars.columns)

    def test_nonexistent_agenttype(self):
        """Test that requesting data for a non-existent agent type raises a warning."""
        data_collector = self.model.datacollector

        class NonExistentAgent(Agent):
            pass

        with self.assertWarns(UserWarning):
            non_existent_data = data_collector.get_agenttype_vars_dataframe(
                NonExistentAgent
            )
            self.assertTrue(non_existent_data.empty)

    def test_invalid_agent_type_error(self):
        """Test that passing a non-Agent class raises ValueError during collection."""

        class NotAnAgent:
            pass

        dc = DataCollector(agenttype_reporters={NotAnAgent: {"foo": lambda a: 1}})

        with self.assertRaises(ValueError) as cm:
            dc._record_agenttype(self.model, NotAnAgent)

        self.assertIn("not recognized as an Agent type", str(cm.exception))

    def test_agenttype_reporter_string_attribute(self):
        """Test agent-type-specific reporter with string attribute."""
        model = MockModelWithAgentTypes()
        model.datacollector._new_agenttype_reporter(MockAgentA, "string_attr", "val")
        model.step()

        agent_a_data = model.datacollector.get_agenttype_vars_dataframe(MockAgentA)
        self.assertIn("string_attr", agent_a_data.columns)
        for (_step, agent_id), value in agent_a_data["string_attr"].items():
            expected_value = agent_id
            self.assertEqual(value, expected_value)

    def test_agenttype_reporter_function_with_params(self):
        """Test agent-type-specific reporter with function and parameters."""

        def test_func(agent, multiplier):
            return agent.val * multiplier

        model = MockModelWithAgentTypes()
        model.datacollector._new_agenttype_reporter(
            MockAgentB, "func_param", [test_func, [2]]
        )
        model.step()

        agent_b_data = model.datacollector.get_agenttype_vars_dataframe(MockAgentB)
        self.assertIn("func_param", agent_b_data.columns)
        for (_step, agent_id), value in agent_b_data["func_param"].items():
            expected_value = agent_id * 2
            self.assertEqual(value, expected_value)

    def test_agenttype_reporter_multiple_types(self):
        """Test adding reporters for multiple agent types."""
        model = MockModelWithAgentTypes()
        model.datacollector._new_agenttype_reporter(
            MockAgentA, "type_a_val", lambda a: a.type_a_val
        )
        model.datacollector._new_agenttype_reporter(
            MockAgentB, "type_b_val", lambda a: a.type_b_val
        )
        model.step()

        agent_a_data = model.datacollector.get_agenttype_vars_dataframe(MockAgentA)
        agent_b_data = model.datacollector.get_agenttype_vars_dataframe(MockAgentB)

        self.assertIn("type_a_val", agent_a_data.columns)
        self.assertIn("type_b_val", agent_b_data.columns)
        self.assertNotIn("type_b_val", agent_a_data.columns)
        self.assertNotIn("type_a_val", agent_b_data.columns)

    def test_agenttype_superclass_reporter(self):
        """Test adding a reporter for a superclass of an agent type."""
        model = MockModelWithAgentTypes()
        model.datacollector._new_agenttype_reporter(MockAgent, "val", lambda a: a.val)
        model.datacollector._new_agenttype_reporter(Agent, "val", lambda a: a.val)
        for _ in range(3):
            model.step()

        super_data = model.datacollector.get_agenttype_vars_dataframe(MockAgent)
        agent_data = model.datacollector.get_agenttype_vars_dataframe(Agent)
        self.assertIn("val", super_data.columns)
        self.assertIn("val", agent_data.columns)
        self.assertEqual(len(super_data), 30)  # 10 agents * 3 steps
        self.assertEqual(len(agent_data), 30)
        self.assertTrue(super_data.equals(agent_data))


class MockModelForErrors(Model):
    """Test model for error handling."""

    def __init__(self):
        """Initialize the test model for error handling."""
        super().__init__()
        self.num_agents = 10
        self.valid_attribute = "test"

    def valid_method(self):
        """Valid method for testing."""
        return self.num_agents


def helper_function(model, param1):
    """Test function with parameters."""
    return model.num_agents * param1


class TestDataCollectorErrorHandling(unittest.TestCase):
    """Test error handling in DataCollector."""

    def setUp(self):
        """Set up test cases."""
        self.model = MockModelForErrors()

    def test_lambda_error(self):
        """Test error when lambda tries to access non-existent attribute."""
        dc_lambda = DataCollector(
            model_reporters={"bad_lambda": lambda m: m.nonexistent_attr}
        )
        with self.assertRaises(RuntimeError):
            dc_lambda.collect(self.model)

    def test_method_error(self):
        """Test error when accessing non-existent method."""

        def bad_method(model):
            raise RuntimeError("This method is not valid.")

        dc_method = DataCollector(model_reporters={"test": bad_method})

        with self.assertRaises(RuntimeError):
            dc_method.collect(self.model)

    def test_attribute_error(self):
        """Test error when accessing non-existent attribute."""
        dc_attribute = DataCollector(
            model_reporters={"bad_attribute": "nonexistent_attribute"}
        )
        with self.assertRaises(Exception):
            dc_attribute.collect(self.model)

    def test_agent_missing_attribute_error(self):
        """Test that DataCollector raises AttributeError for missing agent attributes.

        This tests the fix for GitHub issue: DataCollector silently skips reporters
        for non-existent attributes. Now it should raise an informative error.
        """
        model = Model()
        agent = Agent(model)
        agent.wealth = 100
        agent.status = "active"
        # Note: agent does NOT have 'health' attribute

        # Create DataCollector with reporter for missing attribute
        dc = DataCollector(
            agent_reporters={
                "wealth": "wealth",  # Exists
                "health": "health",  # Does NOT exist
                "status": "status",  # Exists
            }
        )

        # Should raise AttributeError when trying to collect
        with self.assertRaises(AttributeError) as cm:
            dc.collect(model)

        # Check error message is informative
        error_msg = str(cm.exception)
        self.assertIn("health", error_msg)
        self.assertIn("Agent", error_msg)

    def test_agent_valid_attributes_still_work(self):
        """Test that valid agent attribute reporters still work correctly."""
        model = Model()
        agent = Agent(model)
        agent.wealth = 100
        agent.status = "active"

        # Create DataCollector with only valid reporters
        dc = DataCollector(agent_reporters={"wealth": "wealth", "status": "status"})

        # Should work without errors
        dc.collect(model)

        # Verify data was collected
        self.assertIn(0, dc._agent_records)
        records = dc._agent_records[0]
        self.assertEqual(len(records), 1)

        _step, _agent_id, wealth, status = records[0]
        self.assertEqual(wealth, 100)
        self.assertEqual(status, "active")

    def test_lambda_reporters_still_work(self):
        """Test that lambda and callable reporters still work correctly."""
        model = Model()
        agent = Agent(model)
        agent.wealth = 100

        # Create DataCollector with callable reporters
        dc = DataCollector(
            agent_reporters={
                "wealth": lambda a: a.wealth,
                "doubled": lambda a: a.wealth * 2,
            }
        )

        # Should work without errors
        dc.collect(model)

        # Verify data was collected correctly
        _step, _agent_id, wealth, doubled = dc._agent_records[0][0]
        self.assertEqual(wealth, 100)
        self.assertEqual(doubled, 200)

    def test_function_error(self):
        """Test error when function list is not callable."""
        dc_function = DataCollector(
            model_reporters={"bad_function": ["not_callable", [1, 2]]}
        )
        with self.assertRaises(ValueError):
            dc_function.collect(self.model)


class TestMethodReporterValidation(unittest.TestCase):
    """Tests for method reporter validation fix.

    These tests verify that the fix for method reporter validation in
    DataCollector._validate_model_reporter() works correctly.

    The fix changes:
        BEFORE: if not callable(reporter) and not isinstance(reporter, types.LambdaType):
                    pass

        AFTER:  if callable(reporter) and not isinstance(reporter, types.LambdaType):
                    try:
                        reporter()
                    except Exception as e:
                        raise RuntimeError(...)
    """

    def test_broken_method_reporter_raises_runtime_error(self):
        """Test that a broken method reporter raises RuntimeError with clear message.

        ORIGINAL BUG: The condition `not callable(reporter)` is False for methods,
        so method reporters were never validated. Broken methods would only fail
        during collect(), with an unclear error message.

        FIX: Changed to `callable(reporter)` so methods ARE validated.
        """

        class BrokenModel(Model):
            def __init__(self):
                super().__init__()
                self.datacollector = DataCollector(
                    model_reporters={"Broken": self.broken_method}
                )

            def broken_method(self):
                # This method has a bug - references non-existent attribute
                return self.nonexistent_attr

        model = BrokenModel()

        # Before fix: This would raise AttributeError without context
        # After fix: This raises RuntimeError with reporter name
        with self.assertRaises(RuntimeError) as context:
            model.datacollector.collect(model)

        # Verify error message contains the reporter name for easy debugging
        self.assertIn("Broken", str(context.exception))
        self.assertIn("failed validation", str(context.exception))

    def test_working_method_reporter_succeeds(self):
        """Test that working method reporters continue to work correctly.

        This ensures the fix doesn't break valid use cases.
        """

        class WorkingModel(Model):
            def __init__(self):
                super().__init__()
                self.value = 42
                self.datacollector = DataCollector(
                    model_reporters={"Value": self.get_value}
                )

            def get_value(self):
                return self.value

        model = WorkingModel()
        model.datacollector.collect(model)

        data = model.datacollector.get_model_vars_dataframe()
        self.assertEqual(data["Value"][0], 42)

    def test_method_reporter_exception_is_wrapped(self):
        """Test that exceptions from method reporters are wrapped in RuntimeError.

        This provides better error messages for debugging.
        """

        class ExceptionModel(Model):
            def __init__(self):
                super().__init__()
                self.datacollector = DataCollector(
                    model_reporters={"Raises": self.raising_method}
                )

            def raising_method(self):
                raise ValueError("Something went wrong in the reporter")

        model = ExceptionModel()

        with self.assertRaises(RuntimeError) as context:
            model.datacollector.collect(model)

        # Error message should contain both reporter name and original error
        error_msg = str(context.exception)
        self.assertIn("Raises", error_msg)
        self.assertIn("Something went wrong", error_msg)

    def test_mixed_lambda_and_method_reporters(self):
        """Test that lambdas and method reporters work together correctly.

        Validates that the fix doesn't interfere with lambda reporter handling.
        """

        class MixedModel(Model):
            def __init__(self):
                super().__init__()
                self.value = 10
                self.datacollector = DataCollector(
                    model_reporters={
                        "Lambda": lambda m: m.value,
                        "Method": self.double_value,
                    }
                )

            def double_value(self):
                return self.value * 2

        model = MixedModel()
        model.datacollector.collect(model)

        data = model.datacollector.get_model_vars_dataframe()
        self.assertEqual(data["Lambda"][0], 10)
        self.assertEqual(data["Method"][0], 20)

    def test_method_reporter_called_without_args(self):
        """Test that method reporters are called without arguments.

        Bound methods (self.method) already have 'self' bound, so they should
        be called with reporter() not reporter(model).

        This matches how collect() calls them at line 341:
            self.model_vars[var].append(deepcopy(reporter()))
        """

        class ArgCheckModel(Model):
            def __init__(self):
                super().__init__()
                self.call_count = 0
                self.datacollector = DataCollector(
                    model_reporters={"CallCount": self.count_calls}
                )

            def count_calls(self):
                # This method takes no args besides self (which is already bound)
                self.call_count += 1
                return self.call_count

        model = ArgCheckModel()
        model.datacollector.collect(model)

        # Validation call + actual collect call = at least 1
        self.assertGreaterEqual(model.call_count, 1)


def test_mutable_data_independence():
    """Test that mutable agent data is deep-copied, preventing historical records from changing."""

    class MutableAgent(Agent):
        """Agent with mutable list attribute."""

        def __init__(self, model):
            super().__init__(model)
            self.data = []

    class MutableModel(Model):
        """Model that modifies agent data after collection."""

        def __init__(self):
            super().__init__()
            self.agent = MutableAgent(self)
            self.datacollector = DataCollector(agent_reporters={"Data": "data"})

        def step(self):
            self.datacollector.collect(self)
            self.agent.data.append(self.steps)  # Modify after collection

    model = MutableModel()

    model.step()
    model.step()
    model.step()

    df = model.datacollector.get_agent_vars_dataframe()

    # Each step should preserve its historical state
    assert df.loc[(1, 1), "Data"] == []
    assert df.loc[(2, 1), "Data"] == [1]
    assert df.loc[(3, 1), "Data"] == [1, 2]


if __name__ == "__main__":
    unittest.main()
