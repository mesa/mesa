"""Tests for patch coverage in mesa/datacollection.py."""

import pytest

from mesa.agent import Agent
from mesa.datacollection import DataCollector
from mesa.model import Model


def test_pandas_missing(monkeypatch):
    """Test that methods raise ImportError when pandas is not available."""
    # Monkeypatch sys.modules to hide pandas if it is imported
    # But mesa.datacollection tries to import it at top level.
    # Since it's likely already imported, we need to mess with the module's globals.

    import mesa.datacollection  # noqa: PLC0415

    # We need to simulate 'pd' not being in globals() of mesa.datacollection
    # The module imports pd inside a try/except block or top level.
    # Looking at the file:
    # with contextlib.suppress(ImportError):
    #     import pandas as pd

    # So we can remove it from reference in the module
    monkeypatch.delattr(mesa.datacollection, "pd", raising=False)

    dc = DataCollector()

    with pytest.raises(ImportError, match="pandas not found"):
        dc.get_table_dataframe("any_table")

    with pytest.raises(ImportError, match="pandas not found"):
        dc.get_agenttype_vars_dataframe(Agent)

    with pytest.raises(ImportError, match="pandas not found"):
        dc.get_agent_vars_dataframe()

    with pytest.raises(ImportError, match="pandas not found"):
        dc.get_model_vars_dataframe()


def test_empty_reporters_warning():
    """Test that UserWarning is raised when asking for dataframes with no reporters."""
    # Initialize with no reporters (default is empty dicts inside __init__ if None passed)
    dc = DataCollector(model_reporters={}, agent_reporters={})

    # Ensure pandas is present for this test (it should be, unless previous test messed it up permanently,
    # but monkeypatch undoes changes)

    with pytest.raises(UserWarning, match="No agent reporters have been defined"):
        dc.get_agent_vars_dataframe()

    with pytest.raises(UserWarning, match="No model reporters have been defined"):
        dc.get_model_vars_dataframe()


def test_deepcopy_vs_immutable():
    """Test that mutable values are deepcopied and immutable ones are not."""

    class MockAgent(Agent):
        def __init__(self, model):
            super().__init__(model)
            self.my_list = [1, 2]
            self.my_int = 10

    model = Model()
    agent = MockAgent(model)
    # model.agents is auto-populated by register_agent inside Agent.__init__
    # But usually one shouldn't rely on model.agents if we are manually constructing things?
    # Actually register_agent adds it to model._agents and model._all_agents
    # datacollection uses model.agents (which is _all_agents property)

    model.time = 0

    # Reporter for mutable (list) and immutable (int)
    agent_reporters = {"mutable": lambda a: a.my_list, "immutable": lambda a: a.my_int}

    dc = DataCollector(agent_reporters=agent_reporters)

    # First collection
    dc.collect(model)

    # Modify the agent's mutable state
    agent.my_list.append(3)
    agent.my_int = 20
    model.time = 1

    # Second collection
    dc.collect(model)

    # Verify the collected data
    # Structure of _agent_records: {step: [(time, unique_id, val1, val2), ...]}
    # But here collect() appends to _agent_records as a dict?
    # Check code: self._agent_records[model.time] = list(agent_records)

    # Step 0
    records_0 = dc._agent_records[0]
    # format: (time, unique_id, mutable_val, immutable_val)
    # The order depends on dictionary iteration order, but insertion order is preserved in modern python.
    # We defined mutable first, then immutable in the dict above.

    entry_0 = records_0[0]
    # Check that the list stored matches what it was at time 0
    assert entry_0[2] == [1, 2]
    # Check that it is NOT the same object as the current agent list (which is [1, 2, 3])
    assert entry_0[2] is not agent.my_list

    # Step 1
    records_1 = dc._agent_records[1]
    entry_1 = records_1[0]
    assert entry_1[2] == [1, 2, 3]

    # Check immutable
    assert entry_0[3] == 10
    assert entry_1[3] == 20


def test_missing_agent_attribute_error():
    """Test that collecting a missing attribute raises specific AttributeError."""

    class SimpleAgent(Agent):
        pass

    model = Model()
    SimpleAgent(model)
    # model.agents = [agent]  # Not needed, auto-registered

    # Reporter asking for non-existent attribute
    dc = DataCollector(agent_reporters={"bad_rep": "missing_attr"})

    with pytest.raises(
        AttributeError,
        match=r"Agent .* of type SimpleAgent has no attribute 'missing_attr' \(reporter: 'bad_rep'\)",
    ):
        dc.collect(model)
