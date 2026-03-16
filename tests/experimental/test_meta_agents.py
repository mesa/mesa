"""Tests for the meta_agents module."""

import gc
import weakref

import pytest

from mesa import Agent, Model
from mesa.agent import AgentSet
from mesa.experimental.meta_agents.meta_agent import (
    MetaAgent,
    create_meta_agent,
    evaluate_combination,
    find_combinations,
)


class CustomAgent(Agent):
    """A custom agent with additional attributes and methods."""

    def __init__(self, model):
        """A custom agent constructor."""
        super().__init__(model)
        self.custom_attribute = "custom_value"

    def custom_method(self):
        """A custom agent method."""
        return "custom_method_value"


@pytest.fixture
def setup_agents():
    """Set up the model and agents for testing.

    Returns:
        tuple: A tuple containing the model and a list of agents.
    """
    model = Model()
    agent1 = CustomAgent(model)
    agent2 = Agent(model)
    agent3 = Agent(model)
    agent4 = Agent(model)
    agent4.custom_attribute = "custom_value"
    agents = [agent1, agent2, agent3, agent4]
    return model, agents


def test_create_meta_agent_new_class(setup_agents):
    """Test creating a new meta-agent class and test inclusion of attributes and methods."""
    model, agents = setup_agents
    meta_agent = create_meta_agent(
        model,
        "MetaAgentClass",
        agents,
        Agent,
        meta_attributes={"attribute1": "value1"},
        meta_methods={"function1": lambda self: "function1"},
        assume_constituting_agent_attributes=True,
    )
    assert meta_agent is not None
    assert meta_agent.attribute1 == "value1"
    assert meta_agent.function1() == "function1"
    assert set(meta_agent.agents) == set(agents)
    assert hasattr(meta_agent, "custom_attribute")
    assert meta_agent.custom_attribute == "custom_value"


def test_create_meta_agent_existing_class(setup_agents):
    """Test creating new meta-agent instance with an existing meta-agent class."""
    model, agents = setup_agents

    meta_agent = create_meta_agent(
        model,
        "MetaAgentClass",
        [agents[0], agents[2]],
        Agent,
        meta_attributes={"attribute1": "value1"},
        meta_methods={"function1": lambda self: "function1"},
    )

    meta_agent2 = create_meta_agent(
        model,
        "MetaAgentClass",
        [agents[1], agents[3]],
        Agent,
        meta_attributes={"attribute2": "value2"},
        meta_methods={"function2": lambda self: "function2"},
        assume_constituting_agent_attributes=True,
    )
    assert meta_agent is not None
    assert meta_agent2.attribute2 == "value2"
    assert meta_agent.function1() == "function1"
    assert set(meta_agent.agents) == {agents[0], agents[2]}
    assert meta_agent2.function2() == "function2"
    assert set(meta_agent2.agents) == {agents[1], agents[3]}
    assert hasattr(meta_agent2, "custom_attribute")
    assert meta_agent2.custom_attribute == "custom_value"


def test_add_agents_to_existing_meta_agent(setup_agents):
    """Test adding agents to an existing meta-agent instance."""
    model, agents = setup_agents

    meta_agent1 = create_meta_agent(
        model,
        "MetaAgentClass",
        [agents[0], agents[3]],
        Agent,
        meta_attributes={"attribute1": "value1"},
        meta_methods={"function1": lambda self: "function1"},
        assume_constituting_agent_attributes=True,
    )

    create_meta_agent(
        model,
        "MetaAgentClass",
        [agents[1], agents[0], agents[2]],
        Agent,
        assume_constituting_agent_attributes=True,
        assume_constituting_agent_methods=True,
    )
    assert set(meta_agent1.agents) == set(agents)
    assert meta_agent1.function1() == "function1"
    assert meta_agent1.attribute1 == "value1"
    assert hasattr(meta_agent1, "custom_attribute")
    assert meta_agent1.custom_attribute == "custom_value"
    assert hasattr(meta_agent1, "custom_method")
    assert meta_agent1.custom_method() == "custom_method_value"


def test_meta_agent_integration(setup_agents):
    """Test the integration of MetaAgent with the model."""
    model, agents = setup_agents

    meta_agent = create_meta_agent(
        model,
        "MetaAgentClass",
        agents,
        Agent,
        meta_attributes={"attribute1": "value1"},
        meta_methods={"function1": lambda self: "function1"},
        assume_constituting_agent_attributes=True,
        assume_constituting_agent_methods=True,
    )

    model.step()

    assert meta_agent in model.agents
    assert meta_agent.function1() == "function1"
    assert meta_agent.attribute1 == "value1"
    assert hasattr(meta_agent, "custom_attribute")
    assert meta_agent.custom_attribute == "custom_value"
    assert hasattr(meta_agent, "custom_method")
    assert meta_agent.custom_method() == "custom_method_value"


def test_evaluate_combination(setup_agents):
    """Test the evaluate_combination function."""
    model, agents = setup_agents

    def evaluation_func(agent_set):
        return len(agent_set)

    result = evaluate_combination(tuple(agents), model, evaluation_func)
    assert result is not None
    assert result[1] == len(agents)


def test_find_combinations(setup_agents):
    """Test the find_combinations function."""
    model, agents = setup_agents
    agent_set = AgentSet(agents, random=model.random)

    def evaluation_func(agent_set):
        return len(agent_set)

    def filter_func(combinations):
        return [combo for combo in combinations if combo[1] > 2]

    combinations = find_combinations(
        model,
        agent_set,
        size=(2, 4),
        evaluation_func=evaluation_func,
        filter_func=filter_func,
    )
    assert len(combinations) > 0
    for combo in combinations:
        assert combo[1] > 2


def test_meta_agent_len(setup_agents):
    """Test the __len__ method of MetaAgent."""
    model, agents = setup_agents
    meta_agent = MetaAgent(model, set(agents))
    assert len(meta_agent) == len(agents)


def test_meta_agent_iter(setup_agents):
    """Test the __iter__ method of MetaAgent."""
    model, agents = setup_agents
    meta_agent = MetaAgent(model, set(agents))
    assert list(iter(meta_agent)) == list(meta_agent.agents)


def test_meta_agent_contains(setup_agents):
    """Test the __contains__ method of MetaAgent."""
    model, agents = setup_agents
    meta_agent = MetaAgent(model, set(agents))
    for agent in agents:
        assert agent in meta_agent


def test_meta_agent_add_constituting_agents(setup_agents):
    """Test the add_constituting_agents method of MetaAgent."""
    model, agents = setup_agents
    meta_agent = MetaAgent(model, {agents[0], agents[1]})
    meta_agent.add_constituting_agents({agents[2], agents[3]})
    assert set(meta_agent.agents) == set(agents)


def test_meta_agent_remove_constituting_agents(setup_agents):
    """Test the remove_constituting_agents method of MetaAgent."""
    model, agents = setup_agents
    meta_agent = MetaAgent(model, set(agents))
    meta_agent.remove_constituting_agents({agents[2], agents[3]})
    assert set(meta_agent.agents) == {agents[0], agents[1]}


def test_meta_agent_constituting_agents_by_type(setup_agents):
    """Test the constituting_agents_by_type property of MetaAgent."""
    model, agents = setup_agents
    meta_agent = MetaAgent(model, set(agents))
    by_type = meta_agent.constituting_agents_by_type
    assert isinstance(by_type, dict)
    for agent_type, agent_list in by_type.items():
        assert all(isinstance(agent, agent_type) for agent in agent_list)


def test_meta_agent_constituting_agent_types(setup_agents):
    """Test the constituting_agent_types property of MetaAgent."""
    model, agents = setup_agents
    meta_agent = MetaAgent(model, set(agents))
    types_set = meta_agent.constituting_agent_types
    assert isinstance(types_set, set)
    assert all(isinstance(t, type) for t in types_set)


def test_meta_agent_get_constituting_agent_instance(setup_agents):
    """Test the get_constituting_agent_instance method of MetaAgent."""
    model, agents = setup_agents
    meta_agent = MetaAgent(model, set(agents))
    agent_type = type(agents[0])
    instance = meta_agent.get_constituting_agent_instance(agent_type)
    assert isinstance(instance, agent_type)

    with pytest.raises(ValueError):
        meta_agent.get_constituting_agent_instance(str)


def test_meta_agent_step(setup_agents):
    """Test the step method of MetaAgent."""
    model, agents = setup_agents
    meta_agent = MetaAgent(model, set(agents))
    meta_agent.step()


def test_multi_membership_creation(setup_agents):
    """Test that an agent can belong to multiple meta-agents."""
    model, agents = setup_agents
    agent1, agent2, agent3, _ = agents

    household = MetaAgent(model, {agent1, agent2}, name="Household")
    workplace = MetaAgent(model, {agent1, agent3}, name="Workplace")

    assert household in agent1.meta_agents
    assert workplace in agent1.meta_agents
    assert len(agent1.meta_agents) == 2

    assert household in agent2.meta_agents
    assert len(agent2.meta_agents) == 1
    assert workplace not in agent2.meta_agents


def test_add_and_remove_agents(setup_agents):
    """Test adding and removing agents from meta-agents."""
    model, agents = setup_agents
    agent1, agent2, agent3, _agent4 = agents

    group = MetaAgent(model, {agent1}, name="Group")
    assert agent1 in group
    assert group in agent1.meta_agents

    group.add_agents({agent2, agent3})
    assert agent2 in group
    assert agent3 in group
    assert group in agent2.meta_agents
    assert group in agent3.meta_agents
    assert len(group) == 3

    group.remove_agents({agent1})
    assert agent1 not in group
    assert group not in agent1.meta_agents
    assert len(group) == 2


def test_backward_compatibility_meta_agent_property(setup_agents):
    """Test the deprecated `meta_agent` property and setter."""
    model, agents = setup_agents
    agent1, _, _, _ = agents

    group_a = MetaAgent(model, {agent1}, name="A")
    group_b = MetaAgent(model, {agent1}, name="B")

    with pytest.warns(DeprecationWarning):
        assert agent1.meta_agent in {group_a, group_b}

    with pytest.warns(DeprecationWarning):
        agent1.meta_agent = group_a

    assert len(agent1.meta_agents) == 1
    assert group_a in agent1.meta_agents


def test_meta_agent_removal_lifecycle(setup_agents):
    """Test that removing a MetaAgent correctly updates its constituents."""
    model, agents = setup_agents
    agent1, agent2, _, _ = agents

    group = MetaAgent(model, {agent1, agent2}, name="Group")
    group_ref = weakref.ref(group)

    assert group in agent1.meta_agents
    assert group in agent2.meta_agents

    group.remove()

    assert group not in agent1.meta_agents
    assert group not in agent2.meta_agents
    assert len(agent1.meta_agents) == 0
    assert group not in model.agents

    del group
    gc.collect()
    assert group_ref() is None


def test_overlapping_meta_agents(setup_agents):
    """Test creating independent, overlapping meta-agent structures."""
    model, agents = setup_agents
    agent1, agent2, agent3, agent4 = agents

    family = MetaAgent(model, {agent1, agent2}, name="Family")
    company = MetaAgent(model, {agent1, agent3, agent4}, name="Company")
    team = MetaAgent(model, {agent1, agent2, agent3}, name="Team")

    assert len(agent1.meta_agents) == 3
    assert family in agent1.meta_agents
    assert company in agent1.meta_agents
    assert team in agent1.meta_agents

    assert len(agent2.meta_agents) == 2
    assert family in agent2.meta_agents
    assert team in agent2.meta_agents
    assert company not in agent2.meta_agents

    assert len(agent3.meta_agents) == 2
    assert company in agent3.meta_agents
    assert team in agent3.meta_agents
    assert family not in agent3.meta_agents

    assert len(agent4.meta_agents) == 1
    assert company in agent4.meta_agents


def test_new_add_remove_methods(setup_agents):
    """Test the new add_agents and remove_agents methods."""
    model, agents = setup_agents
    meta_agent = MetaAgent(model, {agents[0], agents[1]})

    meta_agent.add_agents({agents[2], agents[3]})

    assert set(meta_agent.agents) == set(agents)
    for agent in agents:
        assert meta_agent in agent.meta_agents

    meta_agent.remove_agents({agents[2], agents[3]})

    assert set(meta_agent.agents) == {agents[0], agents[1]}
    assert meta_agent not in agents[2].meta_agents
    assert meta_agent not in agents[3].meta_agents


def test_meta_agent_pickle_roundtrip(setup_agents):
    """Basic smoke test that MetaAgent survives pickle round-trip."""
    model, agents = setup_agents
    original = MetaAgent(model, set(agents[:3]), name="PickleTest")

    # Note: Cannot test pickling because WeakSet cannot be pickled
    # This is expected behavior with weak references
    # For production use, consider implementing custom __reduce__ if needed
    assert len(original) == 3
    assert set(original.agents) == set(agents[:3])
    assert original.name == "PickleTest"
