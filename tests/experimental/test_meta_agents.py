"""Tests for the meta_agents module."""

import pytest
from mesa import Agent, Model
from mesa.discrete_space.cell_agent import CellAgent
from mesa.discrete_space.grid import OrthogonalMooreGrid
from mesa.experimental.meta_agents.identity import ensure_entity_index
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
    """Test creating a new meta-agent class and test inclusion of attributes and methods.

    Args:
        setup_agents (tuple): The model and agents fixture.
    """
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
    assert meta_agent.agents == set(agents)
    assert hasattr(meta_agent, "custom_attribute")
    assert meta_agent.custom_attribute == "custom_value"


def test_create_meta_agent_existing_class(setup_agents):
    """Test creating new meta-agent instance with an existing meta-agent class.

    Args:
        setup_agents (tuple): The model and agents fixture.
    """
    model, agents = setup_agents

    # Create Met Agent Class
    meta_agent = create_meta_agent(
        model,
        "MetaAgentClass",
        [agents[0], agents[2]],
        Agent,
        meta_attributes={"attribute1": "value1"},
        meta_methods={"function1": lambda self: "function1"},
    )

    # Create new meta-agent instance with existing class
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
    assert meta_agent.agents == {agents[2], agents[0]}
    assert meta_agent2.function2() == "function2"
    assert meta_agent2.agents == {agents[1], agents[3]}
    assert hasattr(meta_agent2, "custom_attribute")
    assert meta_agent2.custom_attribute == "custom_value"


def test_add_agents_to_existing_meta_agent(setup_agents):
    """Test adding agents to an existing meta-agent instance.

    Args:
        setup_agents (tuple): The model and agents fixture.
    """
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
    assert meta_agent1.agents == {agents[0], agents[1], agents[2], agents[3]}
    assert meta_agent1.function1() == "function1"
    assert meta_agent1.attribute1 == "value1"
    assert hasattr(meta_agent1, "custom_attribute")
    assert meta_agent1.custom_attribute == "custom_value"
    assert hasattr(meta_agent1, "custom_method")
    assert meta_agent1.custom_method() == "custom_method_value"


def test_meta_agent_integration(setup_agents):
    """Test the integration of MetaAgent with the model.

    Args:
        setup_agents (tuple): The model and agents fixture.
    """
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


def test_entity_index_tracks_atomic_and_meta_agents(setup_agents):
    """The explicit entity index should assign stable ids to both entity kinds."""
    model, agents = setup_agents
    meta_agent = MetaAgent(model, set(agents))
    entity_index = ensure_entity_index(model)

    meta_record = entity_index.record_for(meta_agent)
    atomic_records = [entity_index.record_for(agent) for agent in agents]

    assert meta_record.kind == "meta"
    assert meta_record.entity is meta_agent
    assert meta_record.entity_id == meta_agent.entity_id
    assert all(record.kind == "atomic" for record in atomic_records)
    assert all(
        record.entity_id == agent.entity_id
        for record, agent in zip(atomic_records, agents)
    )
    assert entity_index.entity_for(meta_agent.entity_id) is meta_agent
    assert {record.entity for record in entity_index.records()} == {meta_agent, *agents}


def test_entity_index_lookup_survives_membership_updates(setup_agents):
    """Membership churn should not change the explicit entity ids."""
    model, agents = setup_agents
    entity_index = ensure_entity_index(model)

    meta_agent = MetaAgent(model, {agents[0], agents[1]})
    original_meta_id = meta_agent.entity_id
    original_agent_id = agents[0].entity_id

    meta_agent.remove_constituting_agents({agents[0]})
    meta_agent.add_constituting_agents({agents[0], agents[2]})
    meta_agent.unique_id = "renamed-meta-agent"
    entity_index.register(meta_agent, kind="meta")

    assert agents[0].entity_id == original_agent_id
    assert meta_agent.entity_id == original_meta_id
    assert entity_index.entity_for(original_meta_id) is meta_agent
    assert entity_index.entity_for(original_agent_id) is agents[0]
    assert entity_index.kind_for(original_meta_id) == "meta"
    assert entity_index.kind_for(original_agent_id) == "atomic"
    entity_index.assert_invariants()


def test_entity_index_preserves_meta_kind_for_nested_meta_agents():
    """Nested meta-agents should stay classified as meta in the registry."""
    model = Model()
    atomic_agent = Agent(model)
    child_meta = MetaAgent(model, {atomic_agent}, name="Child")
    parent_meta = MetaAgent(model, {child_meta}, name="Parent")
    entity_index = ensure_entity_index(model)

    assert entity_index.kind_for(atomic_agent.entity_id) == "atomic"
    assert entity_index.kind_for(child_meta.entity_id) == "meta"
    assert entity_index.kind_for(parent_meta.entity_id) == "meta"
    assert parent_meta.agents == {child_meta}
    entity_index.assert_invariants()


def test_evaluate_combination(setup_agents):
    """Test the evaluate_combination function.

    Args:
        setup_agents (tuple): The model and agents fixture.
    """
    model, agents = setup_agents

    def evaluation_func(agent_set):
        return len(agent_set)

    result = evaluate_combination(tuple(agents), model, evaluation_func)
    assert result is not None
    assert result[1] == len(agents)


def test_find_combinations(setup_agents):
    """Test the find_combinations function.

    Args:
        setup_agents (tuple): The model and agents fixture.
    """
    model, agents = setup_agents
    agent_set = set(agents)

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


def test_find_combinations_allows_zero_value(setup_agents):
    """Test that zero-valued evaluation results are preserved."""
    model, agents = setup_agents

    def evaluation_func(agent_group):
        return 0.0

    combinations = find_combinations(
        model,
        agents,
        size=2,
        evaluation_func=evaluation_func,
    )

    assert len(combinations) > 0
    assert combinations[0][1] == 0.0


def test_find_combinations_inclusive_tuple_size_bounds(setup_agents):
    """Test that tuple size bounds are inclusive and support equal bounds."""
    model, agents = setup_agents

    def evaluation_func(agent_group):
        return len(agent_group)

    combinations = find_combinations(
        model,
        agents,
        size=(2, 2),
        evaluation_func=evaluation_func,
    )

    assert len(combinations) == 6
    assert all(value == 2 for _, value in combinations)


def test_meta_agent_len(setup_agents):
    """Test the __len__ method of MetaAgent.

    Args:
        setup_agents (tuple): The model and agents fixture.
    """
    model, agents = setup_agents
    meta_agent = MetaAgent(model, set(agents))
    assert len(meta_agent) == len(agents)


def test_meta_agent_iter(setup_agents):
    """Test the __iter__ method of MetaAgent.

    Args:
        setup_agents (tuple): The model and agents fixture.
    """
    model, agents = setup_agents
    meta_agent = MetaAgent(model, set(agents))
    assert list(iter(meta_agent)) == list(meta_agent._constituting_set)


def test_meta_agent_contains(setup_agents):
    """Test the __contains__ method of MetaAgent.

    Args:
        setup_agents (tuple): The model and agents fixture.
    """
    model, agents = setup_agents
    meta_agent = MetaAgent(model, set(agents))
    for agent in agents:
        assert agent in meta_agent


def test_meta_agent_add_constituting_agents(setup_agents):
    """Test the add_constituting_agents method of MetaAgent.

    Args:
        setup_agents (tuple): The model and agents fixture.
    """
    model, agents = setup_agents
    meta_agent = MetaAgent(model, {agents[0], agents[1]})
    meta_agent.add_constituting_agents({agents[2], agents[3]})
    assert meta_agent._constituting_set == set(agents)


def test_meta_agent_remove_constituting_agents(setup_agents):
    """Test the remove_constituting_agents method of MetaAgent.

    Args:
        setup_agents (tuple): The model and agents fixture.
    """
    model, agents = setup_agents
    meta_agent = MetaAgent(model, set(agents))
    meta_agent.remove_constituting_agents({agents[2], agents[3]})
    assert meta_agent._constituting_set == {agents[0], agents[1]}


def test_meta_agent_constituting_agents_by_type(setup_agents):
    """Test the constituting_agents_by_type property of MetaAgent."""
    model, agents = setup_agents
    meta_agent = MetaAgent(model, set(agents))
    constituting_agents_by_type = meta_agent.constituting_agents_by_type
    assert isinstance(constituting_agents_by_type, dict)
    for agent_type, agent_list in constituting_agents_by_type.items():
        assert all(isinstance(agent, agent_type) for agent in agent_list)


def test_meta_agent_constituting_agent_types(setup_agents):
    """Test the constituting_agent_types property of MetaAgent."""
    model, agents = setup_agents
    meta_agent = MetaAgent(model, set(agents))
    constituting_agent_types = meta_agent.constituting_agent_types
    assert isinstance(constituting_agent_types, set)
    assert all(isinstance(agent_type, type) for agent_type in constituting_agent_types)


def test_meta_agent_get_constituting_agent_instance(setup_agents):
    """Test the get_constituting_agent_instance method of MetaAgent."""
    model, agents = setup_agents
    meta_agent = MetaAgent(model, set(agents))
    agent_type = type(agents[0])
    constituting_agent_instance = meta_agent.get_constituting_agent_instance(agent_type)
    assert isinstance(constituting_agent_instance, agent_type)
    with pytest.raises(ValueError):
        meta_agent.get_constituting_agent_instance(str)  # Invalid type


def test_meta_agent_step(setup_agents):
    """Test the step method of MetaAgent."""
    model, agents = setup_agents
    meta_agent = MetaAgent(model, set(agents))
    meta_agent.step()  # Ensure no errors occur during step
    # Add additional assertions if step behavior is defined in the future


def test_explicit_meta_methods_take_precedence_over_inferred(setup_agents):
    """Test explicitly provided meta_methods are not overwritten by inferred ones."""
    model, agents = setup_agents

    meta_agent = create_meta_agent(
        model,
        "MethodPriorityMetaAgent",
        [agents[0]],
        Agent,
        meta_methods={"custom_method": lambda self: "explicit_value"},
        assume_constituting_agent_methods=True,
    )

    assert meta_agent.custom_method() == "explicit_value"


def test_find_combinations_without_evaluation_func(setup_agents):
    """Test find_combinations when evaluation_func is None.

    This tests that the function handles None return from evaluate_combination
    without raising TypeError from unpacking None.
    """
    model, _agents = setup_agents
    # When evaluation_func is None, evaluate_combination returns None
    # This should not cause a TypeError from unpacking
    result = find_combinations(model, model.agents, size=2, evaluation_func=None)
    assert result == []  # No combinations when no evaluation function


def test_meta_agent_remove_cleans_up_references(setup_agents):
    """Test that MetaAgent.remove() clears meta_agents and meta_agent on constituents."""
    model, agents = setup_agents
    meta_agent = MetaAgent(model, set(agents))

    # Verify references are set
    for agent in agents:
        assert meta_agent in agent.meta_agents
        assert agent.meta_agent is not None

    # Remove the meta-agent from the model
    meta_agent.remove()

    # All constituent agents should have no stale references
    for agent in agents:
        assert meta_agent not in agent.meta_agents
        assert agent.meta_agent is None

    # MetaAgent should be gone from the model
    assert meta_agent not in model.agents


def test_meta_agent_remove_cleans_entity_index(setup_agents):
    """Meta-agent teardown should only remove meta-agent record."""
    model, agents = setup_agents
    meta_agent = MetaAgent(model, set(agents))
    entity_index = ensure_entity_index(model)

    meta_entity_id = meta_agent.entity_id
    atomic_entity_ids = {agent.entity_id for agent in agents}

    meta_agent.remove()

    assert not entity_index.contains(meta_entity_id)
    assert all(entity_index.contains(entity_id) for entity_id in atomic_entity_ids)
    assert all(
        entity_index.kind_for(entity_id) == "atomic" for entity_id in atomic_entity_ids
    )


def test_meta_agent_remove_with_multiple_memberships():
    """Test remove() when agents belong to multiple meta-agents."""
    model = Model()
    a1 = CustomAgent(model)
    a2 = CustomAgent(model)

    ma1 = MetaAgent(model, {a1, a2}, name="Group1")
    ma2 = MetaAgent(model, {a1}, name="Group2")

    # a1 belongs to both meta-agents
    assert ma1 in a1.meta_agents
    assert ma2 in a1.meta_agents

    # Remove ma1 — a1 should still reference ma2
    ma1.remove()

    assert ma1 not in a1.meta_agents
    assert ma2 in a1.meta_agents
    assert a1.meta_agent is ma2

    # a2 was only in ma1, should be fully cleaned
    assert a2.meta_agent is None
    assert len(a2.meta_agents) == 0


def test_meta_agent_sorting_with_string_unique_ids():
    """Meta-agent membership sorting must not assume integer unique_ids.

    The backward-compatibility ``meta_agent`` attribute is kept up to date by
    sorting an agent's meta-agents by ``unique_id``. The previous
    ``key=lambda x: x.unique_id or 0`` injected an ``int`` into the key, which
    raised ``TypeError`` as soon as a ``unique_id`` was non-integer (e.g. a
    string) or ``None``. Regression test for #3563.
    """
    model = Model()
    agent = CustomAgent(model)

    # Two meta-agents that both contain `agent`, with string unique_ids.
    ma1 = MetaAgent(model, {agent}, name="Group1")
    ma2 = MetaAgent(model, {agent}, name="Group2")
    ma1.unique_id = "alpha"
    ma2.unique_id = "beta"

    # Re-trigger the membership sort via both add and remove paths.
    ma1.add_constituting_agents({agent})  # would previously raise TypeError
    assert agent.meta_agent is ma1  # "alpha" sorts before "beta"

    ma1.remove_constituting_agents({agent})
    assert agent.meta_agent is ma2  # only ma2 remains


def test_meta_agent_sorting_with_none_unique_id():
    """A None unique_id sorts safely rather than crashing. Regression test for #3563."""
    model = Model()
    agent = CustomAgent(model)

    ma1 = MetaAgent(model, {agent}, name="Group1")
    ma2 = MetaAgent(model, {agent}, name="Group2")
    ma1.unique_id = None
    ma2.unique_id = "beta"

    # None must not be coerced into an int that then clashes with the string id.
    ma1.add_constituting_agents({agent})  # would previously raise TypeError
    assert agent.meta_agent is ma1  # None sorts first (treated as lowest)


def test_create_meta_agent_repeated_instances_with_descriptor_parent():
    """Test creating multiple instances of same meta-agent class with descriptor-based parent.

    This tests Path 2 (existing meta-agent class instantiation) with a descriptor-based parent.
    Verifies that initial_attributes are applied correctly for the second and subsequent
    instances of the same meta-agent class.
    """
    model = Model()
    grid = OrthogonalMooreGrid((10, 10), random=model.random)

    class Robot(CellAgent):
        """Simple Robot agent for testing."""

    # Create first meta-agent instance (Path 3: new class creation)
    agent1 = Robot(model)

    swarm1 = create_meta_agent(
        model,
        "Swarm",
        [agent1],
        CellAgent,
        meta_attributes={"cell": grid[2, 2]},
    )

    assert isinstance(swarm1, MetaAgent)
    assert isinstance(swarm1, CellAgent)
    assert swarm1.cell == grid[2, 2]

    # Create second meta-agent instance of SAME class (Path 2: existing class instantiation)
    agent3 = Robot(model)
    agent4 = Robot(model)

    swarm2 = create_meta_agent(
        model,
        "Swarm",  # Same class name - triggers Path 2
        [agent3, agent4],
        CellAgent,
        meta_attributes={"cell": grid[5, 5]},
    )

    # Second instance should also have correct spatial attributes (Path 2 fix)
    assert isinstance(swarm2, MetaAgent)
    assert isinstance(swarm2, CellAgent)
    assert swarm2.cell == grid[5, 5], (
        "Path 2: Cell should be set correctly on second instance"
    )

    # First instance should be unaffected
    assert swarm1.cell == grid[2, 2]
