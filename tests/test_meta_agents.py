"""Tests for meta-agent creation, discovery helpers, and group reuse."""

import pytest

from mesa import Agent, Model
from mesa.discrete_space.cell_agent import CellAgent
from mesa.discrete_space.grid import OrthogonalMooreGrid
from mesa.meta_agents import (
    MetaAgents,
    evaluate_combination,
    find_combinations,
)
from mesa.meta_agents.meta_agent import MetaAgent


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
    """Set up the model, membership manager, and agents for testing."""
    model = Model()
    MetaAgents(model)
    agent1 = CustomAgent(model)
    agent2 = Agent(model)
    agent3 = Agent(model)
    agent4 = Agent(model)
    agent4.custom_attribute = "custom_value"
    agents = [agent1, agent2, agent3, agent4]
    return model, agents


def test_create_records_attributes_and_methods(setup_agents):
    """Create a group with explicit attributes and methods."""
    model, agents = setup_agents
    meta_agent = model.meta_agents.create(
        "MetaAgentClass",
        agents,
        Agent,
        meta_attributes={"attribute1": "value1"},
        meta_methods={"function1": lambda self: "function1"},
    )
    assert meta_agent.attribute1 == "value1"
    assert meta_agent.function1() == "function1"
    assert set(model.meta_agents.members_of(meta_agent)) == set(agents)


def test_create_reuses_existing_class_for_new_instance(setup_agents):
    """Same class name with disjoint members creates a second instance."""
    model, agents = setup_agents
    meta_agent = model.meta_agents.create(
        "MetaAgentClass",
        [agents[0], agents[2]],
        Agent,
        meta_attributes={"attribute1": "value1"},
        meta_methods={"function1": lambda self: "function1"},
    )
    meta_agent2 = model.meta_agents.create(
        "MetaAgentClass",
        [agents[1], agents[3]],
        Agent,
        meta_attributes={"attribute2": "value2"},
        meta_methods={"function2": lambda self: "function2"},
    )
    assert meta_agent is not meta_agent2
    assert meta_agent.function1() == "function1"
    assert set(model.meta_agents.members_of(meta_agent)) == {agents[2], agents[0]}
    assert meta_agent2.function2() == "function2"
    assert set(model.meta_agents.members_of(meta_agent2)) == {agents[1], agents[3]}


def test_create_adds_to_existing_group_of_same_class(setup_agents):
    """Reusing a class name with an overlapping member extends that group."""
    model, agents = setup_agents
    meta_agent1 = model.meta_agents.create(
        "MetaAgentClass",
        [agents[0], agents[3]],
        Agent,
        meta_attributes={"attribute1": "value1"},
        meta_methods={"function1": lambda self: "function1"},
    )
    reused = model.meta_agents.create(
        "MetaAgentClass",
        [agents[1], agents[0], agents[2]],
        Agent,
    )
    assert reused is meta_agent1
    assert set(model.meta_agents.members_of(meta_agent1)) == {
        agents[0],
        agents[1],
        agents[2],
        agents[3],
    }
    assert meta_agent1.function1() == "function1"
    assert meta_agent1.attribute1 == "value1"


def test_create_registers_group_on_model(setup_agents):
    """Created groups are registered model agents."""
    model, agents = setup_agents
    meta_agent = model.meta_agents.create(
        "MetaAgentClass",
        agents,
        Agent,
        meta_attributes={"attribute1": "value1"},
        meta_methods={"function1": lambda self: "function1"},
    )
    model.step()
    assert meta_agent in model.agents
    assert meta_agent.function1() == "function1"
    assert meta_agent.attribute1 == "value1"


def test_evaluate_combination(setup_agents):
    """evaluate_combination returns the group and its score."""
    model, agents = setup_agents

    def evaluation_func(agent_set):
        return len(agent_set)

    result = evaluate_combination(tuple(agents), model, evaluation_func)
    assert result is not None
    assert result[1] == len(agents)


def test_find_combinations(setup_agents):
    """find_combinations scores and filters candidate groups."""
    model, agents = setup_agents

    def evaluation_func(agent_set):
        return len(agent_set)

    def filter_func(combinations):
        return [combo for combo in combinations if combo[1] > 2]

    combinations = find_combinations(
        model,
        set(agents),
        size=(2, 4),
        evaluation_func=evaluation_func,
        filter_func=filter_func,
    )
    assert len(combinations) > 0
    for combo in combinations:
        assert combo[1] > 2


def test_find_combinations_allows_zero_value(setup_agents):
    """Zero-valued evaluation results are preserved."""
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
    """Tuple size bounds are inclusive and support equal bounds."""
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


def test_find_combinations_without_evaluation_func(setup_agents):
    """No evaluation function yields no combinations."""
    model, _agents = setup_agents
    result = find_combinations(model, model.agents, size=2, evaluation_func=None)
    assert result == []


def test_meta_agent_constructor_requires_membership_manager():
    """Direct MetaAgent construction is rejected."""
    model = Model()
    with pytest.raises(RuntimeError, match=r"model\.meta_agents\.create"):
        MetaAgent(model)


def test_create_repeated_instances_with_descriptor_parent():
    """Multiple instances of the same class work with a descriptor parent."""
    model = Model()
    MetaAgents(model)
    grid = OrthogonalMooreGrid((10, 10), random=model.random)

    class Robot(CellAgent):
        """Simple Robot agent for testing."""

    agent1 = Robot(model)
    swarm1 = model.meta_agents.create(
        "Swarm",
        [agent1],
        CellAgent,
        meta_attributes={"cell": grid[2, 2]},
    )
    assert isinstance(swarm1, MetaAgent)
    assert isinstance(swarm1, CellAgent)
    assert swarm1.cell == grid[2, 2]

    agent3 = Robot(model)
    agent4 = Robot(model)
    swarm2 = model.meta_agents.create(
        "Swarm",
        [agent3, agent4],
        CellAgent,
        meta_attributes={"cell": grid[5, 5]},
    )
    assert swarm2 is not swarm1
    assert isinstance(swarm2, MetaAgent)
    assert isinstance(swarm2, CellAgent)
    assert swarm2.cell == grid[5, 5]
    assert swarm1.cell == grid[2, 2]
