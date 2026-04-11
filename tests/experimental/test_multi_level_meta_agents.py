"""Tests for multi-level and overlapping meta-agents."""

import pytest

from mesa import Agent, Model
from mesa.experimental.meta_agents.meta_agent import MetaAgent, create_meta_agent


def test_overlapping_meta_agents():
    """Test that an agent can belong to multiple meta-agents simultaneously."""
    model = Model()
    agent1 = Agent(model)
    agent2 = Agent(model)
    agent3 = Agent(model)

    # Create first meta-agent
    group1 = MetaAgent(model, {agent1, agent2}, name="Group1")

    # Create second meta-agent with agent1 overlapping
    group2 = MetaAgent(model, {agent1, agent3}, name="Group2")

    # Verify agent1 belongs to both
    assert hasattr(agent1, "meta_agents")
    assert group1 in agent1.meta_agents
    assert group2 in agent1.meta_agents
    assert len(agent1.meta_agents) == 2

    # Verify agent2 belongs to Group1 only
    assert group1 in agent2.meta_agents
    assert group2 not in agent2.meta_agents
    assert len(agent2.meta_agents) == 1

    # Verify agent3 belongs to Group2 only
    assert group1 not in agent3.meta_agents
    assert group2 in agent3.meta_agents
    assert len(agent3.meta_agents) == 1

    # Verify backward compatibility (agent.meta_agent)
    assert agent1.meta_agent in [group1, group2]
    assert agent2.meta_agent == group1
    assert agent3.meta_agent == group2


def test_remove_from_multiple_groups():
    """Test that removing an agent from one group keeps other memberships intact."""
    model = Model()
    agent1 = Agent(model)
    group1 = MetaAgent(model, {agent1}, name="Group1")
    group2 = MetaAgent(model, {agent1}, name="Group2")

    assert len(agent1.meta_agents) == 2

    # Remove from Group1
    group1.remove_constituting_agents({agent1})

    assert group1 not in agent1.meta_agents
    assert group2 in agent1.meta_agents
    assert len(agent1.meta_agents) == 1
    assert agent1.meta_agent == group2

    # Remove from Group2
    group2.remove_constituting_agents({agent1})

    assert group2 not in agent1.meta_agents
    assert len(agent1.meta_agents) == 0
    assert agent1.meta_agent is None


def test_create_meta_agent_independent_groups_with_overlap():
    """Test that create_meta_agent creates separate groups for different class names even if agents overlap."""
    model = Model()
    agent1 = Agent(model)
    agent2 = Agent(model)

    # Path 3: Create a new meta-agent class for GroupA
    meta1 = create_meta_agent(model, "GroupA", [agent1], Agent)

    # Path 3: Create a new meta-agent class for GroupB
    # Now it should NOT merge because the class name is different
    meta2 = create_meta_agent(model, "GroupB", [agent1, agent2], Agent)

    assert meta1 != meta2
    assert meta1.__class__.__name__ == "GroupA"
    assert meta2.__class__.__name__ == "GroupB"

    assert len(agent1.meta_agents) == 2
    assert any(ma.__class__.__name__ == "GroupA" for ma in agent1.meta_agents)
    assert any(ma.__class__.__name__ == "GroupB" for ma in agent1.meta_agents)

    assert len(agent2.meta_agents) == 1
    assert any(ma.__class__.__name__ == "GroupB" for ma in agent2.meta_agents)


def test_create_meta_agent_supports_custom_join_strategy():
    """Test that users can choose which existing same-class group gets expanded."""
    model = Model()
    a1 = Agent(model)
    a2 = Agent(model)
    a3 = Agent(model)

    group1 = create_meta_agent(model, "StrategyGroup", [a1], Agent)
    group2 = create_meta_agent(model, "StrategyGroupAlt", [a2], Agent)
    group2 = create_meta_agent(model, "StrategyGroup", [a2], Agent)

    assert group1 != group2

    chosen = create_meta_agent(
        model,
        "StrategyGroup",
        [a1, a2, a3],
        Agent,
        join_existing_meta_agent_func=lambda candidates: max(
            candidates, key=lambda x: x.unique_id
        ),
    )

    assert chosen == max([group1, group2], key=lambda x: x.unique_id)
    assert a3 in chosen.agents


def test_create_meta_agent_rejects_invalid_join_strategy_result():
    """Test that join strategy must return one of the candidate meta-agents."""
    model = Model()
    a1 = Agent(model)
    a2 = Agent(model)
    a3 = Agent(model)

    create_meta_agent(model, "StrictGroup", [a1], Agent)
    create_meta_agent(model, "OtherGroup", [a2], Agent)
    create_meta_agent(model, "StrictGroup", [a2], Agent)

    with pytest.raises(ValueError, match="must return one of the provided"):
        create_meta_agent(
            model,
            "StrictGroup",
            [a1, a2, a3],
            Agent,
            join_existing_meta_agent_func=lambda _candidates: object(),
        )
