"""Tests for multi-level and overlapping meta-agents."""

from mesa import Agent, Model
from mesa.meta_agents import MetaAgents


def test_overlapping_meta_agents():
    """An agent can belong to multiple meta-agents simultaneously."""
    model = Model()
    meta_agents = MetaAgents(model)
    agent1 = Agent(model)
    agent2 = Agent(model)
    agent3 = Agent(model)

    group1 = meta_agents.create("Group1", {agent1, agent2}, Agent)
    group2 = meta_agents.create("Group2", {agent1, agent3}, Agent)

    assert set(meta_agents.groups_of(agent1)) == {group1, group2}
    assert set(meta_agents.groups_of(agent2)) == {group1}
    assert set(meta_agents.groups_of(agent3)) == {group2}


def test_remove_from_multiple_groups():
    """Removing an agent from one group keeps other memberships intact."""
    model = Model()
    meta_agents = MetaAgents(model)
    agent1 = Agent(model)
    group1 = meta_agents.create("Group1", {agent1}, Agent)
    group2 = meta_agents.create("Group2", {agent1}, Agent)

    assert len(meta_agents.groups_of(agent1)) == 2

    meta_agents.remove_member(group1, agent1)

    assert group1 not in meta_agents.groups_of(agent1)
    assert group2 in meta_agents.groups_of(agent1)
    assert len(meta_agents.groups_of(agent1)) == 1

    meta_agents.remove_member(group2, agent1)

    assert group2 not in meta_agents.groups_of(agent1)
    assert len(meta_agents.groups_of(agent1)) == 0


def test_create_independent_groups_with_overlap():
    """Different class names create separate groups even if agents overlap."""
    model = Model()
    meta_agents = MetaAgents(model)
    agent1 = Agent(model)
    agent2 = Agent(model)

    meta1 = meta_agents.create("GroupA", [agent1], Agent)
    meta2 = meta_agents.create("GroupB", [agent1, agent2], Agent)

    assert meta1 is not meta2
    assert meta1.__class__.__name__ == "GroupA"
    assert meta2.__class__.__name__ == "GroupB"
    assert set(meta_agents.groups_of(agent1)) == {meta1, meta2}
    assert set(meta_agents.groups_of(agent2)) == {meta2}
