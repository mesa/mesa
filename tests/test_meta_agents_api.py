"""Tests for the meta-agents membership manager."""

import pytest

from mesa import Agent, Model
from mesa.agent import AgentSet
from mesa.meta_agents import MembershipEdge, MembershipView, MetaAgents


def test_meta_agents_create_records_memberships():
    """Create should return live objects and record memberships."""
    model = Model()
    meta_agents = MetaAgents(model)
    agent_1 = Agent(model)
    agent_2 = Agent(model)

    meta_agent = meta_agents.create("Group", [agent_1, agent_2])

    assert meta_agents.backend.as_triplets() == {
        (agent_1.unique_id, meta_agent.unique_id, "member"),
        (agent_2.unique_id, meta_agent.unique_id, "member"),
    }

    view = meta_agents.query_memberships(agent_1)

    assert isinstance(view, MembershipView)
    assert view.subject is agent_1
    assert view.as_triplets() == {(agent_1, meta_agent, "member")}
    assert len(view) == 1
    assert isinstance(view.memberships[0], MembershipEdge)
    assert view.memberships[0].agent is agent_1
    assert view.memberships[0].group is meta_agent


def test_meta_agents_create_defaults_mesa_agent_type_to_agent():
    """Omitting mesa_agent_type should default to a plain Agent subclass."""
    model = Model()
    meta_agents = MetaAgents(model)
    agent = Agent(model)

    meta_agent = meta_agents.create("Group", [agent])

    assert isinstance(meta_agent, Agent)
    assert meta_agents.backend.as_triplets() == {
        (agent.unique_id, meta_agent.unique_id, "member"),
    }


def test_meta_agent_has_one_membership_manager():
    """A model can have only one membership manager."""
    model = Model()
    meta_agents = MetaAgents(model)

    assert model.meta_agents is meta_agents
    with pytest.raises(RuntimeError, match="different membership manager"):
        MetaAgents(model)


def test_members_of_and_groups_of():
    """members_of and groups_of return live AgentSets from the backend."""
    model = Model()
    meta_agents = MetaAgents(model)
    agent_1 = Agent(model)
    agent_2 = Agent(model)
    group = meta_agents.create("Group", [agent_1])

    meta_agents.add_member(group, agent_2)
    assert set(meta_agents.members_of(group)) == {agent_1, agent_2}
    assert set(meta_agents.groups_of(agent_2)) == {group}

    meta_agents.remove_member(group, agent_1)
    assert set(meta_agents.members_of(group)) == {agent_2}
    assert set(meta_agents.groups_of(agent_1)) == set()

    group.remove()
    assert meta_agents.backend.as_triplets() == set()
    assert group not in model.agents
    assert set(meta_agents.groups_of(agent_2)) == set()


def test_create_memberships_overrides_agents_list():
    """When memberships= is given, it replaces the agents list."""
    model = Model()
    meta_agents = MetaAgents(model)
    listed = Agent(model)
    actual = Agent(model)
    group = meta_agents.create(
        "Group",
        [listed],
        memberships=[(actual, "member")],
    )
    assert listed not in meta_agents.members_of(group)
    assert actual in meta_agents.members_of(group)
    assert group in meta_agents.groups_of(actual)
    assert group not in meta_agents.groups_of(listed)
    assert meta_agents.backend.as_triplets() == {
        (actual.unique_id, group.unique_id, "member")
    }


def test_member_remove_deactivates_memberships():
    """Removing a member from the model deactivates its memberships."""
    model = Model()
    meta_agents = MetaAgents(model)
    member = Agent(model)
    other = Agent(model)
    group = meta_agents.create("Group", [member, other])
    member.remove()
    assert meta_agents.backend.groups_of(member) == set()
    assert member not in meta_agents.members_of(group)
    assert other in meta_agents.members_of(group)
    assert group in meta_agents.groups_of(other)
    assert member not in model.agents
    assert group in model.agents


def test_meta_agent_remove_still_dissolves_after_agent_removal():
    """Bound group.remove() still dissolves after the agent leaves the model."""
    model = Model()
    meta_agents = MetaAgents(model)
    member = Agent(model)
    group = meta_agents.create("Group", [member])
    group.remove()
    assert meta_agents.backend.as_triplets() == set()
    assert group not in model.agents
    assert group not in meta_agents.groups_of(member)


def test_add_and_remove_member_by_group_name():
    """add_member/remove_member/members_of resolve a group by create() name."""
    model = Model()
    meta_agents = MetaAgents(model)
    alice = Agent(model)
    bob = Agent(model)
    carol = Agent(model)
    team = meta_agents.create("Team", [alice, bob])

    meta_agents.add_member("Team", carol)
    assert set(meta_agents.members_of("Team")) == {alice, bob, carol}
    assert team in meta_agents.groups_of(carol)

    meta_agents.remove_member("Team", alice)
    assert set(meta_agents.members_of("Team")) == {bob, carol}

    with pytest.raises(ValueError, match="No group named"):
        meta_agents.add_member("Missing", carol)

    meta_agents.create("Team", [Agent(model)])
    with pytest.raises(ValueError, match="Ambiguous group name"):
        meta_agents.add_member("Team", carol)


def test_add_and_remove_member_by_unique_id():
    """add_member/remove_member resolve unique_ids to live objects."""
    model = Model()
    meta_agents = MetaAgents(model)
    member = Agent(model)
    group = meta_agents.create("Group", [])
    meta_agents.add_member(group, member.unique_id)
    assert member in meta_agents.members_of(group)
    assert group in meta_agents.groups_of(member)
    assert meta_agents.backend.groups_of(member) == {group.unique_id}
    meta_agents.remove_member(group.unique_id, member.unique_id)
    assert member not in meta_agents.members_of(group)
    assert group not in meta_agents.groups_of(member)
    assert meta_agents.backend.groups_of(member) == set()


def test_remove_member_preserves_overlapping_memberships():
    """Removing one relation should keep unrelated memberships intact."""
    model = Model()
    meta_agents = MetaAgents(model)
    agent = Agent(model)
    partner = Agent(model)
    group_one = meta_agents.create("GroupOne", [agent, partner])
    group_two = meta_agents.create("GroupTwo", [agent])

    assert len(meta_agents.groups_of(agent)) == 2

    view = meta_agents.remove_member(group_one, agent)

    assert view.as_triplets() == {(agent, group_two, "member")}
    assert meta_agents.backend.groups_of(agent) == {group_two.unique_id}
    assert group_one not in meta_agents.groups_of(agent)
    assert group_two in meta_agents.groups_of(agent)
    assert set(meta_agents.groups_of(partner)) == {group_one}


def test_dissolve_cleans_only_target_group():
    """Dissolving a group should keep overlapping memberships on other groups."""
    model = Model()
    meta_agents = MetaAgents(model)
    agent_1 = Agent(model)
    agent_2 = Agent(model)
    agent_3 = Agent(model)
    group_one = meta_agents.create("GroupOne", [agent_1, agent_2])
    group_two = meta_agents.create("GroupTwo", [agent_1, agent_3])

    snapshot = meta_agents.dissolve(group_one)

    assert snapshot.as_triplets() == {
        (agent_1, group_one, "member"),
        (agent_2, group_one, "member"),
    }
    assert meta_agents.backend.groups_of(agent_1) == {group_two.unique_id}
    assert meta_agents.backend.groups_of(agent_2) == set()
    assert meta_agents.backend.groups_of(agent_3) == {group_two.unique_id}
    assert group_one not in model.agents
    assert group_two in model.agents
    assert group_one not in meta_agents.groups_of(agent_1)
    assert group_two in meta_agents.groups_of(agent_1)


def test_deactivate_detaches_all_memberships_without_removing_entity():
    """Deactivate should clear memberships but keep the entity registered."""
    model = Model()
    meta_agents = MetaAgents(model)
    agent_1 = Agent(model)
    agent_2 = Agent(model)
    group = meta_agents.create("Group", [agent_1, agent_2])

    snapshot = meta_agents.deactivate(agent_1)

    assert snapshot.as_triplets() == {(agent_1, group, "member")}
    assert meta_agents.backend.groups_of(agent_1) == set()
    assert agent_1 in model.agents
    assert group not in meta_agents.groups_of(agent_1)
    assert group in meta_agents.groups_of(agent_2)


def test_dissolve_after_member_already_removed():
    """Dissolve handles a group whose live members were deregistered first."""
    model = Model()
    meta_agents = MetaAgents(model)
    agent = Agent(model)
    group = meta_agents.create("Group", [agent])
    agent.remove()
    assert meta_agents.backend.as_triplets() == set()
    group.remove()
    assert meta_agents.backend.as_triplets() == set()
    assert group not in model.agents


def _four_level_hierarchy():
    """Build world -> region -> city -> household -> person hierarchy."""
    model = Model()
    meta_agents = MetaAgents(model)

    person_a = Agent(model)
    person_b = Agent(model)
    person_c = Agent(model)
    household = meta_agents.create("Household", [person_a, person_b])
    city = meta_agents.create("City", [household, person_c])
    region = meta_agents.create("Region", [city])
    world = meta_agents.create("World", [region])

    return (
        model,
        meta_agents,
        world,
        region,
        city,
        household,
        person_a,
        person_b,
        person_c,
    )


def test_at_level_four_level_hierarchy():
    """Each containment depth from root returns the expected AgentSet."""
    (
        _model,
        meta_agents,
        world,
        region,
        city,
        household,
        person_a,
        person_b,
        person_c,
    ) = _four_level_hierarchy()

    level0 = meta_agents.at_level(0, root=world)
    assert isinstance(level0, AgentSet)
    assert set(level0) == {world}

    assert set(meta_agents.at_level(1, root=world)) == {region}
    assert set(meta_agents.at_level(2, root=world)) == {city}
    assert set(meta_agents.at_level(3, root=world)) == {household, person_c}
    assert set(meta_agents.at_level(4, root=world)) == {person_a, person_b}


def test_at_level_siblings_and_agentset_ops():
    """Siblings share a level and results support AgentSet selection."""
    _, meta_agents, world, _, _, _, person_a, person_b, _ = _four_level_hierarchy()

    level4 = meta_agents.at_level(4, root=world)
    assert set(level4) == {person_a, person_b}
    selected = level4.select(lambda a: a is person_a)
    assert set(selected) == {person_a}


def test_at_level_order_is_deterministic():
    """Same-level agents keep stable order (sorted member ids per group)."""
    model = Model()
    meta_agents = MetaAgents(model)
    root = Agent(model)
    fillers = [Agent(model) for _ in range(9)]
    agent_11 = Agent(model)
    agent_2 = fillers[0]

    meta_agents.backend.add_membership(agent_11, root, "member")
    meta_agents.backend.add_membership(agent_2, root, "member")

    assert str(agent_11.unique_id) < str(agent_2.unique_id)
    assert agent_2.unique_id < agent_11.unique_id
    assert list(meta_agents.at_level(1, root=root)) == [agent_11, agent_2]
    assert list(meta_agents.at_level(1, root=root)) == list(
        meta_agents.at_level(1, root=root)
    )


def test_at_level_default_relation_and_explicit_relation():
    """Default ignores non-member edges; explicit relation includes them."""
    model = Model()
    meta_agents = MetaAgents(model)
    root = Agent(model)
    member = Agent(model)
    ally = Agent(model)
    meta_agents.backend.add_membership(member, root, "member")
    meta_agents.backend.add_membership(ally, root, "ally")

    assert set(meta_agents.at_level(1, root=root)) == {member}
    assert set(meta_agents.at_level(1, root=root, relation="ally")) == {ally}
    assert set(meta_agents.at_level(1, root=root, relation=None)) == {member, ally}


def test_at_level_overlapping_paths_use_nearest_depth():
    """An agent on multiple paths appears only at its shallowest level."""
    model = Model()
    meta_agents = MetaAgents(model)
    root = Agent(model)
    mid = Agent(model)
    leaf = Agent(model)
    meta_agents.backend.add_membership(leaf, root, "member")
    meta_agents.backend.add_membership(mid, root, "member")
    meta_agents.backend.add_membership(leaf, mid, "member")

    assert set(meta_agents.at_level(1, root=root)) == {leaf, mid}
    assert set(meta_agents.at_level(2, root=root)) == set()


def test_at_level_empty_and_validation():
    """Absent levels are empty; invalid level/root raise ValueError."""
    model = Model()
    meta_agents = MetaAgents(model)
    root = Agent(model)
    outsider = object()

    assert len(meta_agents.at_level(3, root=root)) == 0

    with pytest.raises(ValueError, match="non-negative"):
        meta_agents.at_level(-1, root=root)

    with pytest.raises(ValueError, match="not registered"):
        meta_agents.at_level(0, root=outsider)


def test_at_level_cyclic_membership_terminates():
    """Cyclic membership graphs must not loop indefinitely."""
    model = Model()
    meta_agents = MetaAgents(model)
    a = Agent(model)
    b = Agent(model)
    meta_agents.backend.add_membership(b, a, "member")
    meta_agents.backend.add_membership(a, b, "member")

    assert set(meta_agents.at_level(0, root=a)) == {a}
    assert set(meta_agents.at_level(1, root=a)) == {b}
    assert set(meta_agents.at_level(2, root=a)) == set()


def test_live_entity_cache_lifecycle_synchronization():
    """MetaAgents live entity cache should track additions, removals, and pre-existing agents."""
    model = Model()
    early_agent = Agent(model)

    # 1. Initialize MetaAgents after early_agent was created
    meta_agents = MetaAgents(model)
    assert early_agent.unique_id in meta_agents._live_entity_cache
    assert meta_agents._live_entity_cache[early_agent.unique_id] is early_agent

    # 2. Add dynamic agent after MetaAgents was created
    dynamic_agent = Agent(model)
    assert dynamic_agent.unique_id in meta_agents._live_entity_cache
    assert meta_agents._live_entity_cache[dynamic_agent.unique_id] is dynamic_agent

    # 3. Create a group (which registers a MetaAgent)
    group = meta_agents.create("Squad", [early_agent, dynamic_agent])
    assert group.unique_id in meta_agents._live_entity_cache
    assert meta_agents._live_entity_cache[group.unique_id] is group
    assert set(meta_agents.members_of(group)) == {early_agent, dynamic_agent}

    # 4. Remove an agent and check cache purge
    dynamic_agent.remove()
    assert dynamic_agent.unique_id not in meta_agents._live_entity_cache
    assert set(meta_agents.members_of(group)) == {early_agent}

    # 5. Dissolve group and check group purge
    meta_agents.dissolve(group)
    assert group.unique_id not in meta_agents._live_entity_cache
