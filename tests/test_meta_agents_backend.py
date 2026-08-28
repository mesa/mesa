"""Tests for membership storage."""

from mesa import Agent, Model
from mesa.meta_agents import MetaAgents
from mesa.meta_agents.backend import MembershipBackend


def test_add_and_query():
    """Add edges and verify basic query behavior."""
    backend = MembershipBackend()
    backend.add_membership("a1", "g1", "member")
    backend.add_membership("a2", "g1", "member")
    backend.add_membership("a1", "g2", "leader")

    assert backend.groups_of("a1") == {"g1", "g2"}
    assert backend.groups_of("a1", relation="member") == {"g1"}
    assert backend.agents_of("g1") == {"a1", "a2"}
    assert backend.relations_between("a1", "g1") == {"member"}
    backend.assert_invariants()


def test_multiple_relations_same_pair():
    """Allow multiple relation labels on the same agent-group pair."""
    backend = MembershipBackend()
    backend.add_membership("a1", "g1", "member")
    backend.add_membership("a1", "g1", "mentor")

    assert backend.relations_between("a1", "g1") == {"member", "mentor"}
    assert backend.groups_of("a1", relation="mentor") == {"g1"}
    backend.assert_invariants()


def test_idempotent_add_and_remove():
    """Repeated add/remove call should remain safe and deterministic."""
    backend = MembershipBackend()
    backend.add_membership("a1", "g1", "member")
    backend.add_membership("a1", "g1", "member")

    assert backend.as_triplets() == {("a1", "g1", "member")}

    backend.remove_membership("a1", "g1", "member")
    backend.remove_membership("a1", "g1", "member")
    assert backend.as_triplets() == set()
    backend.assert_invariants()


def test_replace_relation():
    """Replace an existing relation label for one edge."""
    backend = MembershipBackend()
    backend.add_membership("a1", "g1", "member")
    backend.replace_relation("a1", "g1", "member", "leader")

    assert backend.relations_between("a1", "g1") == {"leader"}
    assert backend.groups_of("a1", relation="member") == set()
    backend.assert_invariants()


def test_remove_agent_cascades_edges():
    """Removing an agent should clear all its incident edges."""
    backend = MembershipBackend()
    backend.bulk_add(
        [("a1", "g1", "member"), ("a1", "g2", "leader"), ("a2", "g1", "member")]
    )

    backend.remove_agent("a1")

    assert backend.groups_of("a1") == set()
    assert backend.agents_of("g1") == {"a2"}
    assert backend.agents_of("g2") == set()
    backend.assert_invariants()


def test_remove_group_cascades_edges():
    """Removing a group should clear all incident edges."""
    backend = MembershipBackend()
    backend.bulk_add(
        [("a1", "g1", "member"), ("a1", "g2", "leader"), ("a2", "g1", "member")]
    )

    backend.remove_group("g1")

    assert backend.agents_of("g1") == set()
    assert backend.groups_of("a1") == {"g2"}
    assert backend.groups_of("a2") == set()
    backend.assert_invariants()


def test_non_string_relation_key():
    """Allow non-string hashable relation keys."""
    backend = MembershipBackend()
    rel = ("role", 1)
    backend.add_membership("a1", "g1", rel)

    assert backend.relations_between("a1", "g1") == {rel}
    backend.assert_invariants()


def test_backend_uses_unique_ids_for_mesa_agents():
    """Membership bookkeeping should use unique_id values."""
    model = Model()
    meta_agents = MetaAgents(model)
    agent = Agent(model)
    group = meta_agents.create("Group", [agent])

    assert meta_agents.backend.as_triplets() == {
        (agent.unique_id, group.unique_id, "member")
    }
    assert meta_agents.backend.groups_of(agent) == {group.unique_id}
    assert meta_agents.backend.agents_of(group) == {agent.unique_id}
    meta_agents.backend.assert_invariants()
