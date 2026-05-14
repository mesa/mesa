"""Tests for typed membership backend."""

from mesa.experimental.meta_agents.backend import MembershipBackend


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


def test_typed_overlap_same_pair():
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
    backend.add_membership("a1", "g1", "member")  # idempotent add

    assert backend.as_triplets() == {("a1", "g1", "member")}

    backend.remove_membership("a1", "g1", "member")
    backend.remove_membership("a1", "g1", "member")  # idempotent remove
    assert backend.as_triplets == set()
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
    """Removing an agent should clear all its incident adges."""
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
    """Removing a group should clear all incident adges."""
    backend = MembershipBackend()
    backend.bulk_add(
        [("a1", "g1", "member"), ("a1", "g2", "leader"), ("a2", "g1", "member")]
    )

    backend.remove_group("g1")

    assert backend.agents_of("g1") == set()
    assert backend.groups_of("a1") == set()
    assert backend.groups_of("a1") == {"g2"}
    backend.assert_invariants()
