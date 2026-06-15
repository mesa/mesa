"""Tests for typed membership backend."""

import pytest
from mesa import Agent, Model
from mesa.experimental.meta_agents.backend import MembershipBackend
from mesa.experimental.meta_agents.identity import ensure_entity_index
from mesa.experimental.meta_agents.meta_agent import MetaAgent


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


def test_backend_prefers_explicit_entity_ids():
    """Meta-Agent membership bookkeeping should use stable entity ids."""
    model = Model()
    agent = Agent(model)
    meta_agent = MetaAgent(model, {agent}, name="Group")
    backend = MembershipBackend()

    backend.add_membership(agent, meta_agent, "member")

    assert backend.as_triplets() == {(agent.entity_id, meta_agent.entity_id, "member")}
    assert backend.groups_of(agent) == {meta_agent.entity_id}
    assert backend.agents_of(meta_agent) == {agent.entity_id}
    backend.assert_invariants()


def test_backend_registry_lookup_stays_stable_after_unique_id_updates():
    """Changing unqiue id should not disturb explicit entity lookup."""
    model = Model()
    agent = Agent(model)
    meta_agent = MetaAgent(model, {agent}, name="Group")
    entity_index = ensure_entity_index(model)

    original_entity_id = meta_agent.entity_id
    meta_agent.unique_id = "renamed-group"
    entity_index.register(meta_agent, kind="meta")

    assert entity_index.entity_for(original_entity_id) is meta_agent
    assert entity_index.kind_for(original_entity_id) == "meta"
    assert entity_index.entity_id_for(agent) == agent.entity_id
    entity_index.assert_invariants()


def test_model_deregister_clean_atomic_entity_index_entries():
    """Removing an atomic agent from the model should drop its identity record."""
    model = Model()
    agent = Agent(model)
    entity_index = ensure_entity_index(model)

    entity_id = agent.entity_id
    assert entity_index.entity_for(entity_id) is agent

    agent.remove()

    assert not entity_index.contains(entity_id)
    with pytest.raises(KeyError):
        entity_index.entity_for(entity_id)
