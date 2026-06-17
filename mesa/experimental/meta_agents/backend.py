"""Backend foundation for typed overlapping meta-agent memberships.

- Canonical typed membership representation
- Safe update operations
- Invariant checks
- Internal-only API (no public facade changes yet)
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable

RelationKey = Hashable
Triplet = tuple[Hashable, Hashable, RelationKey]


class MembershipBackend:
    """Canonical backend for typed overlapping memberships."""

    def __init__(self) -> None:
        """Initialize empty triplet storage and bidirectional indexes."""
        self._triplets: set[Triplet] = set()
        self._by_agent: dict[Hashable, set[tuple[Hashable, RelationKey]]] = defaultdict(
            set
        )
        self._by_group: dict[Hashable, set[tuple[Hashable, RelationKey]]] = defaultdict(
            set
        )

    def _to_id(self, entity: Hashable) -> Hashable:
        """Normalize entity to canonical ID.

        Uses an explicit ``entity_id`` when available, then falls back to
        ``mesa.agent.Agent.unique_id`` and finally the entity as-is (for already
        hashable external IDs).
        """
        entity_id = getattr(entity, "entity_id", None)
        if entity_id is not None:
            return entity_id
        return getattr(entity, "unique_id", entity)

    def add_membership(
        self, agent: Hashable, group: Hashable, relation: RelationKey
    ) -> None:
        """Add one typed membership edge if it does not already exist."""
        agent_id = self._to_id(agent)
        group_id = self._to_id(group)
        triplet = (agent_id, group_id, relation)
        if triplet in self._triplets:
            return
        self._triplets.add(triplet)
        self._by_agent[agent_id].add((group_id, relation))
        self._by_group[group_id].add((agent_id, relation))

    def bulk_add(self, memberships: Iterable[Triplet]) -> None:
        """Add many typed membership edges."""
        for agent, group, relation in memberships:
            self.add_membership(agent, group, relation)

    def remove_membership(
        self, agent: Hashable, group: Hashable, relation: RelationKey
    ) -> None:
        """Remove one typed membership edge if present."""
        agent_id = self._to_id(agent)
        group_id = self._to_id(group)
        triplet = (agent_id, group_id, relation)
        if triplet not in self._triplets:
            return
        self._triplets.remove(triplet)

        self._by_agent[agent_id].discard((group_id, relation))
        if not self._by_agent[agent_id]:
            del self._by_agent[agent_id]

        self._by_group[group_id].discard((agent_id, relation))
        if not self._by_group[group_id]:
            del self._by_group[group_id]

    def replace_relation(
        self,
        agent: Hashable,
        group: Hashable,
        old_relation: RelationKey,
        new_relation: RelationKey,
    ) -> None:
        """Replace one relation label for one agent-group pair."""
        self.remove_membership(agent, group, old_relation)
        self.add_membership(agent, group, new_relation)

    def remove_agent(self, agent: Hashable) -> None:
        """Remove an agent and all incident memberships."""
        agent_id = self._to_id(agent)
        # TODO(perf): Current removal is O(degree(agent)) with per-edge updates.
        # Revisit with bulk/index-aware deletion once benchmark baselines are in place.
        edges = list(self._by_agent.get(agent_id, set()))
        for group_id, relation in edges:
            self.remove_membership(agent_id, group_id, relation)

    def remove_group(self, group: Hashable) -> None:
        """Remove a group and all incident memberships."""
        group_id = self._to_id(group)
        # TODO(perf): Current removal is O(degree(agent)) with per-edge updates.
        # Revisit with bulk/index-aware deletion once benchmark baselines are in place.
        edges = list(self._by_group.get(group_id, set()))
        for agent_id, relation in edges:
            self.remove_membership(agent_id, group_id, relation)

    def groups_of(
        self, agent: Hashable, relation: RelationKey | None = None
    ) -> set[Hashable]:
        """Return groups for an agent, optionally filtered by relation."""
        agent_id = self._to_id(agent)
        entries = self._by_agent.get(agent_id, set())
        if relation is None:
            return {group for group, _ in entries}
        return {group for group, rel in entries if rel == relation}

    def agents_of(
        self, group: Hashable, relation: RelationKey | None = None
    ) -> set[Hashable]:
        """Return agents for a group, optionally filtered relation."""
        group_id = self._to_id(group)
        entries = self._by_group.get(group_id, set())
        if relation is None:
            return {agent for agent, _ in entries}
        return {agent for agent, rel in entries if rel == relation}

    def relations_between(self, agent: Hashable, group: Hashable) -> set[RelationKey]:
        """Return all relation types between one agent and one group."""
        agent_id = self._to_id(agent)
        group_id = self._to_id(group)
        return {
            relation
            for linked_group, relation in self._by_agent.get(agent_id, set())
            if linked_group == group_id
        }

    def as_triplets(self) -> set[Triplet]:
        """Return all memberships as canonical triplets."""
        return set(self._triplets)

    def assert_invariants(self) -> None:
        """Assert internal consistency between triplets and indexes."""
        # Triplets must match both indexes.
        for agent, group, relation in self._triplets:
            assert (group, relation) in self._by_agent.get(agent, set())
            assert (agent, relation) in self._by_group.get(group, set())

        # Agent index must match triplets
        for agent, edges in self._by_agent.items():
            for group, relation in edges:
                assert (agent, group, relation) in self._triplets

        # Group index must match triplets
        for group, edges in self._by_group.items():
            for agent, relation in edges:
                assert (agent, group, relation) in self._triplets
