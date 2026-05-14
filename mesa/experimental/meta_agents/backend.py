"""Backend foundation for typed overlapping meta-agent memberships.

Phase 1 scope:
- Canonical typed membership representation
- Safe update operations
- Invariant checks
- Internal-only API (no public facade changes yet)
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable

Triplet = tuple[Hashable, Hashable, str]


class MembershipBackend:
    """Canonical backend for typed overlapping memberships."""

    def __init__(self) -> None:
        """Initialize empty triplet storage and bidirectional indexes."""
        self._triplets: set[Triplet] = set()
        self._by_agent: dict[Hashable, set[tuple[Hashable, str]]] = defaultdict(set)
        self._by_group: dict[Hashable, set[tuple[Hashable, str]]] = defaultdict(set)

    def add_membership(self, agent: Hashable, group: Hashable, relation: str) -> None:
        """Add one typed membership edge if it does not already exist."""
        if not isinstance(relation, str):
            raise TypeError("relation must be a string")
        triplet = (agent, group, relation)
        if triplet in self._triplets:
            return
        self._triplets.add(triplet)
        self._by_agent[agent].add((group, relation))
        self._by_group[agent].add((agent, relation))

    def bulk_add(self, memberships: Iterable[Triplet]) -> None:
        """Add many typed membership edges."""
        for agent, group, relation in memberships:
            self.add_membership(agent, group, relation)

    def remove_membership(
        self, agent: Hashable, group: Hashable, relation: str
    ) -> None:
        """Remove one typed membership edge if present."""
        triplet = (agent, group, relation)
        if triplet not in self._triplets:
            return
        self._triplets.remove(triplet)

        self._by_agent[agent].discard((group, relation))
        if not self._by_agent[agent]:
            del self._by_agent[agent]

        self._by_group[group].discard((agent, relation))
        if not self._by_group[group]:
            del self._by_group[group]

    def replace_relation(
        self, agent: Hashable, group: Hashable, old_relation: str, new_relation: str
    ) -> None:
        """Replace one relation label for one agent-group pair."""
        self.remove_membership(agent, group, old_relation)
        self.add_membership(agent, group, new_relation)

    def remove_agent(self, agent: Hashable) -> None:
        """Remove an agent and all incident memberships."""
        edges = list(self._by_agent.get(agent, set()))
        for group, relation in edges:
            self.remove_membership(agent, group, relation)

    def remove_group(self, group: Hashable) -> None:
        """Remove a group and all incident memberships."""
        edges = list(self._by_group.get(group, set()))
        for agent, relation in edges:
            self.remove_membership(agent, group, relation)

    def groups_of(self, agent: Hashable, relation: str | None = None) -> set[Hashable]:
        """Return groups for an agent, optionally filtered by relation."""
        entries = self._by_agent.get(agent, set())
        if relation is None:
            return {group for group, _ in entries}
        return {group for group, rel in entries if rel == relation}

    def agents_of(self, group: Hashable, relation: str | None = None) -> set[Hashable]:
        """Return agents for a group, optionally filtered relation."""
        entries = self._by_group.get(group, set())
        if relation is None:
            return {agent for agent, _ in entries}
        return {agent for agent, rel in entries if rel == relation}

    def relations_between(self, agent: Hashable, group: Hashable) -> set[str]:
        """Return all relation types between one agent and one group."""
        return {
            relation
            for linked_group, relation in self._by_agent.get(agent, set())
            if linked_group == group
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
