"""Membership manager for meta-agents."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass
from typing import Any

from mesa.agent import Agent, AgentSet
from mesa.experimental.mesa_signals import ModelSignals

from .backend import MembershipBackend, RelationKey, Triplet
from .meta_agent import (
    _create_meta_agent_instance,
    _deduplicate_preserving_order,
)


@dataclass(frozen=True, slots=True)
class MembershipEdge:
    """A membership edge with live agent and group objects."""

    agent: Any
    group: Any
    relation: RelationKey


@dataclass(frozen=True, slots=True)
class MembershipView:
    """Read-only snapshot of memberships for one entity."""

    subject: Any
    memberships: tuple[MembershipEdge, ...]

    def __iter__(self):
        """Iterate over the resolved memberships."""
        return iter(self.memberships)

    def __len__(self) -> int:
        """Return the number of resolved memberships."""
        return len(self.memberships)

    def as_triplets(self) -> set[tuple[Any, Any, RelationKey]]:
        """Return the memberships as live-object triplets."""
        return {(edge.agent, edge.group, edge.relation) for edge in self.memberships}

    @property
    def agents(self) -> set[Any]:
        """Return all unique agents referenced by the view."""
        return {edge.agent for edge in self.memberships}

    @property
    def groups(self) -> set[Any]:
        """Return all unique groups referenced by the view."""
        return {edge.group for edge in self.memberships}

    @property
    def relations(self) -> set[RelationKey]:
        """Return all unique relation labels referenced by the view."""
        return {edge.relation for edge in self.memberships}


class MetaAgents:
    """Membership manager for agents composed of other agents.

    Tracks who belongs to which group. Change memberships with ``create``,
    ``add_member``, ``remove_member``, ``deactivate``, and ``dissolve``.
    Removing an agent from the model also removes its memberships.
    """

    def __init__(self, model: Any, backend: MembershipBackend | None = None) -> None:
        """Create a membership manager bound to one model."""
        existing_api = getattr(model, "meta_agents", None)
        if existing_api is not None and existing_api is not self:
            raise RuntimeError("Model already has a different membership manager")
        self.model = model
        self.backend = backend or MembershipBackend()
        model.meta_agents = self

        self.model.observe("agents", ModelSignals.AGENT_REMOVED, self._on_agent_removed)

    def _on_agent_removed(self, signal) -> None:
        """Deactivate memberships when a live agent leaves the model."""
        args = signal.additional_kwargs.get("args") or ()
        if not args:
            return
        self.deactivate(args[0])

    def _entity_id(self, entity: Hashable) -> Hashable:
        """Return the backend identity for a live entity or hashable external id."""
        return getattr(entity, "unique_id", entity)

    def _live_entity_lookup(self) -> dict[Hashable, Any]:
        """Build a lookup from backend ids back to live model objects."""
        lookup: dict[Hashable, Any] = {}
        for entity in self.model.agents:
            entity_id = getattr(entity, "unique_id", None)
            if entity_id is not None:
                lookup[entity_id] = entity
        return lookup

    def _resolve_entity(self, entity_id: Hashable) -> Any:
        """Resolve a backend id back to a live object when possible."""
        return self._live_entity_lookup().get(entity_id, entity_id)

    def _resolve_group(self, group: Hashable) -> Any:
        """Resolve a group from a live object, unique id, or group name."""
        lookup = self._live_entity_lookup()
        entity_id = self._entity_id(group)
        if entity_id in lookup:
            return lookup[entity_id]
        if isinstance(group, str):
            matches = list(
                dict.fromkeys(
                    entity
                    for entity in self.model.agents
                    if getattr(entity, "name", None) == group
                    or entity.__class__.__name__ == group
                )
            )
            if len(matches) == 1:
                return matches[0]
            if not matches:
                raise ValueError(f"No group named {group!r}")
            raise ValueError(f"Ambiguous group name {group!r}")
        return group

    def _resolve_view(
        self, entity: Hashable, triplets: Iterable[Triplet]
    ) -> MembershipView:
        """Convert backend triplets into a user-facing snapshot."""
        lookup = self._live_entity_lookup()
        resolved_edges: list[MembershipEdge] = []
        for agent_id, group_id, relation in sorted(
            triplets,
            key=lambda triplet: (
                str(triplet[0]),
                str(triplet[1]),
                repr(triplet[2]),
            ),
        ):
            resolved_edges.append(
                MembershipEdge(
                    agent=lookup.get(agent_id, agent_id),
                    group=lookup.get(group_id, group_id),
                    relation=relation,
                )
            )

        return MembershipView(
            subject=self._resolve_entity(entity),
            memberships=tuple(resolved_edges),
        )

    def _detach_entity(self, entity: Hashable) -> MembershipView:
        """Remove all incident memberships for one entity."""
        snapshot = self.query_memberships(entity)
        self.backend.remove_agent(entity)
        self.backend.remove_group(entity)
        return snapshot

    def create(
        self,
        new_agent_class: str,
        agents: Iterable[Any],
        mesa_agent_type: type[Agent] | None,
        meta_attributes: dict[str, Any] | None = None,
        meta_methods: dict[str, Callable] | None = None,
        relation: RelationKey = "member",
        memberships: Iterable[tuple[Any, RelationKey]] | None = None,
    ) -> Any:
        """Create a meta-agent and record its memberships."""
        member_relations = list(memberships) if memberships is not None else None
        if member_relations is not None:
            agents = _deduplicate_preserving_order(
                member for member, _ in member_relations
            )
        else:
            agents = _deduplicate_preserving_order(agents)

        meta_agent = _create_meta_agent_instance(
            self.model,
            new_agent_class,
            agents,
            mesa_agent_type,
            meta_attributes=meta_attributes,
            meta_methods=meta_methods,
            _membership_api=self,
        )

        if member_relations is None:
            member_relations = [(agent, relation) for agent in agents]

        self.backend.bulk_add(
            [(member, meta_agent, rel) for member, rel in member_relations]
        )

        return meta_agent

    def add_member(
        self,
        group: Hashable,
        member: Hashable,
        relation: RelationKey = "member",
    ) -> MembershipView:
        """Add one member to one group."""
        lookup = self._live_entity_lookup()
        member = lookup.get(self._entity_id(member), member)
        group = self._resolve_group(group)

        self.backend.add_membership(member, group, relation)
        return self.query_memberships(member)

    def remove_member(
        self,
        group: Hashable,
        member: Hashable,
        relation: RelationKey = "member",
    ) -> MembershipView:
        """Remove one member from one group."""
        lookup = self._live_entity_lookup()
        member = lookup.get(self._entity_id(member), member)
        group = self._resolve_group(group)

        self.backend.remove_membership(member, group, relation)
        return self.query_memberships(member)

    def members_of(
        self, group: Hashable, relation: RelationKey | None = None
    ) -> AgentSet:
        """Return the live members of one group as an AgentSet."""
        group = self._resolve_group(group)
        lookup = self._live_entity_lookup()
        members = [
            lookup[member_id]
            for member_id in sorted(
                self.backend.agents_of(group, relation=relation), key=str
            )
            if member_id in lookup
        ]
        return AgentSet(members, random=self.model.random)

    def groups_of(
        self, agent: Hashable, relation: RelationKey | None = None
    ) -> AgentSet:
        """Return the live groups that contain one agent as an AgentSet."""
        lookup = self._live_entity_lookup()
        groups = [
            lookup[group_id]
            for group_id in sorted(
                self.backend.groups_of(agent, relation=relation), key=str
            )
            if group_id in lookup
        ]
        return AgentSet(groups, random=self.model.random)

    def query_memberships(
        self, entity: Hashable, relation: RelationKey | None = None
    ) -> MembershipView:
        """Return a resolved, read-only snapshot of one entity's memberships."""
        entity_id = self._entity_id(entity)
        triplets = (
            triplet
            for triplet in self.backend.as_triplets()
            if (triplet[0] == entity_id or triplet[1] == entity_id)
            and (relation is None or triplet[2] == relation)
        )
        return self._resolve_view(entity_id, triplets)

    def dissolve(self, entity: Hashable) -> MembershipView:
        """Remove an entity's memberships and delete it from the model when possible."""
        snapshot = self._detach_entity(entity)
        live_entity = self._resolve_entity(self._entity_id(entity))
        if hasattr(live_entity, "_remove_from_model"):
            live_entity._remove_from_model()
        elif hasattr(live_entity, "remove"):
            live_entity.remove()
        return snapshot

    def deactivate(self, entity: Hashable) -> MembershipView:
        """Remove an entity from all memberships without deleting the object."""
        return self._detach_entity(entity)

    def at_level(
        self,
        level: int,
        *,
        root: Hashable,
        relation: RelationKey | None = "member",
    ) -> AgentSet:
        """Return agents at a containment depth below ``root``.

        Levels are structural shortest-path distances through membership edges,
        not a persistent ``agent.level`` property. ``root`` is level ``0``; its
        direct members are level ``1``. When an agent is reachable by more than
        one path, it appears only at its shallowest depth from ``root``.

        Parameters
        ----------
        level : int
            Non-negative containment depth relative to ``root``.
        root : Hashable
            Hierarchy root (live agent or unique id). Must be registered on
            this model's membership manager.
        relation : RelationKey or None, default ``"member"``
            Membership relation to traverse. ``None`` includes all relations.

        Returns:
        -------
        AgentSet
            Live agents at the requested level (empty if none).

        Raises:
        ------
        ValueError
            If ``level`` is negative or ``root`` is not in the model.

        Examples:
        --------
        >>> model.meta_agents.at_level(4, root=world)
        """
        if level < 0:
            raise ValueError(f"level must be non-negative, got {level}")

        lookup = self._live_entity_lookup()
        root_id = self._entity_id(root)
        if root_id not in lookup:
            raise ValueError(f"root {root!r} is not registered in the model")

        if level == 0:
            return AgentSet([lookup[root_id]], random=self.model.random)

        # BFS downward: group -> members. First visit is nearest depth.
        visited: set[Hashable] = {root_id}
        queue: deque[tuple[Hashable, int]] = deque([(root_id, 0)])
        at_depth: list[Any] = []

        while queue:
            group_id, depth = queue.popleft()
            if depth >= level:
                continue
            member_ids = sorted(
                self.backend.agents_of(group_id, relation=relation),
                key=str,
            )
            for member_id in member_ids:
                if member_id in visited:
                    continue
                visited.add(member_id)
                next_depth = depth + 1
                if next_depth == level:
                    entity = lookup.get(member_id)
                    if entity is not None:
                        at_depth.append(entity)
                elif next_depth < level:
                    queue.append((member_id, next_depth))

        return AgentSet(at_depth, random=self.model.random)


__all__ = [
    "MembershipEdge",
    "MembershipView",
    "MetaAgents",
]
