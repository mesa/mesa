"""Explicit entity indexing for atomic agents and meta-agents.

The meta-agent rewrite currently leans on ``unique_id`` as convenient lookup
key, but that is still an implicit identity scheme. This module provides a
small registry that assigns each tracked entity a stable ``entity_id`` and keeps
its current object reference, kind, and ``unique_id`` in sync.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable
from dataclasses import dataclass
from typing import Any, Literal

EntityKind = Literal["atomic", "meta"]


@dataclass(slots=True)
class EntityRecord:
    """Snapshot of one registered entity."""

    entity_id: int
    entity: Any
    kind: EntityKind
    unique_id: Hashable | None

    @property
    def class_name(self) -> str:
        """Return the concrete class name for debugging and display."""
        return self.entity.__class__.__name__


class EntityIndex:
    """Stable registry for entities participating in meta-agent workflows."""

    def __init__(self) -> None:
        """Create an empty entity index."""
        self._next_entity_id = 1
        self._records_by_id: dict[int, EntityRecord] = {}
        self._entity_id_by_object: dict[int, int] = {}
        self._entity_ids_by_unique_id: dict[Hashable, set[int]] = defaultdict(set)

    def _infer_kind(self, entity: Any) -> EntityKind:
        """Infer the entity kind when a caller does not provide one."""
        return "meta" if hasattr(entity, "_constituting_set") else "atomic"

    def _set_entity_id(self, entity: Any, entity_id: int) -> None:
        """Persist the stable entity id on the entity object when possible."""
        try:
            setattr(entity, "entity_id", entity_id)
        except Exception:
            # Some external objects may not allow attribute assignment.
            # The registry still tracks them via object identity.
            pass

    def _sync_unique_id_alias(
        self, record: EntityRecord, unique_id: Hashable | None
    ) -> None:
        """Keep the reverse lookup table aligned with the entity's unique_id."""
        if record.unique_id == unique_id:
            return

        if record.unique_id is not None:
            ids = self._entity_ids_by_unique_id.get(record.unique_id)
            if ids is not None:
                ids.discard(record.entity_id)
                if not ids:
                    del self._entity_ids_by_unique_id[record.unique_id]

        record.unique_id = unique_id
        if unique_id is not None:
            self._entity_ids_by_unique_id[unique_id].add(record.entity_id)

    def register(self, entity: Any, kind: EntityKind | None = None) -> EntityRecord:
        """Register an entity and return its stable record.

        Re-registering an existing object is idempotent. The stored record is
        refreshed so callers can update the kind or pick up a changed
        ``unique_id`` without changing the stable ``entity_id``.
        """
        object_key = id(entity)
        unique_id = getattr(entity, "unique_id", None)
        existing_entity_id = self._entity_id_by_object.get(object_key)

        if existing_entity_id is not None:
            record = self._records_by_id[existing_entity_id]
            if kind is not None:
                record.kind = kind
            self._sync_unique_id_alias(record, unique_id)
            record.entity = entity
            self._set_entity_id(entity, record.entity_id)
            return record

        entity_id = self._next_entity_id
        self._next_entity_id += 1
        record = EntityRecord(
            entity_id=entity_id,
            entity=entity,
            kind=kind or self._infer_kind(entity),
            unique_id=unique_id,
        )
        self._records_by_id[entity_id] = record
        self._entity_id_by_object[object_key] = entity_id
        self._set_entity_id(entity, entity_id)

        if unique_id is not None:
            self._entity_ids_by_unique_id[unique_id].add(entity_id)

        return record

    def entity_id_for(self, entity: Any) -> int:
        """Return the stable entity id for a registered object."""
        record = self.record_for(entity)
        return record.entity_id

    def record_for(self, entity_or_id: Any) -> EntityRecord:
        """Return the record for an entity object, entity_id, or unique_id.

        ``entity_id`` lookup is preferred. ``unique_id`` lookup is supported for
        compatibility, but it is secondary to the explicit identity layer.
        """
        if isinstance(entity_or_id, EntityRecord):
            return entity_or_id

        if isinstance(entity_or_id, int) and entity_or_id in self._records_by_id:
            return self._records_by_id[entity_or_id]

        object_key = id(entity_or_id)
        entity_id = self._entity_id_by_object.get(object_key)
        if entity_id is not None:
            return self._records_by_id[entity_id]

        raise KeyError(f"Unknown entity or entity id: {entity_or_id!r}")

    def entity_for(self, entity_or_id: Any) -> Any:
        """Return the live entity object for a record, object, or id."""
        return self.record_for(entity_or_id).entity

    def kind_for(self, entity_or_id: Any) -> EntityKind:
        """Return the registered kind for an entity."""
        return self.record_for(entity_or_id).kind

    def contains(self, entity_or_id: Any) -> bool:
        """Return whether the registry knows about the given entity."""
        try:
            self.record_for(entity_or_id)
        except KeyError:
            return False
        return True

    def entities(self, kind: EntityKind | None = None) -> list[Any]:
        """Return the live entities, optionally filtered by kind."""
        return [record.entity for record in self.records(kind=kind)]

    def records(self, kind: EntityKind | None = None) -> list[EntityRecord]:
        """Return all records, optionally filtered by kind."""
        records = list(self._records_by_id.values())
        if kind is None:
            return records
        return [record for record in records if record.kind == kind]

    def remove(self, entity_or_id: Any) -> None:
        """Remove an entity from the registry if it exists."""
        try:
            record = self.record_for(entity_or_id)
        except KeyError:
            return

        self._records_by_id.pop(record.entity_id, None)
        self._entity_id_by_object.pop(id(record.entity), None)
        if record.unique_id is not None:
            ids = self._entity_ids_by_unique_id.get(record.unique_id)
            if ids is not None:
                ids.discard(record.entity_id)
                if not ids:
                    del self._entity_ids_by_unique_id[record.unique_id]

    def assert_invariants(self) -> None:
        """Verify the registry's forward and reverse indexes are aligned."""
        for entity_id, record in self._records_by_id.items():
            assert record.entity_id == entity_id
            assert self._entity_id_by_object[id(record.entity)] == entity_id
            if record.unique_id is not None:
                assert entity_id in self._entity_ids_by_unique_id[record.unique_id]

        for unique_id, entity_ids in self._entity_ids_by_unique_id.items():
            for entity_id in entity_ids:
                assert self._records_by_id[entity_id].unique_id == unique_id

    def __getstate__(self) -> dict[str, Any]:
        """Return a pickle-friendly snapshot of the registry state."""
        return {
            "_next_entity_id": self._next_entity_id,
            "_records_by_id": self._records_by_id,
            "_entity_ids_by_unique_id": {
                unique_id: set(entity_ids)
                for unique_id, entity_ids in self._entity_ids_by_unique_id.items()
            },
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the registry state and rebuild object-id indexes."""
        self._next_entity_id = state["_next_entity_id"]
        self._records_by_id = state["_records_by_id"]
        self._entity_id_by_object = {}
        self._entity_ids_by_unique_id = defaultdict(set)

        for unique_id, entity_ids in state["_entity_ids_by_unique_id"].items():
            self._entity_ids_by_unique_id[unique_id].update(entity_ids)

        for entity_id, record in self._records_by_id.items():
            self._entity_id_by_object[id(record.entity)] = entity_id


def ensure_entity_index(model: Any) -> EntityIndex:
    """Return the model's entity index, creating it lazily when needed."""
    entity_index = getattr(model, "entity_index", None)
    if entity_index is None:
        entity_index = EntityIndex()
        setattr(model, "entity_index", entity_index)
    return entity_index
