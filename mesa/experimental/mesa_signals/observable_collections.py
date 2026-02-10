"""Observable collection types that emit signals when modified.

This module extends Mesa's reactive programming capabilities to collection types like
lists. Observable collections emit signals when items are added, removed, or modified,
allowing other components to react to changes in the collection's contents.

The module provides:
- ListSignals: Enum defining signal types for list collections
- ObservableList: A list descriptor that emits signals on modifications
- SignalingList: The underlying list implementation that manages signal emission

These classes enable building models where components need to track and react to
changes in collections of agents, resources, or other model elements.
"""

from collections.abc import Iterable, MutableSequence
from typing import Any

from .core import BaseObservable, HasObservables
from .signal_types import ListSignals
from .signals_util import SignalType

__all__ = [
    "ObservableList",
]


class ObservableList(BaseObservable):
    """An ObservableList that emits signals on changes to the underlying list."""

    signal_types: type[SignalType] = ListSignals

    def __init__(self):
        """Initialize the ObservableList."""
        super().__init__(fallback_value=[])

    def __set__(self, instance: "HasObservables", value: Iterable):
        """Set the value of the descriptor attribute.

        Args:
            instance: The instance on which to set the attribute.
            value: The value to set the attribute to.

        """
        old_value = getattr(instance, self.private_name, self.fallback_value)
        # Snapshot into batch context before replacing the list
        ctx = instance._batch_context
        if ctx is not None:
            old_value = [] if old_value is None else list(old_value)
            ctx.capture_original_value_once(self.public_name, old_value)
        setattr(
            instance,
            self.private_name,
            SignalingList(value, instance, self.public_name),
        )
        instance.notify(
            self.public_name,
            ListSignals.SET,
            old=old_value,
            new=value,
        )


class SignalingList(MutableSequence[Any]):
    """A standard list that emits signals on changes to the underlying list."""

    __slots__ = ["data", "name", "owner"]

    def __init__(self, iterable: Iterable, owner: HasObservables, name: str):
        """Initialize a SignalingList.

        Args:
            iterable: initial values in the list
            owner: the HasObservables instance on which this list is defined
            name: the attribute name to which this list is assigned

        """
        self.owner: HasObservables = owner
        self.name: str = name
        self.data = list(iterable)

    def _snapshot_if_batching(self):
        """Snapshot list state into the batch context before the first mutation."""
        ctx = self.owner._batch_context
        if ctx is not None:
            ctx.capture_original_value_once(self.name, list(self.data))

    def __setitem__(self, index: int | slice, value: Any) -> None:
        """Set item(s) by index or slice.

        Args:
            index: the index or slice to set
            value: the item (or iterable for slices) to set

        """
        self._snapshot_if_batching()
        if isinstance(index, slice):
            # this resolves negative numbers in slice
            index = slice(*index.indices(len(self.data)))
            old_value = self.data[index]
            new_value = list(value)
            self.data[index] = new_value
            self.owner.notify(
                self.name,
                ListSignals.REPLACED,
                index=index,
                old=old_value,
                new=new_value,
            )
        else:
            if index < 0:
                index += len(self.data)
            old_value = self.data[index]
            self.data[index] = value
            self.owner.notify(
                self.name, ListSignals.REPLACED, index=index, old=old_value, new=value
            )

    def __delitem__(self, index: int | slice) -> None:
        """Delete item(s) by index or slice.

        Args:
            index: the index or slice to delete

        """
        self._snapshot_if_batching()
        if isinstance(index, slice):
            # this resolves negative numbers in slice
            index = slice(*index.indices(len(self.data)))
            old_value = self.data[index]
        else:
            if index < 0:
                index += len(self.data)
            old_value = self.data[index]
        del self.data[index]
        self.owner.notify(self.name, ListSignals.REMOVED, index=index, old=old_value)

    def __getitem__(self, index) -> Any:
        """Get item at index.

        Args:
            index: The index of the item to retrieve

        Returns:
            the item at index
        """
        return self.data[index]

    def __len__(self) -> int:
        """Return the length of the list."""
        return len(self.data)

    def insert(self, index, value):
        """Insert value at index.

        Args:
            index: the index to insert value into
            value: the value to insert

        """
        self._snapshot_if_batching()
        # Normalize before insert: clamp to [0, len] to match list.insert behavior
        if index < 0:
            index = max(0, index + len(self.data))
        elif index > len(self.data):
            index = len(self.data)
        self.data.insert(index, value)
        self.owner.notify(self.name, ListSignals.INSERTED, index=index, new=value)

    def append(self, value):
        """Append value to list.

        Args:
            value: the value to append

        """
        self._snapshot_if_batching()
        index = len(self.data)
        self.data.append(value)
        self.owner.notify(self.name, ListSignals.APPENDED, index=index, new=value)

    def __str__(self):
        return self.data.__str__()

    def __repr__(self):
        return self.data.__repr__()
