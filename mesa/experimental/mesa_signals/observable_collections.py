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
        setattr(
            instance,
            self.private_name,
            SignalingList(value, instance, self.public_name),
        )
        instance.notify(
            self.public_name,
            ListSignals.SET,
            old=getattr(instance, self.private_name, self.fallback_value),
            new=value,
        )


class SignalingList(MutableSequence[Any]):
    """A basic lists that emits signals on changes."""

    __slots__ = ["data", "name", "owner"]

    def __init__(self, iterable: Iterable, owner: HasObservables, name: str):
        """Initialize a SignalingList.

        Args:
            iterable: initial values in the list
            owner: the HasObservables instance on which this list is defined
            name: the attribute name to which  this list is assigned

        """
        self.owner: HasObservables = owner
        self.name: str = name
        self.data = list(iterable)

    def __setitem__(self, index: int, value: Any) -> None:
        """Set item to index.

        Args:
            index: the index to set item to
            value: the item to set

        """
        old_value = self.data[index]
        self.data[index] = value
        self.owner.notify(
            self.name, ListSignals.REPLACED, index=index, old=old_value, new=value
        )

    def __delitem__(self, index: int) -> None:
        """Delete item at index.

        Args:
            index: The index of the item to remove

        """
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
        self.data.insert(index, value)
        self.owner.notify(self.name, ListSignals.INSERTED, index=index, new=value)

    def append(self, value):
        """Insert value at index.

        Args:
            value: the value to append

        """
        index = len(self.data)
        self.data.append(value)
        self.owner.notify(self.name, ListSignals.APPENDED, index=index, new=value)

    def __str__(self):
        return self.data.__str__()

    def __repr__(self):
        return self.data.__repr__()
