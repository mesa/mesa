"""Observable collection types that emit signals when modified.

This module extends Mesa's reactive programming capabilities to collection types like
lists. Observable collections emit signals when items are added, removed, or modified,
allowing other components to react to changes in the collection's contents.

The module provides:
- ListSignalType: Enum defining signal types for list collections
- ObservableList: A list descriptor that emits signals on modifications
- SignalingList: The underlying list implementation that manages signal emission

These classes enable building models where components need to track and react to
changes in collections of agents, resources, or other model elements.
"""

from collections.abc import Iterable, MutableSequence
from enum import Enum
from typing import Any

from .mesa_signal import BaseObservable, HasObservables

__all__ = [
    "ListSignalType",
    "ObservableList",
]


class ListSignalType(str, Enum):
    """Enumeration of signal types that observable lists can emit.

    Provides list-specific signal types with IDE autocomplete and type safety.
    Inherits from str for backward compatibility with existing string-based code.
    Includes all list-specific signals (INSERT, APPEND, REMOVE, REPLACE) plus
    the base CHANGE signal inherited from the observable protocol.

    Note on Design:
        This enum does NOT extend SignalType because Python Enums cannot be extended
        once they have members defined. Instead, we include CHANGE as a member here
        to maintain compatibility. The string inheritance provides value equality:
        ListSignalType.CHANGE == SignalType.CHANGE == "change" (all True).

    Attributes:
        CHANGE: Emitted when the list itself is replaced/assigned.
        INSERT: Emitted when an item is inserted into the list.
        APPEND: Emitted when an item is appended to the list.
        REMOVE: Emitted when an item is removed from the list.
        REPLACE: Emitted when an item is replaced/modified in the list.

    Examples:
        >>> from mesa.experimental.mesa_signals import ObservableList, HasObservables, ListSignalType
        >>> class MyModel(HasObservables):
        ...     items = ObservableList()
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.items = []
        >>> model = MyModel()
        >>> model.observe("items", ListSignalType.INSERT, lambda s: print(f"Inserted {s.new}"))
        >>> model.items.insert(0, "first")
        Inserted first

    Note:
        String-based signal types are still supported for backward compatibility:
        >>> model.observe("items", "insert", handler)  # Still works
        Also compatible with SignalType.CHANGE since both equal "change" as strings.
    """

    CHANGE = "change"
    INSERT = "insert"
    APPEND = "append"
    REMOVE = "remove"
    REPLACE = "replace"

    def __str__(self):
        """Return the string value of the signal type."""
        return self.value


class ObservableList(BaseObservable):
    """An ObservableList that emits signals on changes to the underlying list."""

    def __init__(self):
        """Initialize the ObservableList."""
        super().__init__()
        # Use all members of ListSignalType enum
        self.signal_types: set = set(ListSignalType)
        self.fallback_value = []

    def __set__(self, instance: "HasObservables", value: Iterable):
        """Set the value of the descriptor attribute.

        Args:
            instance: The instance on which to set the attribute.
            value: The value to set the attribute to.

        """
        super().__set__(instance, value)
        setattr(
            instance,
            self.private_name,
            SignalingList(value, instance, self.public_name),
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
            self.name, old_value, value, ListSignalType.REPLACE, index=index
        )

    def __delitem__(self, index: int) -> None:
        """Delete item at index.

        Args:
            index: The index of the item to remove

        """
        old_value = self.data
        del self.data[index]
        self.owner.notify(
            self.name, old_value, None, ListSignalType.REMOVE, index=index
        )

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
        self.owner.notify(self.name, None, value, ListSignalType.INSERT, index=index)

    def append(self, value):
        """Insert value at index.

        Args:
            index: the index to insert value into
            value: the value to insert

        """
        index = len(self.data)
        self.data.append(value)
        self.owner.notify(self.name, None, value, ListSignalType.APPEND, index=index)

    def __str__(self):
        return self.data.__str__()

    def __repr__(self):
        return self.data.__repr__()
