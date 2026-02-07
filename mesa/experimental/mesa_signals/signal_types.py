"""Signal types."""

from .signals_util import SignalType


class ObservableSignals(SignalType):
    """Enumeration of signal types that observables can emit.

    This enum provides type-safe signal type definitions with IDE autocomplete support.
    Inherits from str for backward compatibility with existing string-based code.

    Attributes:
        CHANGE: Emitted when an observable's value changes.

    Examples:
        >>> from mesa.experimental.mesa_signals import Observable, HasObservables, SignalType
        >>> class MyModel(HasObservables):
        ...     value = Observable()
        ...     def __init__(self):
        ...         super().__init__()
        ...         self._value = 0
        >>> model = MyModel()
        >>> model.observe("value", ObservableSignals.CHANGE, lambda s: print(s.new))
        >>> model.value = 10
        10

    Note:
        String-based signal types are still supported for backward compatibility:
        >>> model.observe("value", "change", handler)  # Still works
    """

    CHANGED = "changed"

    def __str__(self):
        """Return the string value of the signal type."""
        return self.value


class ListSignals(SignalType):
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
        >>> from mesa.experimental.mesa_signals import ObservableList, HasObservables, ListSignals
        >>> class MyModel(HasObservables):
        ...     items = ObservableList()
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.items = []
        >>> model = MyModel()
        >>> model.observe("items", ListSignals.INSERT, lambda s: print(f"Inserted {s.new}"))
        >>> model.items.insert(0, "first")
        Inserted first

    Note:
        String-based signal types are still supported for backward compatibility:
        >>> model.observe("items", "insert", handler)  # Still works
        Also compatible with SignalType.CHANGE since both equal "change" as strings.
    """

    SET = "set"
    INSERTED = "inserted"
    APPENDED = "appended"
    REMOVED = "removed"
    REPLACED = "replaced"


class ModelSignals(SignalType):
    """Signal types for model-level events."""

    AGENT_ADDED = "agent_added"
    AGENT_REMOVED = "agent_removed"
