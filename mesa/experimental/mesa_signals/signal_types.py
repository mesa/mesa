"""Signal types."""

from .signals_util import SignalType


class ObservableSignals(SignalType):
    """Enumeration of signal types that observables can emit.

    This enum provides type-safe signal type definitions with IDE autocomplete support.
    Inherits from str for backward compatibility with existing string-based code.

    Attributes:
        CHANGED: Emitted when an observable's value changes.

    Examples:
        >>> from mesa.experimental.mesa_signals import Observable, HasEmitters, SignalType
        >>> class MyModel(HasEmitters):
        ...     value = Observable()
        ...     def __init__(self):
        ...         super().__init__()
        ...         self._value = 0
        >>> model = MyModel()
        >>> model.observe("value", ObservableSignals.CHANGED, lambda s: print(s.new))
        >>> model.value = 10
        10

    Note:
        String-based signal types are still supported for backward compatibility:
        >>> model.observe("value", "changed", handler)  # Still works
    """

    CHANGED = "changed"

    def __str__(self):
        """Return the string value of the signal type."""
        return self.value


class ListSignals(SignalType):
    """Enumeration of signal types that observable lists can emit.

    Provides list-specific signal types with IDE autocomplete and type safety.
    Inherits from str for backward compatibility with existing string-based code.
    Includes all list-specific signals (INSERTED, APPENDED, REMOVED, REPLACED) plus
    a SET signal for when the entire list is modified.

    Attributes:
        SET: Emitted when the list itself is replaced/assigned.
        INSERTED: Emitted when an item is inserted into the list.
        APPENDED: Emitted when an item is appended to the list.
        REMOVED: Emitted when an item is removed from the list.
        REPLACED: Emitted when an item is replaced/modified in the list.

    Examples:
        >>> from mesa.experimental.mesa_signals import ObservableList, HasEmitters, ListSignals
        >>> class MyModel(HasEmitters):
        ...     items = ObservableList()
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.items = []
        >>> model = MyModel()
        >>> model.observe("items", ListSignals.INSERTED, lambda m: print(f"Inserted {m.kwargs['new']}"))
        >>> model.items.insert(0, "first")
        Inserted first

    Note:
        String-based signal types are still supported for backward compatibility:
        >>> model.observe("items", "inserted", handler)  # Still works
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
