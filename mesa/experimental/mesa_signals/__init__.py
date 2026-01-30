"""Mesa Signals (Observables) package that provides reactive programming capabilities.

This package enables tracking changes to properties and state in Mesa models through a
reactive programming paradigm. It enables building models where components can observe
and react to changes in other components' state.

The package provides the core Observable classes and utilities needed to implement
reactive patterns in agent-based models. This includes capabilities for watching changes
to attributes, computing derived values, and managing collections that emit signals
when modified.
"""

from .mesa_signal import (
    HasObservables,
    Observable,
    ObservableSignals,
    computed_property,
    emit,
)
from .observable_collections import ListSignals, ObservableList
from .signals_util import Message, SignalType, ALL

__all__ = [
    "ALL",
    "HasObservables",
    "ListSignals",
    "Message",
    "Observable",
    "ObservableList",
    "ObservableSignals",
    "SignalType",
    "computed_property",
    "emit",
]
