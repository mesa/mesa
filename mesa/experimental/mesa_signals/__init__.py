"""Mesa Signals (Observables) package that provides reactive programming capabilities.

This package enables tracking changes to properties and state in Mesa models through a
reactive programming paradigm. It enables building models where components can observe
and react to changes in other components' state.

The package provides the core Observable classes and utilities needed to implement
reactive patterns in agent-based models. This includes capabilities for watching changes
to attributes, computing derived values, and managing collections that emit signals
when modified.
"""

from .batching import aggregate
from .core import (
    HasObservables,
    Observable,
    computed_property,
    emit,
)
from .observable_collections import ObservableList
from .signal_types import ListSignals, ModelSignals, ObservableSignals
from .signals_util import ALL, Message, SignalType

__all__ = [
    "ALL",
    "HasObservables",
    "ListSignals",
    "Message",
    "ModelSignals",
    "Observable",
    "ObservableList",
    "ObservableSignals",
    "SignalType",
    "aggregate",
    "computed_property",
    "emit",
]
