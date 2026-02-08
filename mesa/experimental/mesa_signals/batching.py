"""Batching and suppression of signals for mesa signals.

This module provides context managers for controlling signal dispatch:

- batch(): Buffers signals and dispatches aggregated results on exit
- suppress(): Silently drops all signals during the context

Both batch() and suppress() are used as context managers within HasObservables. They only
batch or suppress signals emitted by the instance within which they are invoked.

It also provides a functools.singledispatch aggregation system for different signal types, making
it easy for users to add aggregations for their own custom signal types::

    @aggregate.register(MySignalType)
    def _my_aggregate(signal_type, signals, value=None):
        ...

Notes:
    During a batch, computed properties may return stale cached values because
    their triggering signals are deferred. At flush, aggregated signals dispatch
    normally and mark computed properties dirty.

    During suppress, computed properties may become permanently stale because
    their triggering signals are dropped entirely.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

from .signal_types import ListSignals, ObservableSignals
from .signals_util import Message, SignalType

if TYPE_CHECKING:
    from .core import HasObservables

__all__ = ["aggregate"]


@functools.singledispatch
def aggregate(
    signal_type: SignalType, signals: list[Message], value: Any = None
) -> list[Message]:
    """Default aggregation: keep all signals as-is.

    Users can register custom aggregators for their own signal types::

        @aggregate.register(MySignalType)
        def _my_aggregate(signal_type, signals, value=None):
            ...

    Notes:
        we need signal_type as the first argument because that is
        how singledispatch can determine which handler to call.

    Args:
        signal_type: the type of signal (used for dispatch)
        signals: list of buffered Message objects
        value: the attribute value captured at the time of the first signal

    Returns:
        list of Message objects to dispatch

    """
    return signals


@aggregate.register(ObservableSignals)
def _aggregate_observable(
    signal_type: ObservableSignals,
    signals: list[Message],
    value: Any = None,
) -> list[Message]:
    """Aggregate ObservableSignals: keep old from first, new from last.

    If old == new after aggregation, return empty list (no net change).
    """
    if not signals:
        return []

    first = signals[0]
    last = signals[-1]

    old = first.additional_kwargs["old"]
    new = last.additional_kwargs["new"]

    if old == new:
        return []

    return [
        Message(
            name=first.name,
            owner=first.owner,
            signal_type=ObservableSignals.CHANGED,
            additional_kwargs={"old": old, "new": new},
        )
    ]


@aggregate.register(ListSignals)
def _aggregate_list(
    signal_type: ListSignals,
    signals: list[Message],
    value: Any = None,
) -> list[Message]:
    """Aggregate ListSignals: collapse into a single SET signal.

    Reconstructs the pre-batch list state by reverse-applying the first
    captured operation on ``value`` (the list state right after that operation).
    Reads the current list state at flush time for the ``new`` value.

    If old == new after aggregation, return empty list.
    """
    if not signals:
        return []

    # Reverse-engineer pre-batch state from first signal + value
    first = signals[0]
    kwargs = first.additional_kwargs
    current = list(value) if value is not None else []

    # reverse engineer the original value of list before first signal
    idx = kwargs.get("index")
    match first.signal_type:
        case ListSignals.APPENDED:
            old = current[:idx]
        case ListSignals.INSERTED:
            old = current[:idx] + current[idx + 1 :]
        case ListSignals.REMOVED:
            old = [*current[:idx], kwargs["old"], *current[idx:]]
        case ListSignals.REPLACED:
            old = list(current)
            old[idx] = kwargs["old"]
        case ListSignals.SET:
            old = list(kwargs.get("old", []))
        case _:
            raise ValueError("unknown list signal type")

    # Current list state at flush time
    owner = signals[-1].owner
    name = signals[-1].name
    new = list(getattr(owner, name))

    if old == new:
        return []

    return [
        Message(
            name=name,
            owner=owner,
            signal_type=ListSignals.SET,
            additional_kwargs={"old": old, "new": new},
        )
    ]


# -- Context managers ---------------------------------------------------------


class _BatchContext:
    """Context manager that batches signals for a HasObservables instance.

    Signals emitted during the batch are buffered. On exit, they are
    aggregated per observable name and dispatched as consolidated signals.

    Nesting is supported: inner batches merge their buffers into the outer
    batch, and only the outermost batch aggregates and dispatches.
    """

    __slots__ = ["_captured_values", "_previous", "buffer", "instance"]

    def __init__(self, instance: HasObservables):
        self.instance = instance
        self._previous: _BatchContext | None = None
        self.buffer: dict[
            str, list[Message]
        ] = {}  # we cannot use defaultdict here because of snapshot on first entry
        self._captured_values: dict[str, Any] = {}

    def __enter__(self):
        self._previous = self.instance._batch_context
        self.instance._batch_context = self
        return self

    def capture(self, signal: Message):
        """Capture a signal and add it to the buffer.

        On the first signal for a given observable name, stores the current
        attribute value for later use by ``aggregate``.

        Args:
            signal: the Message to buffer

        """
        name = signal.name

        if name not in self.buffer:
            self.buffer[name] = []
            current_value = getattr(signal.owner, name, None)
            try:
                # we might have to look at this again downstream
                # currently we only have signaling list and model.agents
                # both of which can be cast to a list.
                self._captured_values[name] = list(current_value)
            except TypeError:
                self._captured_values[name] = current_value

        self.buffer[name].append(signal)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.instance._batch_context = self._previous

        if exc_type is not None:
            # On exception, discard buffer
            return False

        if self._previous is not None:
            # Nested batch: merge buffer into parent
            for name, signals in self.buffer.items():
                if name not in self._previous.buffer:
                    self._previous.buffer[name] = []
                    # Transfer captured value if parent doesn't have one
                    if name in self._captured_values:
                        self._previous._captured_values[name] = self._captured_values[
                            name
                        ]
                self._previous.buffer[name].extend(signals)
        else:
            # Outermost batch: aggregate and dispatch
            self._flush()

        return False

    def _flush(self):
        """Aggregate buffered signals and dispatch them."""
        for name, signals in self.buffer.items():
            if not signals:
                continue

            signal_type = signals[0].signal_type
            value = self._captured_values.get(name, None)
            aggregated = aggregate(signal_type, signals, value=value)

            for signal in aggregated:
                key = (signal.name, signal.signal_type)
                if key in self.instance.subscribers:
                    self.instance._mesa_notify(signal)


class _SuppressContext:
    """Context manager that suppresses all signals for a HasObservables instance.

    No signals are emitted, buffered, or dispatched during suppress.

    Nesting is supported: only the outermost exit restores signal dispatch.
    """

    __slots__ = ["_previous", "instance"]

    def __init__(self, instance: HasObservables):
        self.instance = instance
        self._previous: bool = False

    def __enter__(self):
        self._previous = self.instance._suppress
        self.instance._suppress = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.instance._suppress = self._previous
        return False
