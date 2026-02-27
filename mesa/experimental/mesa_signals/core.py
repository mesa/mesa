"""Core implementation of Mesa's reactive programming system.

This module provides the foundational classes for Mesa's observable/reactive programming
functionality:

- BaseObservable: Abstract base class defining the interface for all observables
- Observable: Main class for creating observable properties that emit change signals
- computed_property: Decorator for creating properties that automatically update based on dependencies
- HasEmitters: Mixin class that enables an object to contain and manage observables
- emit: Decorator for methods that emit signals

The module implements a robust reactive system where changes to observable properties
automatically trigger updates to dependent computed values and notify subscribed
observers. This enables building models with complex interdependencies while maintaining
clean separation of concerns.
"""

from __future__ import annotations

import functools
import weakref
from collections import defaultdict, namedtuple
from collections.abc import Callable, Generator, Iterable
from typing import Any, Literal

from mesa.experimental.mesa_signals.signals_util import (
    ALL,
    Message,
    SignalType,
    create_weakref,
)

from .signal_types import ObservableSignals

__all__ = [
    "BaseObservable",
    "HasEmitters",
    "Observable",
    "computed_property",
    "emit",
]


ObservableName = str | type(ALL) | Iterable[str]
SignalSpec = str | SignalType | type(ALL) | Iterable[str | SignalType]


_hashable_signal = namedtuple("_HashableSignal", "instance name")

CURRENT_COMPUTED: ComputedState | None = None  # the current Computed that is evaluating
PROCESSING_SIGNALS: set[_hashable_signal] = set()


class BaseObservable:
    """Base class for all Observables."""

    signal_types: type[SignalType] = SignalType

    def __init__(self, fallback_value=None):
        """Initialize a BaseObservable."""
        super().__init__()
        self.public_name: str
        self.private_name: str
        self.fallback_value = fallback_value

    def __get__(self, instance: HasEmitters, owner):  # noqa: D105
        value = getattr(instance, self.private_name)

        # fixme this makes signaling list part of computed
        if CURRENT_COMPUTED is not None:
            # there is a computed dependent on this Observable, so let's add
            # this Observable as a parent
            CURRENT_COMPUTED._add_parent(instance, self.public_name, value)

            # fixme, this can be done more cleanly
            #  problem here is that we cannot use self (i.e., the observable), we need to add the instance as well
            PROCESSING_SIGNALS.add(_hashable_signal(instance, self.public_name))

        return value

    def __set_name__(self, owner: HasEmitters, name: str):  # noqa: D105
        self.public_name = name
        self.private_name = f"_{name}"

    def __set__(self, instance: HasEmitters, value):  # noqa: D105
        # If no one is listening, Avoid overhead of fetching old value and
        # creating Message object.
        if not instance._has_subscribers(self.public_name, ObservableSignals.CHANGED):
            return
        change_signal = self.signal_types(
            "changed"
        )  # look up "changed" in the descriptor's enum

        # this only emits an on change signal, subclasses need to specify
        # this in more detail
        instance.notify(
            self.public_name,
            change_signal,
            old=value,
            new=getattr(instance, self.private_name, self.fallback_value),
        )

    def __str__(self):  # noqa: D105
        return f"{self.__class__.__name__}: {self.public_name}"


class Observable(BaseObservable):
    """Observable descriptor.

    An observable is an attribute that emits ObservableSignals.CHANGED whenever it is changed to a different value.

    """

    # fixme: when we go to 3.13, we might explore changing this into a property
    #    instead of descriptor, which is likely to be more performant
    signal_types = ObservableSignals

    def __set__(self, instance: HasEmitters, value):  # noqa D103
        if (
            CURRENT_COMPUTED is not None
            and _hashable_signal(instance, self.public_name) in PROCESSING_SIGNALS
        ):
            raise ValueError(
                f"cyclical dependency detected: Computed({CURRENT_COMPUTED.name}) tries to change "
                f"{instance.__class__.__name__}.{self.public_name} while also being dependent on it"
            )

        send_notify = False
        old_value = None
        if instance._has_subscribers(self.public_name, ObservableSignals.CHANGED):
            old_value = getattr(instance, self.private_name, None)
            if old_value != value:
                send_notify = True
        setattr(instance, self.private_name, value)

        if send_notify:
            instance.notify(
                self.public_name,
                ObservableSignals.CHANGED,
                old=old_value,
                new=value,
            )
            PROCESSING_SIGNALS.clear()  # we have notified our children, so we can clear this out


class ComputedState:
    """Internal class to hold the state of a computed property for a specific instance."""

    __slots__ = ["__weakref__", "func", "is_dirty", "name", "owner", "parents", "value"]

    def __init__(self, owner: HasEmitters, name: str, func: Callable):
        self.owner = owner
        self.name = name
        self.func = func
        self.value = None
        self.is_dirty = True
        self.parents: weakref.WeakKeyDictionary[HasEmitters, dict[str, Any]] = (
            weakref.WeakKeyDictionary()
        )

    def _set_dirty(self, signal):
        if not self.is_dirty:
            self.is_dirty = True
            self.owner.notify(self.name, ObservableSignals.CHANGED, old=self.value)

    def _add_parent(self, parent: HasEmitters, name: str, current_value: Any) -> None:
        """Add a parent Observable.

        Args:
            parent: the HasEmitters instance to which the Observable belongs
            name: the public name of the Observable
            current_value: the current value of the Observable

        """
        parent.observe(name, ALL, self._set_dirty)

        try:
            self.parents[parent][name] = current_value
        except KeyError:
            self.parents[parent] = {name: current_value}

    def _remove_parents(self):
        """Remove all parent Observables."""
        # we can unsubscribe from everything on each parent
        for parent in self.parents:
            parent.unobserve(ALL, ALL, self._set_dirty)
        self.parents.clear()


class ComputedProperty(property):
    """A custom property class to identify computed properties."""

    signal_types = ObservableSignals


def computed_property(func: Callable) -> property:
    """Decorator to create a computed property.

    Acts like @property, but automatically tracks dependencies (Observables)
    accessed during the function execution.
    """
    key = f"_computed_{func.__name__}"

    @functools.wraps(func)
    def wrapper(self: HasEmitters):
        global CURRENT_COMPUTED  # noqa: PLW0603

        if not hasattr(self, key):
            state = ComputedState(self, func.__name__, func)
            setattr(self, key, state)
        else:
            state = getattr(self, key)

        if state.is_dirty:
            changed = False

            # Check if parents actually changed
            if not state.parents:
                changed = True
            else:
                for parent, observations in state.parents.items():
                    if parent is None:
                        changed = True
                        break
                    for attr, old_val in observations.items():
                        current_val = getattr(parent, attr)
                        if current_val != old_val:
                            changed = True
                            break
                    if changed:
                        break

            if changed:
                state._remove_parents()

                old = CURRENT_COMPUTED
                CURRENT_COMPUTED = state

                try:
                    state.value = func(self)
                except Exception as e:
                    raise e
                finally:
                    CURRENT_COMPUTED = old

            state.is_dirty = False

        if CURRENT_COMPUTED is not None:
            CURRENT_COMPUTED._add_parent(self, func.__name__, state.value)

        return state.value

    return ComputedProperty(wrapper)


class HasEmitters:
    """HasEmitters class.

    Attributes:
        subscribers: mapping of observables/emitters and signal type to subscribers
        observables: mapping of observables/emitters to their available signal types

    HasEmitters automatically discovers the observables/emitters defined on the class.

    """

    # we can't use a weakset here because it does not handle bound methods correctly
    # also, a list is faster for our use case
    subscribers: dict[
        tuple[str, SignalType], list[weakref.ref]
    ]  # (observable_name, signal_type) -> list of weakref subscribers
    observables: dict[str, type[SignalType] | frozenset[SignalType]]

    def __init_subclass__(cls, **kwargs):
        """Initialize a HasEmitters subclass."""
        super().__init_subclass__(**kwargs)
        cls.observables = dict(descriptor_generator(cls))

    def __init__(self, *args, **kwargs) -> None:
        """Initialize a HasEmitters."""
        super().__init__(*args, **kwargs)
        self.subscribers = defaultdict(list)
        self._batch_context = None
        self._suppress = False

    def _has_subscribers(self, name: str, signal_type: str | SignalType) -> bool:
        """Check if there are any subscribers for a given observable and signal type."""
        key = (name, signal_type)
        if key not in self.subscribers:
            return False
        return len(self.subscribers[key]) > 0

    def observe(
        self,
        observable_name: ObservableName,
        signal_type: SignalSpec,
        handler: Callable,
    ):
        """Subscribe to the Observable <name> for signal_type.

        Args:
            observable_name: name of the Observable to subscribe to
            signal_type: the type of signal on the Observable to subscribe to
            handler: the handler to call

        Raises:
            ValueError: if the Observable <name> is not registered or if the Observable
            does not emit the given signal_type

        """
        names = self._process_name(observable_name)
        target_signals = self._process_signal_type(signal_type)

        for name in names:
            if name not in self.observables:
                raise ValueError(
                    f"you are trying to subscribe to {name}, but this Observable is not known"
                )

            signal_types = target_signals or self.observables[name]

            for st in signal_types:
                if st not in self.observables[name]:
                    raise ValueError(
                        f"you are trying to subscribe to a signal of {st} "
                        f"on Observable {name}, which does not emit this signal_type"
                    )

            ref = create_weakref(handler)
            for st in signal_types:
                self.subscribers[(name, st)].append(ref)

    def unobserve(
        self,
        observable_name: ObservableName,
        signal_type: SignalSpec,
        handler: Callable,
    ):
        """Unsubscribe to the Observable <name> for signal_type.

        Args:
            observable_name: name of the Observable to unsubscribe from
            signal_type: the type of signal on the Observable to unsubscribe to
            handler: the handler that is unsubscribing

        """
        names = self._process_name(observable_name)
        target_signals = self._process_signal_type(signal_type)

        for name in names:
            # we need to do this here because signal types might
            # differ for name so for each name we need to check
            signal_types = target_signals or self.observables[name]

            for st in signal_types:
                key = (name, st)
                if key in self.subscribers:
                    remaining = []
                    for ref in self.subscribers[key]:
                        if subscriber := ref():  # noqa: SIM102
                            if subscriber != handler:
                                remaining.append(ref)

                    if remaining:
                        self.subscribers[key] = remaining
                    else:
                        del self.subscribers[key]

    def clear_all_subscriptions(self, name: ObservableName):
        """Clears all subscriptions for the observable <name>.

        if name is ALL, all subscriptions are removed

        Args:
            name: name of the Observable to unsubscribe for all signal types

        """
        if name is ALL:
            self.subscribers.clear()
        elif isinstance(name, str):
            keys_to_remove = [k for k in self.subscribers if k[0] == name]
            for k in keys_to_remove:
                del self.subscribers[k]
        else:
            for n in name:
                keys_to_remove = [k for k in self.subscribers if k[0] == n]
                for k in keys_to_remove:
                    del self.subscribers[k]
                # ignore when unsubscribing to Observables that have no subscription

    def notify(
        self,
        observable: str,
        signal_type: str | SignalType,
        **kwargs,
    ):
        """Emit a signal.

        Args:
            observable: the public name of the observable emitting the signal
            signal_type: the type of signal to emit
            kwargs: additional keyword arguments to include in the signal

        """
        if self._suppress:
            return

        if self._batch_context is not None:
            signal = Message(
                name=observable,
                owner=self,
                signal_type=signal_type,
                additional_kwargs=kwargs,
            )
            self._batch_context.capture(signal)
            return

        # because we are using a list of subscribers
        # we should update this list to subscribers that are still alive
        key = (observable, signal_type)

        if key not in self.subscribers:
            return

        signal = Message(
            name=observable,
            owner=self,
            signal_type=signal_type,
            additional_kwargs=kwargs,
        )

        self._mesa_notify(signal)

    def _mesa_notify(self, signal: Message):
        """Send out the signal.

        Args:
        signal: the signal

        Notes:
        signal must contain name and type attributes because this is how observers are stored.

        """
        # we put this into a helper method, so we can emit signals with other fields
        # than the default ones in notify.
        observable = signal.name
        signal_type = signal.signal_type
        key = (observable, signal_type)

        observers = self.subscribers[key]
        active_observers = []
        for observer in observers:
            if active_observer := observer():
                active_observer(signal)
                active_observers.append(observer)
            # use iteration to also remove inactive observers

        if active_observers:
            self.subscribers[key] = active_observers
        else:
            del self.subscribers[key]

    def batch(self):
        """Return a context manager that batches signals.

        Signals emitted during the batch are buffered and aggregated on exit.
        Nested batches merge into the outer batch; only the outermost dispatches.

        Note:
            Computed properties may return stale cached values during the batch.
            They will be updated when aggregated signals are dispatched on exit.

        """
        from .batching import _BatchContext  # noqa: PLC0415

        return _BatchContext(self)

    def suppress(self):
        """Return a context manager that suppresses all signals.

        No signals are emitted, buffered, or dispatched during suppression.

        Note:
            Computed properties may become permanently stale because their
            triggering signals are dropped entirely.

        """
        from .batching import _SuppressContext  # noqa: PLC0415

        return _SuppressContext(self)

    def _process_name(self, name: ObservableName) -> Iterable[str]:
        """Convert name to an iterable of observable names."""
        if name is ALL:
            return self.observables.keys()
        elif isinstance(name, str):
            return [name]
        else:
            return name

    def _process_signal_type(
        self, signal_type: SignalSpec
    ) -> Iterable[SignalType] | None:
        """Convert signal_type to an iterable of signal types."""
        if signal_type is ALL:
            return None  # None is used to indicate all signal types
        elif isinstance(signal_type, str):
            return [signal_type]
        else:
            return signal_type


def descriptor_generator(
    cls,
) -> Generator[tuple[str, type[SignalType] | frozenset[SignalType]]]:
    """Yield the name and signal_types for each Observable defined on cls.

    This handles both legacy BaseObservable descriptors and new @computed_properties.
    """
    emitters = defaultdict(set)
    for base in cls.__mro__:
        base_dict = vars(base)

        for name, entry in base_dict.items():
            if isinstance(entry, ComputedProperty):
                yield name, entry.signal_types
            elif isinstance(entry, BaseObservable):
                yield entry.public_name, entry.signal_types
            elif hasattr(entry, "_mesa_signal_emitter"):
                observable_name, signal = entry._mesa_signal_emitter
                emitters[observable_name].add(signal)

    for name, signals in emitters.items():
        yield name, frozenset(signals)


def emit(observable_name, signal_to_emit, when: Literal["before", "after"] = "after"):
    """Decorator to emit a signal before or after the call to method.

    Args:
        observable_name: the name of the observable that emits the signal
        signal_to_emit: the signal to emit
        when: whether to emit the signal before or after the function call.

    This only works on HasEmitters subclasses.

    """

    def inner(method):
        """Wrap func."""
        if when == "before":

            @functools.wraps(method)
            def wrapper(self, *args, **kwargs):
                self.notify(observable_name, signal_to_emit, args=args, **kwargs)
                return method(self, *args, **kwargs)
        else:

            @functools.wraps(method)
            def wrapper(self, *args, **kwargs):
                ret = method(self, *args, **kwargs)
                self.notify(observable_name, signal_to_emit, args=args, **kwargs)
                return ret

        wrapper._mesa_signal_emitter = observable_name, signal_to_emit

        return wrapper

    return inner
