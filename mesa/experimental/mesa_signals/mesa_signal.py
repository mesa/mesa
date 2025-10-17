"""Core implementation of Mesa's reactive programming system.

This module provides the foundational classes for Mesa's observable/reactive programming
functionality:

- BaseObservable: Abstract base class defining the interface for all observables
- Observable: Main class for creating observable properties that emit change signals
- Computable: Class for properties that automatically update based on other observables
- HasObservables: Mixin class that enables an object to contain and manage observables
- All: Helper class for subscribing to all signals from an observable

The module implements a robust reactive system where changes to observable properties
automatically trigger updates to dependent computed values and notify subscribed
observers. This enables building models with complex interdependencies while maintaining
clean separation of concerns.
"""

from __future__ import annotations

import contextlib
import functools
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from collections.abc import Callable
from typing import Any

from mesa.experimental.mesa_signals.signals_util import AttributeDict, create_weakref

__all__ = ["All", "Computable", "ContinuousObservable", "HasObservables", "Observable"]

_hashable_signal = namedtuple("_HashableSignal", "instance name")

CURRENT_COMPUTED: Computed | None = None  # the current Computed that is evaluating
PROCESSING_SIGNALS: set[tuple[str,]] = set()


class BaseObservable(ABC):
    """Abstract base class for all Observables."""

    @abstractmethod
    def __init__(self, fallback_value=None):
        """Initialize a BaseObservable."""
        super().__init__()
        self.public_name: str
        self.private_name: str

        # fixme can we make this an inner class enum?
        #  or some SignalTypes helper class?
        #  its even more complicated. Ideally you can define
        #  signal_types throughout the class hierarchy and they are just
        #  combined together.
        #  while we also want to  make sure that any signal being emitted is valid for that class
        self.signal_types: set = set()
        self.fallback_value = fallback_value

    def __get__(self, instance: HasObservables, owner):
        value = getattr(instance, self.private_name)

        if CURRENT_COMPUTED is not None:
            # there is a computed dependent on this Observable, so let's add
            # this Observable as a parent
            CURRENT_COMPUTED._add_parent(instance, self.public_name, value)

            # fixme, this can be done more cleanly
            #  problem here is that we cannot use self (i.e., the observable), we need to add the instance as well
            PROCESSING_SIGNALS.add(_hashable_signal(instance, self.public_name))

        return value

    def __set_name__(self, owner: HasObservables, name: str):
        self.public_name = name
        self.private_name = f"_{name}"
        # owner.register_observable(self)

    @abstractmethod
    def __set__(self, instance: HasObservables, value):
        # this only emits an on change signal, subclasses need to specify
        # this in more detail
        instance.notify(
            self.public_name,
            getattr(instance, self.private_name, self.fallback_value),
            value,
            "change",
        )

    def __str__(self):
        return f"{self.__class__.__name__}: {self.public_name}"


class Observable(BaseObservable):
    """Observable class."""

    def __init__(self, fallback_value=None):
        """Initialize an Observable."""
        super().__init__(fallback_value=fallback_value)

        self.signal_types: set = {
            "change",
        }

    def __set__(self, instance: HasObservables, value):  # noqa D103
        if (
            CURRENT_COMPUTED is not None
            and _hashable_signal(instance, self.public_name) in PROCESSING_SIGNALS
        ):
            raise ValueError(
                f"cyclical dependency detected: Computed({CURRENT_COMPUTED.name}) tries to change "
                f"{instance.__class__.__name__}.{self.public_name} while also being dependent it"
            )

        super().__set__(instance, value)  # send the notify
        setattr(instance, self.private_name, value)

        PROCESSING_SIGNALS.clear()  # we have notified our children, so we can clear this out


class Computable(BaseObservable):
    """A Computable that is depended on one or more Observables.

    .. code-block:: python

       class MyAgent(Agent):
           wealth = Computable()

           def __init__(self, model):
                super().__init__(model)
                wealth = Computed(func, args, kwargs)

    """

    # fixme, with new _register_observable thing
    #  we can do computed without a descriptor, but then you
    #  don't have attribute like access, you would need to do a call operation to get the value

    def __init__(self):
        """Initialize a Computable."""
        super().__init__()

        # fixme have 2 signal: change and is_dirty?
        self.signal_types: set = {
            "change",
        }

    def __get__(self, instance, owner):  # noqa: D105
        computed = getattr(instance, self.private_name)
        old_value = computed._value

        if CURRENT_COMPUTED is not None:
            CURRENT_COMPUTED._add_parent(instance, self.public_name, old_value)

        new_value = computed()

        if new_value != old_value:
            instance.notify(
                self.public_name,
                old_value,
                new_value,
                "change",
            )
            return new_value
        else:
            return old_value

    def __set__(self, instance: HasObservables, value: Computed):  # noqa D103
        if not isinstance(value, Computed):
            raise ValueError("value has to be a Computable instance")

        setattr(instance, self.private_name, value)
        value.name = self.public_name
        value.owner = instance
        getattr(
            instance, self.public_name
        )  # force evaluation of the computed to build the dependency graph


class Computed:
    def __init__(self, func: Callable, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self._is_dirty = True
        self._first = True
        self._value = None
        self.name: str = ""  # set by Computable
        self.owner: HasObservables  # set by Computable

        self.parents: weakref.WeakKeyDictionary[HasObservables, dict[str, Any]] = (
            weakref.WeakKeyDictionary()
        )

    def __str__(self):
        return f"COMPUTED: {self.name}"

    def _set_dirty(self, signal):
        if not self._is_dirty:
            self._is_dirty = True
            self.owner.notify(self.name, self._value, None, "change")

    def _add_parent(
        self, parent: HasObservables, name: str, current_value: Any
    ) -> None:
        """Add a parent Observable.

        Args:
            parent: the HasObservable instance to which the Observable belongs
            name: the public name of the Observable
            current_value: the current value of the Observable

        """
        parent.observe(name, All(), self._set_dirty)

        try:
            self.parents[parent][name] = current_value
        except KeyError:
            self.parents[parent] = {name: current_value}

    def _remove_parents(self):
        """Remove all parent Observables."""
        # we can unsubscribe from everything on each parent
        for parent in self.parents:
            parent.unobserve(All(), All(), self._set_dirty)

    def __call__(self):
        global CURRENT_COMPUTED  # noqa: PLW0603

        if self._is_dirty:
            changed = False

            if self._first:
                # fixme might be a cleaner solution for this
                #  basically, we have no parents.
                changed = True
                self._first = False

            # we might be dirty but values might have changed
            # back and forth in our parents so let's check to make sure we
            # really need to recalculate
            if not changed:
                for parent in self.parents.keyrefs():
                    # does parent still exist?
                    if parent := parent():
                        # if yes, compare old and new values for all
                        # tracked observables on this parent
                        for name, old_value in self.parents[parent].items():
                            new_value = getattr(parent, name)
                            if new_value != old_value:
                                changed = True
                                break  # we need to recalculate
                        else:
                            # trick for breaking cleanly out of nested for loops
                            # see https://stackoverflow.com/questions/653509/breaking-out-of-nested-loops
                            continue
                        break
                    else:
                        # one of our parents no longer exists
                        changed = True
                        break

            if changed:
                # the dependencies of the computable function might have changed
                # so, we rebuilt
                self._remove_parents()

                old = CURRENT_COMPUTED
                CURRENT_COMPUTED = self

                try:
                    self._value = self.func(*self.args, **self.kwargs)
                except Exception as e:
                    raise e
                finally:
                    CURRENT_COMPUTED = old

            self._is_dirty = False

        return self._value


class All:
    """Helper constant to subscribe to all Observables."""

    def __init__(self):  # noqa: D107
        self.name = "all"

    def __copy__(self):  # noqa: D105
        return self

    def __deepcopy__(self, memo):  # noqa: D105
        return self


class HasObservables:
    """HasObservables class."""

    # we can't use a weakset here because it does not handle bound methods correctly
    # also, a list is faster for our use case
    subscribers: dict[str, dict[str, list]]
    observables: dict[str, set[str]]

    def __init__(self, *args, **kwargs) -> None:
        """Initialize a HasObservables."""
        super().__init__(*args, **kwargs)
        self.subscribers = defaultdict(functools.partial(defaultdict, list))
        self.observables = dict(descriptor_generator(self))

    def _register_signal_emitter(self, name: str, signal_types: set[str]):
        """Helper function to register an Observable.

        This method can be used to register custom signals that are emitted by
        the class for a given attribute, but which cannot be covered by the Observable descriptor

        Args:
            name: the name of the signal emitter
            signal_types: the set of signals that might be emitted

        """
        self.observables[name] = signal_types

    def observe(
        self,
        name: str | All,
        signal_type: str | All,
        handler: Callable,
    ):
        """Subscribe to the Observable <name> for signal_type.

        Args:
            name: name of the Observable to subscribe to
            signal_type: the type of signal on the Observable to subscribe to
            handler: the handler to call

        Raises:
            ValueError: if the Observable <name> is not registered or if the Observable
            does not emit the given signal_type

        """
        # fixme should name/signal_type also take a list of str?
        if not isinstance(name, All):
            if name not in self.observables:
                raise ValueError(
                    f"you are trying to subscribe to {name}, but this Observable is not known"
                )
            else:
                names = [
                    name,
                ]
        else:
            names = self.observables.keys()

        for name in names:
            if not isinstance(signal_type, All):
                if signal_type not in self.observables[name]:
                    raise ValueError(
                        f"you are trying to subscribe to a signal of {signal_type} "
                        f"on Observable {name}, which does not emit this signal_type"
                    )
                else:
                    signal_types = [
                        signal_type,
                    ]
            else:
                signal_types = self.observables[name]

            ref = create_weakref(handler)
            for signal_type in signal_types:
                self.subscribers[name][signal_type].append(ref)

    def unobserve(self, name: str | All, signal_type: str | All, handler: Callable):
        """Unsubscribe to the Observable <name> for signal_type.

        Args:
            name: name of the Observable to unsubscribe from
            signal_type: the type of signal on the Observable to unsubscribe to
            handler: the handler that is unsubscribing

        """
        names = (
            [
                name,
            ]
            if not isinstance(name, All)
            else self.observables.keys()
        )

        for name in names:
            # we need to do this here because signal types might
            # differ for name so for each name we need to check
            if isinstance(signal_type, All):
                signal_types = self.observables[name]
            else:
                signal_types = [
                    signal_type,
                ]
            for signal_type in signal_types:
                with contextlib.suppress(KeyError):
                    remaining = []
                    for ref in self.subscribers[name][signal_type]:
                        if subscriber := ref():  # noqa: SIM102
                            if subscriber != handler:
                                remaining.append(ref)
                    self.subscribers[name][signal_type] = remaining

    def clear_all_subscriptions(self, name: str | All):
        """Clears all subscriptions for the observable <name>.

        if name is All, all subscriptions are removed

        Args:
            name: name of the Observable to unsubscribe for all signal types

        """
        if not isinstance(name, All):
            with contextlib.suppress(KeyError):
                del self.subscribers[name]
                # ignore when unsubscribing to Observables that have no subscription
        else:
            self.subscribers = defaultdict(functools.partial(defaultdict, list))

    def notify(
        self,
        observable: str,
        old_value: Any,
        new_value: Any,
        signal_type: str,
        **kwargs,
    ):
        """Emit a signal.

        Args:
            observable: the public name of the observable emitting the signal
            old_value: the old value of the observable
            new_value: the new value of the observable
            signal_type: the type of signal to emit
            kwargs: additional keyword arguments to include in the signal

        """
        signal = AttributeDict(
            name=observable,
            old=old_value,
            new=new_value,
            owner=self,
            type=signal_type,
            **kwargs,
        )

        self._mesa_notify(signal)

    def _mesa_notify(self, signal: AttributeDict):
        """Send out the signal.

        Args:
        signal: the signal

        Notes:
        signal must contain name and type attributes because this is how observers are stored.

        """
        # we put this into a helper method, so we can emit signals with other fields
        # then the default ones in notify.
        observable = signal.name
        signal_type = signal.type

        # because we are using a list of subscribers
        # we should update this list to subscribers that are still alive
        observers = self.subscribers[observable][signal_type]
        active_observers = []
        for observer in observers:
            if active_observer := observer():
                active_observer(signal)
                active_observers.append(observer)
        # use iteration to also remove inactive observers
        self.subscribers[observable][signal_type] = active_observers

    def add_threshold(self, observable_name: str, threshold: float, callback: Callable):
        """Convenience method for adding thresholds."""
        obs = getattr(type(self), observable_name)
        if isinstance(obs, ContinuousObservable):
            obs._thresholds.add(threshold)

            # Check if callback is already subscribed
            existing_subscribers = self.subscribers.get(observable_name, {}).get(
                "threshold_crossed", []
            )
            already_subscribed = any(
                ref() == callback for ref in existing_subscribers if ref() is not None
            )

            # Only subscribe if not already subscribed
            if not already_subscribed:
                self.observe(observable_name, "threshold_crossed", callback)
        else:
            raise ValueError(f"{observable_name} is not a ContinuousObservable")


class ContinuousObservable(Observable):
    """An Observable that changes continuously over time."""

    def __init__(self, initial_value: float, rate_func: Callable):
        """Initialize a ContinuousObservable."""
        super().__init__(fallback_value=initial_value)
        self.signal_types.add("threshold_crossed")
        self._rate_func = rate_func
        self._thresholds = set()

    def __set__(self, instance: HasObservables, value):
        """Set the value, ensuring we store a ContinuousState."""
        # Get or create state
        state = getattr(instance, self.private_name, None)

        if state is None:
            # First time - create ContinuousState
            state = ContinuousState(
                value=float(value),
                last_update=self._get_time(instance),
                rate_func=self._rate_func,
                thresholds=self._thresholds,
            )
            setattr(instance, self.private_name, state)
        else:
            # Update existing - just change the value and reset timestamp
            old_value = state.value
            state.value = float(value)
            state.last_update = self._get_time(instance)

            # Notify changes
            instance.notify(self.public_name, old_value, state.value, "change")

            # Check thresholds
            for threshold, direction in state.check_thresholds(old_value, state.value):
                instance.notify(
                    self.public_name,
                    old_value,
                    state.value,
                    "threshold_crossed",
                    threshold=threshold,
                    direction=direction,
                )

    def __get__(self, instance: HasObservables, owner):
        """Lazy evaluation - compute current value based on elapsed time."""
        if instance is None:
            return self

        # Get stored state
        state = getattr(instance, self.private_name, None)
        if state is None:
            # First access - initialize
            # Use simulator time if available, otherwise fall back to steps
            current_time = self._get_time(instance)
            state = ContinuousState(
                value=self.fallback_value,
                last_update=current_time,
                rate_func=self._rate_func,
                thresholds=self._thresholds,
            )
            setattr(instance, self.private_name, state)

        # Calculate new value based on time
        current_time = self._get_time(instance)
        elapsed = current_time - state.last_update

        if elapsed > 0:
            old_value = state.value
            new_value = state.calculate(elapsed, instance)

            # Check thresholds
            crossed = state.check_thresholds(old_value, new_value)

            # Update stored state
            state.value = new_value
            state.last_update = current_time

            # Emit signals
            if new_value != old_value:
                instance.notify(self.public_name, old_value, new_value, "change")

            for threshold, direction in crossed:
                instance.notify(
                    self.public_name,
                    old_value,
                    new_value,
                    "threshold_crossed",
                    threshold=threshold,
                    direction=direction,
                )

        # Register dependency if inside a Computed
        if CURRENT_COMPUTED is not None:
            CURRENT_COMPUTED._add_parent(instance, self.public_name, state.value)
            PROCESSING_SIGNALS.add(_hashable_signal(instance, self.public_name))

        return state.value

    # TODO: A universal truth for time should be implemented structurally in Mesa. See https://github.com/projectmesa/mesa/discussions/2228
    def _get_time(self, instance):
        """Get current time from model, trying multiple sources."""
        model = instance.model

        # Try simulator time first (for DEVS models)
        if hasattr(model, "simulator") and hasattr(model.simulator, "time"):
            return model.simulator.time

        # Fall back to model.time if it exists
        if hasattr(model, "time"):
            return model.time

        # Last resort: use steps as a proxy for time
        return float(model.steps)


class ContinuousState:
    """Internal state tracker for continuous observables."""

    __slots__ = ["last_update", "rate_func", "thresholds", "value"]

    def __init__(
        self, value: float, last_update: float, rate_func: Callable, thresholds: dict
    ):
        self.value = value
        self.last_update = last_update
        self.rate_func = rate_func
        self.thresholds = thresholds

    def calculate(self, elapsed: float, instance: Any) -> float:
        """Calculate new value based on elapsed time.

        Uses simple linear integration for now. Could be extended
        to support more complex integration methods.
        """
        rate = self.rate_func(self.value, elapsed, instance)
        return self.value + (rate * elapsed)

    def check_thresholds(self, old_value: float, new_value: float) -> list:
        """Check if any thresholds were crossed.

        Returns:
            List of (threshold_value, direction) tuples for crossed thresholds
        """
        crossed = []
        for threshold in self.thresholds:
            # Crossed upward
            if old_value < threshold <= new_value:
                crossed.append((threshold, "up"))
            # Crossed downward
            elif new_value <= threshold < old_value:
                crossed.append((threshold, "down"))
        return crossed


def descriptor_generator(obj) -> [str, BaseObservable]:
    """Yield the name and signal_types for each Observable defined on obj."""
    # we need to traverse the entire class hierarchy to properly get
    # also observables defined in super classes
    for base in type(obj).__mro__:
        base_dict = vars(base)

        for entry in base_dict.values():
            if isinstance(entry, BaseObservable):
                yield entry.public_name, entry.signal_types
