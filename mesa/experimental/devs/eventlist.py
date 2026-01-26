"""Core event management functionality for Mesa's discrete event simulation system.

This module provides the foundational data structures and classes needed for event-based
simulation in Mesa. The EventList class is a priority queue implementation that maintains
simulation events in chronological order while respecting event priorities. Key features:

- Priority-based event ordering
- Weak references to prevent memory leaks from canceled events
- Efficient event insertion and removal using a heap queue
- Support for event cancellation without breaking the heap structure

The module contains three main components:
- Priority: An enumeration defining event priority levels (HIGH, DEFAULT, LOW)
- SimulationEvent: A class representing individual events with timing and execution details
- EventList: A heap-based priority queue managing the chronological ordering of events

The implementation supports both pure discrete event simulation and hybrid approaches
combining agent-based modeling with event scheduling.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from enum import IntEnum
from heapq import heapify, heappop, heappush, nsmallest
from types import MethodType
from typing import TYPE_CHECKING, Any
from weakref import WeakMethod, ref

if TYPE_CHECKING:
    from mesa import Model


class Priority(IntEnum):
    """Enumeration of priority levels."""

    LOW = 10
    DEFAULT = 5
    HIGH = 1


class SimulationEvent:
    """A simulation event.

    The callable is wrapped using weakref, so there is no need to explicitly cancel event if e.g., an agent
    is removed from the simulation.

    Attributes:
        time (float): The simulation time of the event
        fn (Callable): The function to execute for this event
        priority (Priority): The priority of the event
        unique_id (int) the unique identifier of the event
        function_args (list[Any]): Argument for the function
        function_kwargs (Dict[str, Any]): Keyword arguments for the function


    Notes:
        simulation events use a weak reference to the callable. Therefore, you cannot pass a lambda function in fn.
        A simulation event where the callable no longer exists (e.g., because the agent has been removed from the model)
        will fail silently.

    """

    _ids = itertools.count()

    @property
    def CANCELED(self) -> bool:  # noqa: D102
        return self._canceled

    def __init__(
        self,
        time: int | float,
        function: Callable,
        priority: Priority = Priority.DEFAULT,
        function_args: list[Any] | None = None,
        function_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a simulation event.

        Args:
            time: the instant of time of the simulation event
            function: the callable to invoke
            priority: the priority of the event
            function_args: arguments for callable
            function_kwargs: keyword arguments for the callable
        """
        super().__init__()
        if not callable(function):
            raise Exception()

        self.time = time
        self.priority = priority.value
        self._canceled = False

        if isinstance(function, MethodType):
            function = WeakMethod(function)
        else:
            function = ref(function)

        self.fn = function
        self.unique_id = next(self._ids)
        self.function_args = function_args if function_args else []
        self.function_kwargs = function_kwargs if function_kwargs else {}

    def execute(self):
        """Execute this event."""
        if not self._canceled:
            fn = self.fn()
            if fn is not None:
                fn(*self.function_args, **self.function_kwargs)

    def cancel(self) -> None:
        """Cancel this event."""
        self._canceled = True
        self.fn = None
        self.function_args = []
        self.function_kwargs = {}

    def __lt__(self, other):  # noqa
        # Define a total ordering for events to be used by the heapq
        return (self.time, self.priority, self.unique_id) < (
            other.time,
            other.priority,
            other.unique_id,
        )

    def __getstate__(self):
        """Prepare state for pickling."""
        state = self.__dict__.copy()
        # Convert weak reference back to strong reference for pickling
        fn = self.fn() if self.fn is not None else None
        state["_fn_strong"] = fn
        state["fn"] = None  # Don't pickle the weak reference
        return state

    def __setstate__(self, state):
        """Restore state after unpickling."""
        fn = state.pop("_fn_strong")
        self.__dict__.update(state)
        # Recreate weak reference
        if fn is not None:
            if isinstance(fn, MethodType):
                self.fn = WeakMethod(fn)
            else:
                self.fn = ref(fn)
        else:
            self.fn = None


class EventGenerator:
    """A generator that creates recurring events at specified intervals.

    EventGenerator represents a pattern for when things should happen repeatedly.
    Unlike a single SimulationEvent, an EventGenerator is persistent and can be
    stopped or configured with stop conditions.

    Attributes:
        model: The model this generator belongs to
        function: The callable to execute for each generated event
        interval: Time between events (fixed value or callable returning value)
        priority: Priority level for generated events

    """

    def __init__(
        self,
        model: Model,
        function: Callable,
        interval: float | int | Callable[[Model], float | int],
        priority: Priority = Priority.DEFAULT,
    ) -> None:
        """Initialize an EventGenerator.

        Args:
            model: The model this generator belongs to
            function: The callable to execute for each generated event.
                     Use functools.partial to bind arguments.
            interval: Time between events. Can be a fixed value or a callable
                     that takes the model and returns the interval.
            priority: Priority level for generated events

        """
        self.model = model
        self.function = function
        self.interval = interval
        self.priority = priority

        self._active: bool = False
        self._current_event: SimulationEvent | None = None
        self._execution_count: int = 0

        # Stop conditions (mutually exclusive)
        self._max_count: int | None = None
        self._end_time: float | None = None

    @property
    def is_active(self) -> bool:
        """Return whether the generator is currently active."""
        return self._active

    @property
    def execution_count(self) -> int:
        """Return the number of times this generator has executed."""
        return self._execution_count

    def _get_interval(self) -> float | int:
        """Get the next interval value."""
        if callable(self.interval):
            return self.interval(self.model)
        return self.interval

    def _should_stop(self, next_time: float) -> bool:
        """Check if the generator should stop before scheduling the next event."""
        return (
            self._max_count is not None and self._execution_count >= self._max_count
        ) or (self._end_time is not None and next_time > self._end_time)

    def _execute_and_reschedule(self) -> None:
        """Execute the function and schedule the next event."""
        if not self._active:
            return

        self.function()
        self._execution_count += 1

        # Schedule next event if we shouldn't stop
        next_time = self.model.time + self._get_interval()
        if not self._should_stop(next_time):
            self._schedule_next(next_time)
        else:
            self._active = False
            self._current_event = None

    def _schedule_next(self, time: float) -> None:
        """Schedule the next event at the given time."""
        self._current_event = SimulationEvent(
            time,
            self._execute_and_reschedule,
            priority=self.priority,
        )
        self.model._simulator.event_list.add_event(self._current_event)

    def start(
        self,
        at: float | None = None,
        after: float | None = None,
    ) -> EventGenerator:
        """Start the event generator.

        Args:
            at: Absolute time to start generating events
            after: Relative time from now to start generating events

        Returns:
            Self for method chaining

        Raises:
            ValueError: If both `at` and `after` are specified

        """
        if self._active:
            return self

        if at is not None and after is not None:
            raise ValueError("Cannot specify both 'at' and 'after'")

        if at is None and after is None:
            # Default: start at next interval from now
            start_time = self.model.time + self._get_interval()
        elif at is not None:
            if at < self.model.time:
                raise ValueError(f"Cannot start in the past: {at} < {self.model.time}")
            start_time = at
        else:  # after is not None
            if after < 0:
                raise ValueError(f"Cannot start in the past: after={after}")
            start_time = self.model.time + after

        self._active = True
        self._schedule_next(start_time)
        return self

    def stop(
        self,
        at: float | None = None,
        after: float | None = None,
        count: int | None = None,
    ) -> EventGenerator:
        """Stop the event generator.

        Args:
            at: Absolute time to stop generating events
            after: Relative time from now to stop generating events
            count: Number of additional executions before stopping

        Returns:
            Self for method chaining

        Raises:
            ValueError: If more than one stop condition is specified

        """
        conditions = sum(x is not None for x in [at, after, count])
        if conditions > 1:
            raise ValueError("Can only specify one of 'at', 'after', or 'count'")

        if conditions == 0:
            # Immediate stop
            self._active = False
            if self._current_event is not None:
                self._current_event.cancel()
                self._current_event = None
            return self

        if at is not None:
            self._end_time = at
        elif after is not None:
            self._end_time = self.model.time + after
        elif count is not None:
            self._max_count = self._execution_count + count

        return self


class EventList:
    """An event list.

    This is a heap queue sorted list of events. Events are always removed from the left, so heapq is a performant and
    appropriate data structure. Events are sorted based on their time stamp, their priority, and their unique_id
    as a tie-breaker, guaranteeing a complete ordering.


    """

    def __init__(self):
        """Initialize an event list."""
        self._events: list[SimulationEvent] = []
        heapify(self._events)

    def add_event(self, event: SimulationEvent):
        """Add the event to the event list.

        Args:
            event (SimulationEvent): The event to be added

        """
        heappush(self._events, event)

    def peek_ahead(self, n: int = 1) -> list[SimulationEvent]:
        """Look at the first n non-canceled event in the event list.

        Args:
            n (int): The number of events to look ahead

        Returns:
            list[SimulationEvent]

        Raises:
            IndexError: If the eventlist is empty

        Notes:
            this method can return a list shorted then n if the number of non-canceled events on the event list
            is less than n.

        """
        # look n events ahead
        if self.is_empty():
            raise IndexError("event list is empty")

        # Filter out canceled events and get n smallest in correct chronological order
        valid_events = [e for e in self._events if not e.CANCELED]
        return nsmallest(n, valid_events)

    def pop_event(self) -> SimulationEvent:
        """Pop the first element from the event list."""
        while self._events:
            event = heappop(self._events)
            if not event.CANCELED:
                return event
        raise IndexError("Event list is empty")

    def is_empty(self) -> bool:
        """Return whether the event list is empty."""
        return len(self) == 0

    def __contains__(self, event: SimulationEvent) -> bool:  # noqa
        return event in self._events

    def __len__(self) -> int:  # noqa
        return len(self._events)

    def __repr__(self) -> str:
        """Return a string representation of the event list."""
        events_str = ", ".join(
            [
                f"Event(time={e.time}, priority={e.priority}, id={e.unique_id})"
                for e in self._events
                if not e.CANCELED
            ]
        )
        return f"EventList([{events_str}])"

    def remove(self, event: SimulationEvent) -> None:
        """Remove an event from the event list.

        Args:
            event (SimulationEvent): The event to be removed

        """
        # we cannot simply remove items from _eventlist because this breaks
        # heap structure invariant. So, we use a form of lazy deletion.
        # SimEvents have a CANCELED flag that we set to True, while popping and peek_ahead
        # silently ignore canceled events
        event.cancel()

    def clear(self):
        """Clear the event list."""
        self._events.clear()
