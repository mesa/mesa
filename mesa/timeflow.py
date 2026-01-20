"""Unified time advancement and event scheduling for Mesa.

This module provides a clean, integrated API for both traditional time-step
advancement and discrete event scheduling within Mesa models.

Core classes:
- Scheduler: Manages event scheduling (absolute and relative times)
- RunControl: Controls time advancement (run_until, run_for, etc.)

The @scheduled decorator is planned for future implementation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from mesa.experimental.devs.eventlist import (
    EventList,
    Priority,
    RecurringEvent,
    SimulationEvent,
)

if TYPE_CHECKING:
    from mesa.model import Model

# Re-export for convenience
__all__ = [
    "EventList",
    "Priority",
    "RecurringEvent",
    "RunControl",
    "Scheduler",
    "SimulationEvent",
]


class Scheduler:
    """Handles event scheduling for a model.

    Provides methods to schedule callbacks at absolute or relative times,
    with optional priority for simultaneous events.

    Attributes:
        model: The Mesa model instance
        event_list: Internal priority queue of scheduled events
    """

    def __init__(self, model: Model) -> None:
        """Initialize the scheduler.

        Args:
            model: The Mesa model instance to schedule events for
        """
        self.model = model
        self.event_list = EventList()

    def schedule(
        self,
        callback: Callable,
        start_at: int | float | None = None,
        start_after: int | float | None = None,
        interval: int | float | Callable | None = None,
        count: int | None = None,
        end_at: int | float | None = None,
        end_after: int | float | None = None,
        priority: Priority = Priority.DEFAULT,
        args: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> SimulationEvent | RecurringEvent:
        """Schedule an event with flexible timing options.

        Args:
            callback: The function to call when the event fires
            start_at: Absolute simulation time to start (mutually exclusive with start_after)
            start_after: Delay from current time to start (mutually exclusive with start_at)
            interval: Time between recurrences. Can be a number or callable(model) -> number.
                      If None, event is one-off. If provided, event recurs.
            count: Maximum number of executions. For one-off events, defaults to 1.
                   For recurring events, None means infinite. Mutually exclusive with end_at/end_after.
            end_at: Absolute time to stop recurring. Mutually exclusive with count/end_after.
            end_after: Duration from first execution to stop. Mutually exclusive with count/end_at.
            priority: Priority level for simultaneous events
            args: Positional arguments to pass to the callback
            kwargs: Keyword arguments to pass to the callback

        Returns:
            SimulationEvent for one-off events, RecurringEvent for recurring events.
            The returned object can be used to cancel/pause/resume the event.

        Raises:
            ValueError: If conflicting parameters are provided

        Examples:
            # One-off event at absolute time
            model.schedule(self.disaster, start_at=50)

            # One-off event after delay
            model.schedule(self.release, start_after=10)

            # Recurring event every 7 time units
            model.schedule(self.weekly_report, interval=7)

            # Recurring with stochastic interval
            model.schedule(self.arrival, interval=lambda m: m.random.expovariate(2.0))

            # Recurring with count limit
            model.schedule(self.payment, interval=30, count=3)

            # Recurring until absolute time
            model.schedule(self.monitor, interval=1, end_at=100)
        """
        # Validate start parameters
        if start_at is not None and start_after is not None:
            raise ValueError("Cannot specify both start_at and start_after")

        # Validate end parameters for recurring events
        end_params = [p for p in [count, end_at, end_after] if p is not None]
        if len(end_params) > 1:
            raise ValueError("Can only specify one of: count, end_at, end_after")

        # Determine start time
        if start_at is not None:
            start_time = start_at
        elif start_after is not None:
            start_time = self.model.time + start_after
        else:
            # Default: start at next time unit (time + interval for recurring, immediate for one-off)
            if interval is not None:
                # For recurring, first execution at time + interval
                if callable(interval):
                    start_time = self.model.time + interval(self.model)
                else:
                    start_time = self.model.time + interval
            else:
                # For one-off without start time, execute immediately (current time)
                start_time = self.model.time

        # Validate start time
        if start_time < self.model.time:
            raise ValueError(
                f"Cannot schedule event in the past (current time: {self.model.time}, "
                f"requested time: {start_time})"
            )

        # Create appropriate event type
        if interval is not None:
            # Recurring event
            event = RecurringEvent(
                time=start_time,
                function=callback,
                scheduler=self,
                interval=interval,
                priority=priority,
                function_args=args,
                function_kwargs=kwargs,
                count=count,
                end_at=end_at,
                end_after=end_after,
            )
        else:
            # One-off event
            event = SimulationEvent(
                time=start_time,
                function=callback,
                priority=priority,
                function_args=args,
                function_kwargs=kwargs,
            )

        self.event_list.add_event(event)
        return event

    def schedule_at(
        self,
        callback: Callable,
        time: int | float,
        priority: Priority = Priority.DEFAULT,
        args: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> SimulationEvent:
        """Schedule a one-off event at an absolute time.

        Convenience method equivalent to schedule(callback, start_at=time).

        Args:
            callback: The function to call when the event fires
            time: The absolute simulation time to fire the event
            priority: Priority level for simultaneous events
            args: Positional arguments to pass to the callback
            kwargs: Keyword arguments to pass to the callback

        Returns:
            SimulationEvent: The scheduled event (can be canceled)

        Examples:
            model.schedule_at(self.drought, time=50)
        """
        return self.schedule(
            callback, start_at=time, priority=priority, args=args, kwargs=kwargs
        )

    def schedule_after(
        self,
        callback: Callable,
        delay: int | float,
        priority: Priority = Priority.DEFAULT,
        args: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> SimulationEvent:
        """Schedule a one-off event after a delay from current time.

        Convenience method equivalent to schedule(callback, start_after=delay).

        Args:
            callback: The function to call when the event fires
            delay: Time units to wait before firing the event
            priority: Priority level for simultaneous events
            args: Positional arguments to pass to the callback
            kwargs: Keyword arguments to pass to the callback

        Returns:
            SimulationEvent: The scheduled event (can be canceled)

        Examples:
            model.schedule_after(self.release, delay=5)
        """
        return self.schedule(
            callback, start_after=delay, priority=priority, args=args, kwargs=kwargs
        )

    def cancel(self, event: SimulationEvent) -> None:
        """Cancel a scheduled event.

        Args:
            event: The event to cancel

        Examples:
            event = model.schedule_at(callback, time=50)
            model.cancel(event)
        """
        self.event_list.remove(event)

    def clear(self) -> None:
        """Clear all scheduled events."""
        self.event_list.clear()


class RunControl:
    """Controls time advancement for a model.

    Provides methods to run the model for specific durations, until specific
    times, or while conditions are met.

    Attributes:
        model: The Mesa model instance
        scheduler: The scheduler managing events
    """

    def __init__(self, model: Model, scheduler: Scheduler) -> None:
        """Initialize run control.

        Args:
            model: The Mesa model instance
            scheduler: The scheduler managing events for this model
        """
        self.model = model
        self.scheduler = scheduler

    def run_until(self, end_time: int | float) -> None:
        """Run the model until the specified time.

        Executes all events scheduled up to and including end_time.

        Args:
            end_time: The simulation time to run until
        """
        while not self.scheduler.event_list.is_empty():
            try:
                event = self.scheduler.event_list.pop_event()
            except IndexError:
                break

            if event.time <= end_time:
                self.model.time = event.time
                event.execute()
            else:
                # Event is beyond end_time, put it back
                self.scheduler.event_list.add_event(event)
                break

        # Ensure we advance to end_time even if no events
        if self.model.time < end_time:
            self.model.time = end_time

    def run_for(self, duration: int | float) -> None:
        """Run the model for a specific duration from current time.

        Args:
            duration: The amount of time to advance

        Examples:
            model.run_for(50)
        """
        end_time = self.model.time + duration
        self.run_until(end_time)

    def run_while(self, condition: Callable[[Model], bool]) -> None:
        """Run the model while a condition remains true.

        Args:
            condition: A function that takes the model and returns bool

        Examples:
            model.run_while(lambda m: m.running)
            model.run_while(lambda m: len(m.agents) > 10)
        """
        while condition(self.model):
            if self.scheduler.event_list.is_empty():
                break

            try:
                event = self.scheduler.event_list.pop_event()
                self.model.time = event.time
                event.execute()
            except IndexError:
                break

    def run_next_event(self) -> bool:
        """Execute the next scheduled event.

        Useful for debugging or stepping through events one at a time.

        Returns:
            bool: True if an event was executed, False if event list is empty

        Examples:
            while model.run_next_event():
                print(f"Time: {model.time}")
        """
        if self.scheduler.event_list.is_empty():
            return False

        try:
            event = self.scheduler.event_list.pop_event()
            self.model.time = event.time
            event.execute()
            return True
        except IndexError:
            return False
