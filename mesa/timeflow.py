"""Unified time advancement and event scheduling for Mesa.

This module provides a clean, integrated API for both traditional time-step
advancement and discrete event scheduling within Mesa models.

Core classes:
- Scheduler: Manages event scheduling (absolute and relative times)
- RunControl: Controls time advancement (run_until, run_for, etc.)

Decorators:
- scheduled: Mark methods for automatic recurring execution
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from mesa.experimental.devs.eventlist import EventList, Priority, SimulationEvent

if TYPE_CHECKING:
    from mesa.model import Model


def scheduled(interval: int | float = 1.0, priority: Priority = Priority.DEFAULT):
    """Decorator to mark a method for automatic recurring scheduling.

    Args:
        interval: Time between executions (default: 1.0)
        priority: Priority level for the scheduled events

    Examples:
        @scheduled
        def step(self):
            self.agents.shuffle_do("step")

        @scheduled(interval=7)
        def weekly_update(self):
            self.collect_stats()
    """

    def decorator(func):
        func._scheduled = True
        func._scheduled_interval = interval
        func._scheduled_priority = priority
        return func

    return decorator


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

        # Auto-schedule methods marked with @scheduled decorator
        self._setup_scheduled_methods()

    def schedule_at(
        self,
        callback: Callable,
        time: int | float,
        priority: Priority = Priority.DEFAULT,
        args: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> SimulationEvent:
        """Schedule an event at an absolute time.

        Args:
            callback: The function to call when the event fires
            time: The absolute simulation time to fire the event
            priority: Priority level for simultaneous events
            args: Positional arguments to pass to the callback
            kwargs: Keyword arguments to pass to the callback

        Returns:
            SimulationEvent: The scheduled event (can be canceled later)

        Raises:
            ValueError: If time is in the past

        Examples:
            # Schedule a drought at time 50
            model.schedule_at(self.drought, time=50)

            # Schedule with priority
            model.schedule_at(self.critical_update, time=100, priority=Priority.HIGH)
        """
        if time < self.model.time:
            raise ValueError(
                f"Cannot schedule event in the past (current time: {self.model.time}, "
                f"requested time: {time})"
            )

        event = SimulationEvent(
            time=time,
            function=callback,
            priority=priority,
            function_args=args,
            function_kwargs=kwargs,
        )
        self.event_list.add_event(event)
        return event

    def schedule_after(
        self,
        callback: Callable,
        delay: int | float,
        priority: Priority = Priority.DEFAULT,
        args: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> SimulationEvent:
        """Schedule an event after a delay from the current time.

        Args:
            callback: The function to call when the event fires
            delay: Time units to wait before firing the event
            priority: Priority level for simultaneous events
            args: Positional arguments to pass to the callback
            kwargs: Keyword arguments to pass to the callback

        Returns:
            SimulationEvent: The scheduled event (can be canceled later)

        Examples:
            # Release prisoner after 5 time units
            model.schedule_after(self.release_prisoner, delay=5)

            # Agent schedules its own future action
            self.model.schedule_after(self.wake_up, delay=10)
        """
        return self.schedule_at(
            callback=callback,
            time=self.model.time + delay,
            priority=priority,
            args=args,
            kwargs=kwargs,
        )

    def cancel(self, event: SimulationEvent) -> None:
        """Cancel a scheduled event.

        Args:
            event: The event to cancel (returned from schedule_at or schedule_after)

        Examples:
            event = model.schedule_at(callback, time=50)
            # Changed our mind
            model.cancel(event)
        """
        self.event_list.remove(event)

    def clear(self) -> None:
        """Clear all scheduled events."""
        self.event_list.clear()

    def _setup_scheduled_methods(self) -> None:
        """Find and schedule all methods decorated with @scheduled."""
        for name in dir(self.model):
            # Skip private methods and properties
            if name.startswith("_"):
                continue

            attr = getattr(self.model, name)

            # Check if it's a method with _scheduled attribute
            if callable(attr) and hasattr(attr, "_scheduled"):
                interval = attr._scheduled_interval
                priority = attr._scheduled_priority

                # Schedule the first execution
                self._schedule_recurring(attr, interval, priority)

    def _schedule_recurring(
        self, method: Callable, interval: int | float, priority: Priority
    ) -> None:
        """Schedule a method to recur at fixed intervals.

        Args:
            method: The method to schedule
            interval: Time between executions
            priority: Priority level for the events
        """

        def recurring_wrapper():
            # Execute the method
            method()
            # Reschedule for next occurrence
            self._schedule_recurring(method, interval, priority)

        next_time = self.model.time + interval
        self.schedule_at(recurring_wrapper, time=next_time, priority=priority)


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

        Examples:
            # Run until time reaches 100
            model.run_until(100)
        """
        while not self.scheduler.event_list.is_empty():
            try:
                event = self.scheduler.event_list.pop_event()
            except IndexError:
                # No more events
                self.model.time = end_time
                break

            if event.time <= end_time:
                self.model.time = event.time
                event.execute()
            else:
                # Event is beyond end_time, put it back
                self.scheduler.event_list.add_event(event)
                self.model.time = end_time
                break

        # Ensure we advance to end_time even if no events
        if self.model.time < end_time:
            self.model.time = end_time

    def run_for(self, duration: int | float) -> None:
        """Run the model for a specific duration from current time.

        Args:
            duration: The amount of time to advance

        Examples:
            # Run for 50 time units
            model.run_for(50)
        """
        end_time = self.model.time + duration
        self.run_until(end_time)

    def run_while(self, condition: Callable[[Model], bool]) -> None:
        """Run the model while a condition remains true.

        Args:
            condition: A function that takes the model and returns bool

        Examples:
            # Run while the model says it should keep running
            model.run_while(lambda m: m.running)

            # Run until population drops below threshold
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
            # Step through events manually
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
