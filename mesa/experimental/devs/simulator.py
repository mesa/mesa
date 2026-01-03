"""Simulator implementations for different time advancement approaches in Mesa.

This module provides simulator classes that control how simulation time advances and how
events are executed. It supports both discrete-time and continuous-time simulations through
three main classes:

- Simulator: Base class defining the core simulation control interface
- ABMSimulator: A simulator for agent-based models that combines fixed time steps with
  event scheduling. Uses integer time units and automatically schedules model.step()
- DEVSimulator: A pure discrete event simulator using floating-point time units for
  continuous time simulation

Key features:
- Flexible time units (integer or float)
- Event scheduling using absolute or relative times
- Priority-based event execution
- Support for running simulations for specific durations or until specific end times

The simulators enable Mesa models to use traditional time-step based approaches, pure
event-driven approaches, or hybrid combinations of both.
"""

from __future__ import annotations

import numbers
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .eventlist import EventList, Priority, SimulationEvent

if TYPE_CHECKING:
    from mesa import Model


class Simulator:
    """The Simulator controls the time advancement of the model.

    The simulator uses next event time progression to advance the simulation time, and execute the next event

    Attributes:
        event_list (EventList): The list of events to execute
        time (float | int): The current simulation time
        time_unit (type) : The unit of the simulation time
        model (Model): The model to simulate


    """

    # TODO: add replication support
    # TODO: add experimentation support

    def __init__(self, time_unit: type, start_time: int | float):
        """Initialize a Simulator instance.

        Args:
            time_unit: type of the smulaiton time
            start_time: the starttime of the simulator
        """
        # should model run in a separate thread,
        # and we can then interact with start, stop, run_until, and step?
        self.event_list = EventList()
        self.start_time = start_time
        self.time_unit = time_unit
        self.model: Model | None = None

    @property
    def time(self) -> float:
        """Simulator time (deprecated)."""
        warnings.warn(
            "simulator.time is deprecated, use model.time instead",
            FutureWarning,
            stacklevel=2,
        )
        return self.model.time

    def check_time_unit(self, time: int | float) -> bool: ...  # noqa: D102

    def setup(self, model: Model) -> None:
        """Set up the simulator with the model to simulate.

        Args:
            model (Model): The model to simulate

        Raises:
            Exception if simulator.time is not equal to simulator.starttime
            Exception if event list is not empty

        """
        if model.time != self.start_time:
            raise ValueError(
                f"Model time ({model.time}) does not match simulator start_time ({self.start_time}). "
                "Has the model already been run?"
            )
        if not self.event_list.is_empty():
            raise ValueError("Events already scheduled. Call setup before scheduling.")

        self.model = model
        model._simulator = self  # Register simulator with model

    def reset(self):
        """Reset the simulator."""
        self.event_list.clear()
        if self.model is not None:
            self.model._simulator = None
            self.model.time = self.start_time
            self.model = None

    def run_until(self, end_time: int | float) -> None:
        """Run the simulator until the end time.

        Args:
            end_time (int | float): The end time for stopping the simulator

        Raises:
            Exception if simulator.setup() has not yet been called

        """
        if self.model is None:
            raise RuntimeError(
                "Simulator not set up. Call simulator.setup(model) first."
            )

        while True:
            try:
                event = self.event_list.pop_event()
            except IndexError:
                self.model.time = end_time
                break

            if event.time <= end_time:
                self.model.time = event.time
                event.execute()
            else:
                self.model.time = end_time
                self._schedule_event(event)
                break

    def run_next_event(self):
        """Execute the next event.

        Raises:
            Exception if simulator.setup() has not yet been called

        """
        if self.model is None:
            raise RuntimeError(
                "Simulator not set up. Call simulator.setup(model) first."
            )

        try:
            event = self.event_list.pop_event()
        except IndexError:
            return

        self.model.time = event.time
        event.execute()

    def run_for(self, time_delta: int | float):
        """Run the simulator for the specified time delta.

        Args:
            time_delta (float| int): The time delta. The simulator is run from the current time to the current time
                                     plus the time delta

        Raises:
            Exception if simulator.setup() has not yet been called

        """
        try:
            end_time = self.model.time + time_delta
        except AttributeError as e:
            raise RuntimeError(
                "Simulator not set up. Call simulator.setup(model) first."
            ) from e
        else:
            self.run_until(end_time)

    def schedule_event_now(
        self,
        function: Callable,
        priority: Priority = Priority.DEFAULT,
        function_args: list[Any] | None = None,
        function_kwargs: dict[str, Any] | None = None,
    ) -> SimulationEvent:
        """Schedule event for the current time instant.

        Args:
            function (Callable): The callable to execute for this event
            priority (Priority): the priority of the event, optional
            function_args (List[Any]): list of arguments for function
            function_kwargs (Dict[str, Any]):  dict of keyword arguments for function

        Returns:
            SimulationEvent: the simulation event that is scheduled

        """
        return self.schedule_event_relative(
            function,
            0.0,
            priority=priority,
            function_args=function_args,
            function_kwargs=function_kwargs,
        )

    def schedule_event_absolute(
        self,
        function: Callable,
        time: int | float,
        priority: Priority = Priority.DEFAULT,
        function_args: list[Any] | None = None,
        function_kwargs: dict[str, Any] | None = None,
    ) -> SimulationEvent:
        """Schedule event for the specified time instant.

        Args:
            function (Callable): The callable to execute for this event
            time (int | float): the time for which to schedule the event
            priority (Priority): the priority of the event, optional
            function_args (List[Any]): list of arguments for function
            function_kwargs (Dict[str, Any]):  dict of keyword arguments for function

        Returns:
            SimulationEvent: the simulation event that is scheduled

        """
        if self.model.time > time:
            raise ValueError("trying to schedule an event in the past")

        event = SimulationEvent(
            time,
            function,
            priority=priority,
            function_args=function_args,
            function_kwargs=function_kwargs,
        )
        self._schedule_event(event)
        return event

    def schedule_event_relative(
        self,
        function: Callable,
        time_delta: int | float,
        priority: Priority = Priority.DEFAULT,
        function_args: list[Any] | None = None,
        function_kwargs: dict[str, Any] | None = None,
    ) -> SimulationEvent:
        """Schedule event for the current time plus the time delta.

        Args:
            function (Callable): The callable to execute for this event
            time_delta (int | float): the time delta
            priority (Priority): the priority of the event, optional
            function_args (List[Any]): list of arguments for function
            function_kwargs (Dict[str, Any]):  dict of keyword arguments for function

        Returns:
            SimulationEvent: the simulation event that is scheduled

        """
        if time_delta < 0:
            raise ValueError(
                f"Cannot schedule event in the past: time_delta ({time_delta}) "
                f"would result in event time ({self.model.time + time_delta}) "
                f"before current time ({self.model.time})"
            )

        event = SimulationEvent(
            self.model.time + time_delta,
            function,
            priority=priority,
            function_args=function_args,
            function_kwargs=function_kwargs,
        )
        self._schedule_event(event)
        return event

    def cancel_event(self, event: SimulationEvent) -> None:
        """Remove the event from the event list.

        Args:
            event (SimulationEvent): The simulation event to remove

        """
        self.event_list.remove(event)

    def _schedule_event(self, event: SimulationEvent):
        if not self.check_time_unit(event.time):
            raise ValueError(
                f"time unit mismatch {event.time} is not of time unit {self.time_unit}"
            )

        # check timeunit of events
        self.event_list.add_event(event)


class ABMSimulator(Simulator):
    """This simulator uses incremental time progression, while allowing for additional event scheduling.

    The basic time unit of this simulator is an integer. It schedules `model.step` for each tick with the
    highest priority. This implies that by default, `model.step` is the first event executed at a specific tick.
    In addition, discrete event scheduling, using integer as the time unit is fully supported, paving the way
    for hybrid ABM-DEVS simulations.

    """

    def __init__(self):
        """Initialize a ABM simulator."""
        super().__init__(int, 0)

    def setup(self, model):
        """Set up the simulator with the model to simulate.

        Args:
            model (Model): The model to simulate

        """
        super().setup(model)
        self.schedule_event_next_tick(self.model.step, priority=Priority.HIGH)

    def check_time_unit(self, time) -> bool:
        """Check whether the time is of the correct unit.

        Args:
            time (int | float): the time

        Returns:
            bool: whether the time is of the correct unit

        """
        if isinstance(time, int):
            return True
        if isinstance(time, float):
            return time.is_integer()
        else:
            return False

    def schedule_event_next_tick(
        self,
        function: Callable,
        priority: Priority = Priority.DEFAULT,
        function_args: list[Any] | None = None,
        function_kwargs: dict[str, Any] | None = None,
    ) -> SimulationEvent:
        """Schedule a SimulationEvent for the next tick.

        Args:
            function (Callable): the callable to execute
            priority (Priority): the priority of the event
            function_args (List[Any]): List of arguments to pass to the callable
            function_kwargs (Dict[str, Any]): List of keyword arguments to pass to the callable

        """
        return self.schedule_event_relative(
            function,
            1,
            priority=priority,
            function_args=function_args,
            function_kwargs=function_kwargs,
        )

    def run_until(self, end_time: int) -> None:
        """Run the simulator up to and included the specified end time.

        Args:
            end_time (float| int): The end_time delta. The simulator is until the specified end time

        Raises:
            Exception if simulator.setup() has not yet been called

        """
        if self.model is None:
            raise RuntimeError(
                "Simulator not set up. Call simulator.setup(model) first."
            )

        while True:
            try:
                event = self.event_list.pop_event()
            except IndexError:
                self.model.time = float(end_time)
                break

            if event.time <= end_time:
                self.model.time = float(event.time)

                # Reschedule model.step for next tick if this is a step event
                if event.fn() == self.model.step:
                    self.schedule_event_next_tick(
                        self.model.step, priority=Priority.HIGH
                    )

                event.execute()
            else:
                self.model.time = float(end_time)
                self._schedule_event(event)
                break


class DEVSimulator(Simulator):
    """A simulator where the unit of time is a float.

    Can be used for full-blown discrete event simulating using event scheduling.

    """

    def __init__(self):
        """Initialize a DEVS simulator."""
        super().__init__(float, 0.0)

    def check_time_unit(self, time) -> bool:
        """Check whether the time is of the correct unit.

        Args:
            time (float): the time

        Returns:
        bool: whether the time is of the correct unit

        """
        return isinstance(time, numbers.Number)
