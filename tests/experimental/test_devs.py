"""Tests for experimental Simulator classes."""

from functools import partial
from unittest.mock import MagicMock, Mock

import pytest

from mesa import Model
from mesa.experimental.devs.simulator import ABMSimulator, DEVSimulator
from mesa.time import (
    Event,
    EventList,
    Priority,
)

# Ignore deprecation warnings for Simulator classes in this test file
pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


def test_devs_simulator():
    """Tests devs simulator."""
    simulator = DEVSimulator()

    # setup
    model = Model()
    simulator.setup(model)

    assert len(simulator.event_list) == 0
    assert simulator.model == model
    assert model.time == 0.0

    # schedule_event_now
    fn1 = MagicMock()
    event1 = simulator.schedule_event_now(fn1)
    assert event1 in simulator.event_list
    assert len(simulator.event_list) == 1

    # schedule_event_absolute
    fn2 = MagicMock()
    event2 = simulator.schedule_event_absolute(fn2, 1.0)
    assert event2 in simulator.event_list
    assert len(simulator.event_list) == 2

    # schedule_event_relative
    fn3 = MagicMock()
    event3 = simulator.schedule_event_relative(fn3, 0.5)
    assert event3 in simulator.event_list
    assert len(simulator.event_list) == 3

    # run_for
    simulator.run_for(0.8)
    fn1.assert_called_once()
    fn3.assert_called_once()
    assert model.time == 0.8

    simulator.run_for(0.2)
    fn2.assert_called_once()
    assert model.time == 1.0

    simulator.run_for(0.2)
    assert model.time == 1.2

    with pytest.raises(ValueError):
        simulator.schedule_event_absolute(fn2, 0.5)

    # schedule_event_relative with negative time_delta (causality violation)
    with pytest.raises(ValueError, match="Cannot schedule event in the past"):
        simulator.schedule_event_relative(fn2, -0.5)

    # step
    simulator = DEVSimulator()
    model = Model()
    simulator.setup(model)

    fn = MagicMock()
    simulator.schedule_event_absolute(fn, 1.0)
    simulator.run_next_event()
    fn.assert_called_once()
    assert model.time == 1.0
    simulator.run_next_event()
    assert model.time == 1.0

    simulator = DEVSimulator()
    with pytest.raises(Exception):
        simulator.run_next_event()

    # cancel_event
    simulator = DEVSimulator()
    model = Model()
    simulator.setup(model)
    fn = MagicMock()
    event = simulator.schedule_event_relative(fn, 0.5)
    simulator.cancel_event(event)
    assert event.CANCELED

    # simulator reset
    simulator.reset()
    assert len(simulator.event_list) == 0
    assert simulator.model is model

    # run_for without setup
    simulator = DEVSimulator()
    with pytest.raises(RuntimeError, match="Simulator not set up"):
        simulator.run_for(1.0)

    # run_until without setup
    simulator = DEVSimulator()
    with pytest.raises(Exception):
        simulator.run_until(10)

    # setup with time advanced
    simulator = DEVSimulator()
    model = Model()
    model.time = 1.0  # Advance time before setup
    with pytest.raises(ValueError):
        simulator.setup(model)

    # setup with event scheduled
    simulator = DEVSimulator()
    with pytest.raises(RuntimeError, match="Simulator not set up"):
        simulator.event_list.add_event(Event(1.0, Mock(), Priority.DEFAULT))


def test_abm_simulator():
    """Tests abm simulator."""
    simulator = ABMSimulator()

    # setup
    model = Model()
    simulator.setup(model)

    # schedule_event_next_tick
    fn = MagicMock()
    simulator.schedule_event_next_tick(fn)
    assert len(simulator.event_list) == 2

    simulator.run_for(3)
    assert model.time == 3.0

    # run_until without setup
    simulator = ABMSimulator()
    with pytest.raises(Exception):
        simulator.run_until(10)

    # run_for without setup
    simulator = ABMSimulator()
    with pytest.raises(RuntimeError, match="Simulator not set up"):
        simulator.run_for(3)


def test_simulator_time_deprecation():
    """Test that simulator.time emits future warning."""
    simulator = DEVSimulator()
    model = Model()
    simulator.setup(model)

    with pytest.warns(FutureWarning, match="simulator.time is deprecated"):
        _ = simulator.time


def test_simulation_event():
    """Tests for Event class."""
    some_test_function = MagicMock()

    time = 10
    event = Event(
        time,
        some_test_function,
        priority=Priority.DEFAULT,
        function_args=[],
        function_kwargs={},
    )

    assert event.time == time
    assert event.fn() is some_test_function
    assert event.function_args == []
    assert event.function_kwargs == {}
    assert event.priority == Priority.DEFAULT

    # execute
    event.execute()
    some_test_function.assert_called_once()

    with pytest.raises(Exception):
        Event(
            time, None, priority=Priority.DEFAULT, function_args=[], function_kwargs={}
        )

    class NonWeakRefCallable:
        __slots__ = ()

        def __call__(self):
            return None

    with pytest.raises(TypeError, match="function must be weak referenceable"):
        Event(time, NonWeakRefCallable(), priority=Priority.DEFAULT)

    try:
        Event(time, lambda: None, priority=Priority.DEFAULT)
    except ValueError as exc:
        assert "function must be alive at Event creation." in str(exc)
    else:
        pytest.fail("Expected ValueError for inline lambda callback")

    Event(time, partial(some_test_function, "x"), priority=Priority.DEFAULT)

    lambda_called = []

    def callback():
        lambda_called.append("fired")

    Event(time, callback, priority=Priority.DEFAULT).execute()
    assert lambda_called == ["fired"]

    partial_target = MagicMock()
    partial_callback = partial(partial_target, "x")
    Event(time, partial_callback, priority=Priority.DEFAULT).execute()
    partial_target.assert_called_once_with("x")

    # check calling with arguments
    some_test_function = MagicMock()
    event = Event(
        time,
        some_test_function,
        priority=Priority.DEFAULT,
        function_args=["1"],
        function_kwargs={"x": 2},
    )
    event.execute()
    some_test_function.assert_called_once_with("1", x=2)

    # check if we pass over deletion of callable silently because of weakrefs
    def some_test_function(x, y):
        return x + y

    event = Event(time, some_test_function, priority=Priority.DEFAULT)
    del some_test_function
    event.execute()

    # cancel
    some_test_function = MagicMock()
    event = Event(
        time,
        some_test_function,
        priority=Priority.DEFAULT,
        function_args=["1"],
        function_kwargs={"x": 2},
    )
    event.cancel()
    assert event.fn is None
    assert event.function_args == []
    assert event.function_kwargs == {}
    assert event.priority == Priority.DEFAULT
    assert event.CANCELED

    # comparison for sorting
    event1 = Event(
        10,
        some_test_function,
        priority=Priority.DEFAULT,
        function_args=[],
        function_kwargs={},
    )
    event2 = Event(
        10,
        some_test_function,
        priority=Priority.DEFAULT,
        function_args=[],
        function_kwargs={},
    )
    assert event1 < event2  # based on just unique_id as tiebraker

    event1 = Event(
        11,
        some_test_function,
        priority=Priority.DEFAULT,
        function_args=[],
        function_kwargs={},
    )
    event2 = Event(
        10,
        some_test_function,
        priority=Priority.DEFAULT,
        function_args=[],
        function_kwargs={},
    )
    assert event1 > event2

    event1 = Event(
        10,
        some_test_function,
        priority=Priority.DEFAULT,
        function_args=[],
        function_kwargs={},
    )
    event2 = Event(
        10,
        some_test_function,
        priority=Priority.HIGH,
        function_args=[],
        function_kwargs={},
    )
    assert event1 > event2


def test_simulation_event_pickle():
    """Test pickling and unpickling of Event."""

    # Test with regular function
    def test_fn():
        return "test"

    event = Event(
        10.0,
        test_fn,
        priority=Priority.HIGH,
        function_args=["arg1"],
        function_kwargs={"key": "value"},
    )

    # Pickle and unpickle
    state = event.__getstate__()
    assert state["_fn_strong"] is test_fn
    assert state["fn"] is None

    new_event = Event.__new__(Event)
    new_event.__setstate__(state)

    assert new_event.time == 10.0
    assert new_event.priority == Priority.HIGH.value
    assert new_event.function_args == ["arg1"]
    assert new_event.function_kwargs == {"key": "value"}
    assert new_event.fn() is test_fn

    # Test with canceled event
    event.cancel()
    state = event.__getstate__()
    assert state["_fn_strong"] is None

    new_event = Event.__new__(Event)
    new_event.__setstate__(state)
    assert new_event.fn is None


def test_eventlist():
    """Tests for EventList."""
    event_list = EventList()

    assert len(event_list._events) == 0
    assert isinstance(event_list._events, list)
    assert event_list.is_empty()

    # add event
    some_test_function = MagicMock()
    event = Event(
        1,
        some_test_function,
        priority=Priority.DEFAULT,
        function_args=[],
        function_kwargs={},
    )
    event_list.add_event(event)
    assert len(event_list) == 1
    assert event in event_list

    # remove event
    event_list.remove(event)
    assert len(event_list) == 0
    assert event not in event_list
    assert event.CANCELED

    # peak ahead
    event_list = EventList()
    for i in range(10):
        event = Event(
            i,
            some_test_function,
            priority=Priority.DEFAULT,
            function_args=[],
            function_kwargs={},
        )
        event_list.add_event(event)
    events = event_list.peek_ahead(2)
    assert len(events) == 2
    assert events[0].time == 0
    assert events[1].time == 1

    events = event_list.peek_ahead(11)
    assert len(events) == 10

    event_list._events[6].cancel()
    events = event_list.peek_ahead(10)
    assert len(events) == 9

    event_list = EventList()
    with pytest.raises(Exception):
        event_list.peek_ahead()

    # peek_ahead should return events in chronological order
    # This tests the fix for heap iteration bug where events were returned
    event_list = EventList()
    some_test_function = MagicMock()
    times = [5.0, 15.0, 10.0, 25.0, 20.0, 8.0]
    for t in times:
        event = Event(
            t,
            some_test_function,
            priority=Priority.DEFAULT,
            function_args=[],
            function_kwargs={},
        )
        event_list.add_event(event)

    events = event_list.peek_ahead(5)
    event_times = [e.time for e in events]
    # Events should be in chronological order
    assert event_times == sorted(times)[:5]

    # pop event
    event_list = EventList()
    for i in range(10):
        event = Event(
            i,
            some_test_function,
            priority=Priority.DEFAULT,
            function_args=[],
            function_kwargs={},
        )
        event_list.add_event(event)
    event = event_list.pop_event()
    assert event.time == 0

    event_list = EventList()
    event = Event(
        9,
        some_test_function,
        priority=Priority.DEFAULT,
        function_args=[],
        function_kwargs={},
    )
    event_list.add_event(event)
    event.cancel()
    with pytest.raises(Exception):
        event_list.pop_event()

    # clear
    event_list.clear()
    assert len(event_list) == 0


def test_simulator_uses_model_event_list():
    """Test that simulator uses model's internal event list."""
    model = Model()
    simulator = DEVSimulator()
    simulator.setup(model)

    # Simulator's event_list property should return model's event list
    assert simulator.event_list is model._event_list

    # Events scheduled through simulator appear in model's event list
    fn = MagicMock()
    simulator.schedule_event_absolute(fn, 1.0)
    assert len(model._event_list) == 1
