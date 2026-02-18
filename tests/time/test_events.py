"""Tests for experimental Simulator classes."""

from collections.abc import Callable
from functools import partial
from unittest.mock import MagicMock

import pytest

from mesa import Model
from mesa.time import (
    Event,
    EventGenerator,
    EventList,
    Priority,
    Schedule,
)

# Ignore deprecation warnings for Simulator classes in this test file
pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


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

    with pytest.raises(TypeError, match="function must be a callable"):
        Event(
            time, None, priority=Priority.DEFAULT, function_args=[], function_kwargs={}
        )

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


def test_schedule():
    """Tests for Schedule."""
    schedule = Schedule()
    assert schedule.start is None
    assert schedule.end is None
    assert schedule.count is None
    assert schedule.interval == 1

    schedule = Schedule(start=5, end=10, interval=2)
    assert schedule.start == 5
    assert schedule.end == 10
    assert schedule.count is None
    assert schedule.interval == 2

    schedule = Schedule(start=5, interval=2, count=5)
    assert schedule.start == 5
    assert schedule.end is None
    assert schedule.count == 5
    assert schedule.interval == 2

    schedule = Schedule(start=5, interval=lambda m: m.time + 1, count=5)
    assert schedule.start == 5
    assert schedule.end is None
    assert schedule.count == 5
    assert isinstance(schedule.interval, Callable)

    with pytest.raises(ValueError):
        _ = Schedule(start=10, end=5)

    with pytest.raises(ValueError):
        _ = Schedule(count=-1)

    with pytest.raises(ValueError):
        _ = Schedule(interval=-1)


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
    assert event.CANCELED
    assert event not in event_list

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
    assert len(event_list) == 0

    # clear
    event_list.clear()
    assert len(event_list) == 0


@pytest.fixture
def setup():
    """Create a model with simulator and mock function."""
    model = Model()
    return model, MagicMock()


class TestEventGenerator:
    """Tests for EventGenerator."""

    def test_init(self, setup):
        """Test initialization with Schedule."""
        model, fn = setup
        schedule = Schedule(interval=5.0, start=10, end=100, count=5)
        gen = EventGenerator(model, fn, schedule)

        assert gen.model is model
        assert gen.schedule is schedule
        assert gen.priority == Priority.DEFAULT
        assert not gen.is_active
        assert gen.execution_count == 0

    def test_init_with_priority(self, setup):
        """Test initialization with custom priority."""
        model, fn = setup
        gen = EventGenerator(model, fn, Schedule(), priority=Priority.HIGH)
        assert gen.priority == Priority.HIGH

    def test_start_with_schedule_start(self, setup):
        """Test start uses schedule.start time."""
        model, fn = setup
        gen = EventGenerator(model, fn, Schedule(interval=1.0, start=5.0))

        assert gen.start() is gen
        assert gen.is_active

        model.run_for(4.9)
        fn.assert_not_called()
        model.run_for(0.1)
        fn.assert_called_once()

    def test_start_without_schedule_start(self, setup):
        """Test start defaults to now + interval when schedule.start is None."""
        model, fn = setup
        gen = EventGenerator(model, fn, Schedule(interval=2.0))
        gen.start()

        model.run_for(1.9)
        fn.assert_not_called()
        model.run_for(0.1)
        fn.assert_called_once()

    def test_start_when_active_is_noop(self, setup):
        """Test that starting when active does nothing."""
        model, fn = setup
        gen = EventGenerator(model, fn, Schedule(interval=1.0))
        gen.start()
        event_count = len(model._event_list)

        gen.start()  # No-op
        assert len(model._event_list) == event_count

    def test_stop(self, setup):
        """Test immediate stop."""
        model, fn = setup
        gen = EventGenerator(model, fn, Schedule(interval=1.0))
        gen.start().stop()
        assert not gen.is_active

        model.run_for(5.0)
        fn.assert_not_called()

    def test_schedule_end(self, setup):
        """Test schedule.end stops execution."""
        model, fn = setup
        gen = EventGenerator(model, fn, Schedule(interval=1.0, start=0.0, end=2.5))
        gen.start()

        model.run_for(5.0)
        assert fn.call_count == 3  # t=0, 1, 2
        assert not gen.is_active

    def test_schedule_count(self, setup):
        """Test schedule.count limits executions."""
        model, fn = setup
        gen = EventGenerator(model, fn, Schedule(interval=1.0, start=0.0, count=3))
        gen.start()

        model.run_for(10.0)
        assert fn.call_count == 3
        assert gen.execution_count == 3
        assert not gen.is_active

    def test_schedule_end_and_count(self, setup):
        """Test count reached before end."""
        model, fn = setup
        gen = EventGenerator(
            model, fn, Schedule(interval=1.0, start=0.0, end=100, count=2)
        )
        gen.start()

        model.run_for(10.0)
        assert fn.call_count == 2

    def test_recurring_execution(self, setup):
        """Test recurring execution and count tracking."""
        model, fn = setup
        gen = EventGenerator(model, fn, Schedule(interval=2.0, start=0.0))
        gen.start()

        model.run_for(7.0)
        assert fn.call_count == 4  # t=0, 2, 4, 6
        assert gen.execution_count == 4

    def test_callable_interval(self, setup):
        """Test callable interval evaluated each time."""
        model, fn = setup
        intervals = iter([1.0, 2.0, 1.0, 1.0])
        schedule = Schedule(interval=lambda m: next(intervals), start=0.0)
        gen = EventGenerator(model, fn, schedule)
        gen.start()

        model.run_for(4.5)
        assert fn.call_count == 4  # t=0, 1, 3, 4

    def test_callable_interval_raises(self, setup):
        """Test if callable interval raises exception if return is negative."""
        with pytest.raises(ValueError):
            model, fn = setup
            gen = EventGenerator(model, fn, Schedule(start=1, interval=lambda m: -0.5))
            gen.start()

            model.run_for(3)

    def test_functools_partial(self, setup):
        """Test using functools.partial for arguments."""
        model, fn = setup
        partial_func = partial(fn, "a", k="v")

        gen = EventGenerator(model, partial_func, Schedule(interval=1.0, start=0.0))
        gen.start()

        model.run_for(0.5)
        fn.assert_called_once_with("a", k="v")

    def test_priority_ordering(self, setup):
        """Test priority affects execution order at same time."""
        model, _ = setup
        order = []

        def func_a():
            order.append("L")

        def func_b():
            order.append("H")

        low = EventGenerator(
            model,
            func_a,
            Schedule(interval=1.0, start=1.0),
            Priority.LOW,
        )
        high = EventGenerator(
            model,
            func_b,
            Schedule(interval=1.0, start=1.0),
            Priority.HIGH,
        )

        low.start()
        high.start()

        model.run_for(1.5)
        assert order == ["H", "L"]
