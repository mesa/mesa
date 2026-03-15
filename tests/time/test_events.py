"""Tests for Event, EventList, Schedule, and EventGenerator."""
# ruff: noqa: D101, D102

import gc
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

# Ignore deprecation warnings in this test file
pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------
class TestSchedule:
    def test_defaults(self):
        s = Schedule()
        assert s.interval == 1
        assert s.start is None
        assert s.end is None
        assert s.count is None

    def test_custom_values(self):
        s = Schedule(start=5, end=10, interval=2)
        assert s.start == 5
        assert s.end == 10
        assert s.interval == 2
        assert s.count is None

    def test_with_count(self):
        s = Schedule(start=5, interval=2, count=5)
        assert s.count == 5
        assert s.end is None

    def test_callable_interval(self):
        s = Schedule(start=5, interval=lambda m: m.time + 1, count=5)
        assert isinstance(s.interval, Callable)

    def test_rejects_nonpositive_interval(self):
        with pytest.raises(ValueError):
            Schedule(interval=0)
        with pytest.raises(ValueError):
            Schedule(interval=-1)

    def test_rejects_nonpositive_count(self):
        with pytest.raises(ValueError):
            Schedule(interval=1.0, count=0)
        with pytest.raises(ValueError):
            Schedule(interval=1.0, count=-5)

    def test_start_after_end_raises(self):
        with pytest.raises(ValueError):
            Schedule(interval=1.0, start=10, end=5)


# ---------------------------------------------------------------------------
# Event — creation
# ---------------------------------------------------------------------------
class TestEventCreation:
    def test_basic_attributes(self):
        fn = MagicMock()
        event = Event(10, fn, priority=Priority.DEFAULT)
        assert event.time == 10
        assert event.fn() is fn
        assert event.priority == Priority.DEFAULT
        assert event.function_args == []
        assert event.function_kwargs == {}

    def test_with_args_and_kwargs(self):
        fn = MagicMock()
        event = Event(
            10,
            fn,
            priority=Priority.DEFAULT,
            function_args=["1"],
            function_kwargs={"x": 2},
        )
        assert event.function_args == ["1"]
        assert event.function_kwargs == {"x": 2}

    def test_rejects_non_callable(self):
        with pytest.raises(TypeError, match="function must be a callable"):
            Event(10, None, priority=Priority.DEFAULT)

    def test_rejects_non_weakrefable(self):
        class NoWeakRef:
            __slots__ = ()

            def __call__(self):
                return None

        with pytest.raises(TypeError, match="function must be weak referenceable"):
            Event(10, NoWeakRef(), priority=Priority.DEFAULT)

    def test_rejects_inline_lambda(self):
        with pytest.raises(
            ValueError, match="function must be alive at Event creation"
        ):
            Event(10, lambda: None, priority=Priority.DEFAULT)

    def test_accepts_partial(self):
        fn = MagicMock()
        Event(10, partial(fn, "x"), priority=Priority.DEFAULT)


# ---------------------------------------------------------------------------
# Event — execution
# ---------------------------------------------------------------------------
class TestEventExecution:
    def test_basic_execution(self):
        fn = MagicMock()
        event = Event(10, fn, priority=Priority.DEFAULT)
        event.execute()
        fn.assert_called_once()

    def test_execution_with_arguments(self):
        fn = MagicMock()
        event = Event(
            10,
            fn,
            priority=Priority.DEFAULT,
            function_args=["1"],
            function_kwargs={"x": 2},
        )
        event.execute()
        fn.assert_called_once_with("1", x=2)

    def test_partial_execution(self):
        fn = MagicMock()
        callback = partial(fn, "x")
        Event(10, callback, priority=Priority.DEFAULT).execute()
        fn.assert_called_once_with("x")

    def test_silent_noop_when_weakref_dead(self):
        def temp_fn(x, y):
            return x + y

        event = Event(10, temp_fn, priority=Priority.DEFAULT)
        del temp_fn
        event.execute()  # should not raise

    def test_named_callback_executes(self):
        called = []

        def callback():
            called.append("fired")

        Event(10, callback, priority=Priority.DEFAULT).execute()
        assert called == ["fired"]


# ---------------------------------------------------------------------------
# Event — cancellation
# ---------------------------------------------------------------------------
class TestEventCancellation:
    def test_cancel(self):
        fn = MagicMock()
        event = Event(
            10,
            fn,
            priority=Priority.DEFAULT,
            function_args=["1"],
            function_kwargs={"x": 2},
        )
        event.cancel()
        assert event.CANCELED
        assert event.fn is None
        assert event.function_args == []
        assert event.function_kwargs == {}

    def test_canceled_event_not_executed(self):
        fn = MagicMock()
        event = Event(10, fn, priority=Priority.DEFAULT)
        event.cancel()
        event.execute()
        fn.assert_not_called()


# ---------------------------------------------------------------------------
# Event — ordering
# ---------------------------------------------------------------------------
class TestEventOrdering:
    def test_earlier_time_is_less(self):
        fn = MagicMock()
        e1 = Event(9, fn, priority=Priority.DEFAULT)
        e2 = Event(10, fn, priority=Priority.DEFAULT)
        assert e1 < e2

    def test_later_time_is_greater(self):
        fn = MagicMock()
        e1 = Event(11, fn, priority=Priority.DEFAULT)
        e2 = Event(10, fn, priority=Priority.DEFAULT)
        assert e1 > e2

    def test_higher_priority_is_less_at_same_time(self):
        fn = MagicMock()
        e_default = Event(10, fn, priority=Priority.DEFAULT)
        e_high = Event(10, fn, priority=Priority.HIGH)
        assert e_high < e_default

    def test_unique_id_tiebreaker(self):
        fn = MagicMock()
        e1 = Event(10, fn, priority=Priority.DEFAULT)
        e2 = Event(10, fn, priority=Priority.DEFAULT)
        assert e1 < e2  # earlier unique_id wins


# ---------------------------------------------------------------------------
# Event — pickling
# ---------------------------------------------------------------------------
class TestEventPickle:
    def test_getstate_setstate(self):
        def test_fn():
            return "test"

        event = Event(
            10.0,
            test_fn,
            priority=Priority.HIGH,
            function_args=["arg1"],
            function_kwargs={"key": "value"},
        )

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

    def test_canceled_event_pickle(self):
        def test_fn():
            return "test"

        event = Event(10.0, test_fn, priority=Priority.HIGH)
        event.cancel()

        state = event.__getstate__()
        assert state["_fn_strong"] is None

        new_event = Event.__new__(Event)
        new_event.__setstate__(state)
        assert new_event.fn is None


# ---------------------------------------------------------------------------
# EventList
# ---------------------------------------------------------------------------
class TestEventListBasics:
    def test_empty_on_init(self):
        el = EventList()
        assert el.is_empty()
        assert len(el) == 0

    def test_add_and_contains(self):
        el = EventList()
        fn = MagicMock()
        event = Event(1, fn, priority=Priority.DEFAULT)
        el.add_event(event)
        assert len(el) == 1
        assert event in el

    def test_remove(self):
        el = EventList()
        fn = MagicMock()
        event = Event(1, fn, priority=Priority.DEFAULT)
        el.add_event(event)
        el.remove(event)
        assert len(el) == 0
        assert event.CANCELED
        assert event not in el

    def test_clear(self):
        el = EventList()
        fn = MagicMock()
        for i in range(5):
            el.add_event(Event(i, fn, priority=Priority.DEFAULT))
        el.clear()
        assert len(el) == 0


class TestEventListPeekAhead:
    def test_basic_peek(self):
        el = EventList()
        fn = MagicMock()
        for i in range(10):
            el.add_event(Event(i, fn, priority=Priority.DEFAULT))

        events = el.peek_ahead(2)
        assert len(events) == 2
        assert events[0].time == 0
        assert events[1].time == 1

    def test_peek_more_than_available(self):
        el = EventList()
        fn = MagicMock()
        for i in range(10):
            el.add_event(Event(i, fn, priority=Priority.DEFAULT))

        events = el.peek_ahead(11)
        assert len(events) == 10

    def test_peek_skips_canceled(self):
        el = EventList()
        fn = MagicMock()
        for i in range(10):
            el.add_event(Event(i, fn, priority=Priority.DEFAULT))

        el._events[6].cancel()
        events = el.peek_ahead(10)
        assert len(events) == 9

    def test_peek_empty_raises(self):
        el = EventList()
        with pytest.raises(IndexError):
            el.peek_ahead()

    def test_peek_returns_chronological_order(self):
        el = EventList()
        fn = MagicMock()
        times = [5.0, 15.0, 10.0, 25.0, 20.0, 8.0]
        for t in times:
            el.add_event(Event(t, fn, priority=Priority.DEFAULT))

        events = el.peek_ahead(5)
        event_times = [e.time for e in events]
        assert event_times == sorted(times)[:5]


class TestEventListPop:
    def test_pop_returns_earliest(self):
        el = EventList()
        fn = MagicMock()
        for i in range(10):
            el.add_event(Event(i, fn, priority=Priority.DEFAULT))

        event = el.pop_event()
        assert event.time == 0

    def test_pop_skips_canceled(self):
        el = EventList()
        fn = MagicMock()
        event = Event(9, fn, priority=Priority.DEFAULT)
        el.add_event(event)
        event.cancel()
        with pytest.raises(IndexError):
            el.pop_event()

    def test_event_id_tie_breaking(self):
        """Events with identical time and priority execute in event_id order."""
        el = EventList()
        execution_order = []

        def make_fn(i):
            def fn():
                execution_order.append(i)

            return fn

        functions = [make_fn(i) for i in range(10)]
        events = [Event(5, fn, priority=Priority.DEFAULT) for fn in functions]

        for e in reversed(events):
            el.add_event(e)

        while not el.is_empty():
            el.pop_event().execute()

        assert execution_order == list(range(10))

    def test_recursive_same_timestamp(self):
        """Events scheduled at same timestamp during execution execute in same cycle."""
        el = EventList()
        trace = []

        def event_b():
            trace.append("B")

        def event_a():
            trace.append("A")
            el.add_event(Event(5, event_b, priority=Priority.DEFAULT))

        el.add_event(Event(5, event_a, priority=Priority.DEFAULT))

        while not el.is_empty():
            el.pop_event().execute()

        assert trace == ["A", "B"]

    def test_skips_canceled_events(self):
        """Canceled events are never executed."""
        el = EventList()
        execution = []

        def make_fn(i):
            def fn():
                execution.append(i)

            return fn

        functions = [make_fn(i) for i in range(10)]
        events = []
        for fn in functions:
            e = Event(5, fn, priority=Priority.DEFAULT)
            events.append(e)
            el.add_event(e)

        for e in events[:5]:
            e.cancel()

        while not el.is_empty():
            el.pop_event().execute()

        assert execution == list(range(5, 10))


class TestEventListCompact:
    def test_compact_removes_canceled(self):
        el = EventList()
        fn = MagicMock()

        events = []
        for i in range(10):
            e = Event(i, fn, priority=Priority.DEFAULT)
            events.append(e)
            el.add_event(e)

        for e in events[:6]:
            e.cancel()

        assert len(el._events) == 10
        el.compact()
        assert len(el._events) == 4

        remaining = []
        while not el.is_empty():
            remaining.append(el.pop_event().time)

        assert remaining == [6, 7, 8, 9]


# ---------------------------------------------------------------------------
# EventGenerator
# ---------------------------------------------------------------------------
@pytest.fixture
def setup():
    """Create a model and mock function."""
    model = Model()
    return model, MagicMock()


class TestEventGeneratorInit:
    def test_defaults(self, setup):
        model, fn = setup
        schedule = Schedule(interval=5.0, start=10, end=100, count=5)
        gen = EventGenerator(model, fn, schedule)

        assert gen.model is model
        assert gen.schedule is schedule
        assert gen.priority == Priority.DEFAULT
        assert not gen.is_active
        assert gen.execution_count == 0

    def test_custom_priority(self, setup):
        model, fn = setup
        gen = EventGenerator(model, fn, Schedule(), priority=Priority.HIGH)
        assert gen.priority == Priority.HIGH

    def test_rejects_non_callable(self):
        model = Model()
        with pytest.raises(TypeError):
            EventGenerator(model, 42, Schedule(interval=1.0))

    def test_rejects_non_weakrefable(self):
        model = Model()

        class NoWeakRef:
            __slots__ = ()

            def __call__(self):
                pass

        with pytest.raises(TypeError):
            EventGenerator(model, NoWeakRef(), Schedule(interval=1.0))

    def test_rejects_lambda(self):
        model = Model()
        with pytest.raises(ValueError, match="alive"):
            EventGenerator(model, lambda: 10, Schedule(interval=1.0))


class TestEventGeneratorStartStop:
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
        model, fn = setup
        gen = EventGenerator(model, fn, Schedule(interval=1.0))
        gen.start().stop()
        assert not gen.is_active

        model.run_for(5.0)
        fn.assert_not_called()


class TestEventGeneratorExecution:
    def test_recurring_execution(self, setup):
        model, fn = setup
        gen = EventGenerator(model, fn, Schedule(interval=2.0, start=0.0))
        gen.start()

        model.run_for(7.0)
        assert fn.call_count == 4  # t=0, 2, 4, 6
        assert gen.execution_count == 4

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

    def test_count_reached_before_end(self, setup):
        """Test count reached before end."""
        model, fn = setup
        gen = EventGenerator(
            model, fn, Schedule(interval=1.0, start=0.0, end=100, count=2)
        )
        gen.start()
        model.run_for(10.0)
        assert fn.call_count == 2

    def test_callable_interval(self, setup):
        """Test callable interval evaluated each time."""
        model, fn = setup
        intervals = iter([1.0, 2.0, 1.0, 1.0])
        schedule = Schedule(interval=lambda m: next(intervals), start=0.0)
        gen = EventGenerator(model, fn, schedule)
        gen.start()

        model.run_for(4.5)
        assert fn.call_count == 4  # t=0, 1, 3, 4

    def test_callable_interval_negative_raises(self, setup):
        """Test if callable interval raises exception if return is negative."""
        model, fn = setup
        gen = EventGenerator(model, fn, Schedule(start=1, interval=lambda m: -0.5))
        gen.start()

        with pytest.raises(ValueError):
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
            model, func_a, Schedule(interval=1.0, start=1.0), Priority.LOW
        )
        high = EventGenerator(
            model, func_b, Schedule(interval=1.0, start=1.0), Priority.HIGH
        )
        low.start()
        high.start()

        model.run_for(1.5)
        assert order == ["H", "L"]


class TestEventGeneratorIntrospection:
    def test_next_scheduled_time(self, setup):
        model, fn = setup
        gen = EventGenerator(model, fn, Schedule(interval=2.0))

        assert not gen.is_active
        assert gen.next_scheduled_time is None

        gen.start()
        assert gen.is_active
        assert gen.next_scheduled_time == model.time + 2.0

        gen.stop()
        assert not gen.is_active
        assert gen.next_scheduled_time is None


class TestEventGeneratorPauseResume:
    def test_full_lifecycle(self, setup):
        model, fn = setup
        gen = EventGenerator(model, fn, Schedule(interval=1.0))
        gen.start()

        model.run_for(1.0)
        assert fn.call_count == 1

        gen.pause()
        model.run_for(10.0)
        assert fn.call_count == 1

        gen.resume()
        model.run_for(1.0)
        assert fn.call_count == 2

        gen.pause()
        model.run_for(5.0)
        assert fn.call_count == 2

        gen.resume()
        model.run_for(2.0)
        assert fn.call_count == 4

    def test_pause_idempotent(self, setup):
        """Calling pause multiple times should be safe."""
        model, fn = setup
        gen = EventGenerator(model, fn, Schedule(interval=1.0))
        gen.start()
        gen.pause()
        gen.pause()
        gen.pause()
        model.run_for(5.0)
        assert fn.call_count == 0

    def test_resume_idempotent(self, setup):
        """Calling resume while running should do nothing."""
        model, fn = setup
        gen = EventGenerator(model, fn, Schedule(interval=1.0))
        gen.start()
        gen.resume()
        gen.resume()
        model.run_for(1.0)
        assert fn.call_count == 1

    def test_pause_during_execution(self, setup):
        """Pause called inside callback should prevent future scheduling."""
        model, _ = setup
        call_count = {"n": 0}

        def fn():
            call_count["n"] += 1
            if call_count["n"] == 1:
                gen.pause()

        gen = EventGenerator(model, fn, Schedule(interval=1.0))
        gen.start()
        model.run_for(5.0)
        assert call_count["n"] == 1

    def test_stop_while_paused(self, setup):
        """Stopping while paused should fully deactivate generator."""
        model, fn = setup
        gen = EventGenerator(model, fn, Schedule(interval=1.0))
        gen.start()
        gen.pause()
        gen.stop()
        model.run_for(5.0)
        assert fn.call_count == 0
        assert not gen.is_active

    def test_resume_after_stop(self, setup):
        """Resume should do nothing after stop."""
        model, fn = setup
        gen = EventGenerator(model, fn, Schedule(interval=1.0))
        gen.start()
        gen.stop()
        gen.resume()
        model.run_for(5.0)
        assert fn.call_count == 0

    def test_next_scheduled_time_updates(self, setup):
        """next_scheduled_time should reflect pause/resume state."""
        model, fn = setup
        gen = EventGenerator(model, fn, Schedule(interval=2.0))
        gen.start()
        assert gen.next_scheduled_time is not None

        gen.pause()
        assert gen.next_scheduled_time is None

        gen.resume()
        assert gen.next_scheduled_time is not None

    def test_pause_before_start_is_safe(self, setup):
        """Pausing before start should be a no-op."""
        model, fn = setup
        gen = EventGenerator(model, fn, Schedule(interval=1.0))
        gen.pause()  # should not crash
        model.run_for(5.0)
        assert fn.call_count == 0

    def test_resume_schedules_from_current_time(self, setup):
        """Resume should schedule next execution relative to current time."""
        model, fn = setup
        gen = EventGenerator(model, fn, Schedule(interval=2.0))
        gen.start()

        model.run_for(1.0)
        gen.pause()
        model.run_for(5.0)  # time advances while paused

        gen.resume()
        model.run_for(1.9)
        assert fn.call_count == 0
        model.run_for(0.1)
        assert fn.call_count == 1


class TestEventGeneratorPickle:
    def test_getstate_setstate(self):
        model = Model()

        def test_func():
            return "hello"

        gen = EventGenerator(model, test_func, Schedule(interval=1.0))

        state = gen.__getstate__()
        assert state["_fn_strong"] is not None
        assert state["_fn_strong"]() == "hello"
        assert state["function"] is None

        new_gen = EventGenerator.__new__(EventGenerator)
        new_gen.__setstate__(state)
        assert new_gen.function() is not None
        assert new_gen.function()() == "hello"

    def test_getstate_setstate_with_none(self):
        state = {
            "_fn_strong": None,
            "function": None,
            "schedule": Schedule(interval=1.0),
            "priority": Priority.DEFAULT,
            "_active": False,
            "_paused": False,
            "_current_event": None,
            "_execution_count": 0,
            "model": Model(),
        }

        gen = EventGenerator.__new__(EventGenerator)
        gen.__setstate__(state)
        assert gen.function is None


class TestEventGeneratorWeakref:
    def test_stops_silently_when_weakref_dies(self):
        model = Model()
        call_count = [0]

        def temp_func():
            call_count[0] += 1

        gen = EventGenerator(model, temp_func, Schedule(interval=1.0))
        gen.start()

        model.run_for(1.0)
        assert call_count[0] == 1
        assert gen.is_active

        del temp_func
        gc.collect()

        model.run_for(1.0)
        assert not gen.is_active
        assert call_count[0] == 1
