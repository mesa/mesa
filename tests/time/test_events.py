"""Tests for Event, EventList, Schedule, and EventGenerator."""
# ruff: noqa: D101, D102

import gc
import math
import pickle
from collections.abc import Callable
from functools import partial
from unittest.mock import MagicMock

import pytest

from mesa import Model
from mesa.time import (
    Event,
    EventGenerator,
    EventGeneratorState,
    EventList,
    Priority,
    Schedule,
)

# Ignore deprecation warnings in this test file
pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


def _noop():
    """Module-level callback, so Event's weak reference stays alive."""


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

    @pytest.mark.parametrize("interval", [math.inf, math.nan])
    def test_rejects_non_finite_interval(self, interval):
        """NaN slipped past the old <= 0 check and broke event ordering."""
        with pytest.raises(ValueError, match="finite"):
            Schedule(interval=interval)

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

    def test_remove_event_from_different_list(self):
        """Removing an event from another list must not cancel it."""
        queue_a = EventList()
        queue_b = EventList()
        fn = MagicMock()

        event = Event(5, fn)
        queue_b.add_event(event)

        queue_a.remove(event)

        assert not event.CANCELED
        assert len(queue_b) == 1
        assert event in queue_b

        queue_b.pop_event().execute()
        fn.assert_called_once()


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


def _picklable_handler():
    """Module-level callback, so events holding it can be pickled."""


def _recount(el):
    """Count the canceled events actually on the heap."""
    return sum(1 for e in el._events if e.CANCELED)


class TestEventListCompact:
    def test_compact_removes_canceled(self, monkeypatch):
        # ratio 1.0 needs more dead events than the heap holds, so automatic
        # compaction never fires and compact() can be exercised on its own
        monkeypatch.setattr(EventList, "COMPACTION_RATIO", 1.0)
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

    def test_compacts_when_tombstones_dominate(self):
        """Cancelling alone rebuilds the heap; no further push is needed."""
        el = EventList()
        fn = MagicMock()

        events = [Event(i, fn) for i in range(10)]
        for e in events:
            el.add_event(e)
        for e in events:
            e.cancel()

        assert len(el._events) < 10
        assert len(el) == 0
        assert el._n_canceled == _recount(el)

    @pytest.mark.parametrize("ratio", [0.1, 0.25, 0.5])
    def test_ratio_bounds_the_peak_heap(self, monkeypatch, ratio):
        """The ratio caps the heap at L / (1 - ratio)."""
        monkeypatch.setattr(EventList, "COMPACTION_RATIO", ratio)
        el = EventList()
        fn = MagicMock()

        live = [Event(float(i), fn) for i in range(100)]
        for event in live:
            el.add_event(event)

        peak = len(el._events)
        for i in range(600):
            live[i % 100].cancel()
            replacement = Event(1000.0 + i, fn)
            el.add_event(replacement)
            live[i % 100] = replacement
            peak = max(peak, len(el._events))

        assert peak <= 100 / (1 - ratio) + 1

    def test_compaction_preserves_ordering(self):
        el = EventList()
        fn = MagicMock()

        events = [Event(float(i), fn) for i in range(12)]
        for e in events:
            el.add_event(e)
        for e in events[:8]:
            e.cancel()

        el.add_event(Event(0.5, fn))

        drained = []
        while not el.is_empty():
            drained.append(el.pop_event().time)

        assert drained == [0.5, 8.0, 9.0, 10.0, 11.0]


class TestEventListCancelCount:
    @pytest.fixture(autouse=True)
    def _no_automatic_compaction(self, monkeypatch):
        """Observe the count itself; rebuilding is TestEventListCompact's job.

        A ratio of 1.0 would need more canceled events than the heap holds, so
        the trigger never fires and tombstones stay put to be counted.
        """
        monkeypatch.setattr(EventList, "COMPACTION_RATIO", 1.0)

    def test_len_excludes_canceled(self):
        el = EventList()
        fn = MagicMock()
        events = [Event(i, fn) for i in range(5)]
        for e in events:
            el.add_event(e)

        events[0].cancel()
        events[3].cancel()

        assert len(el) == 3
        assert len(el._events) == 5
        assert el._n_canceled == _recount(el)

    def test_count_tracks_mixed_operations(self):
        el = EventList()
        fn = MagicMock()
        events = [Event(i, fn) for i in range(12)]
        for e in events:
            el.add_event(e)

        events[1].cancel()
        events[2].cancel()
        assert el._n_canceled == _recount(el)

        el.pop_event()  # event 0, discards nothing
        assert el._n_canceled == _recount(el)

        el.pop_event()  # skips tombstones 1 and 2, returns event 3
        assert el._n_canceled == _recount(el)

        events[7].cancel()
        el.compact()
        assert el._n_canceled == _recount(el) == 0
        assert len(el) == len(el._events)

    def test_repeated_cancel_counts_once(self):
        el = EventList()
        event = Event(1.0, MagicMock())
        el.add_event(event)

        event.cancel()
        event.cancel()

        assert el._n_canceled == 1
        assert len(el) == 0

    def test_adding_a_canceled_event_raises(self):
        el = EventList()
        event = Event(1.0, MagicMock())
        event.cancel()

        with pytest.raises(ValueError, match="Cannot add a canceled event"):
            el.add_event(event)

        assert len(el._events) == 0
        assert el._n_canceled == 0

    def test_cancel_after_pop_is_not_counted(self):
        """A popped event has left the heap, so cancelling it must not count.

        Model.schedule_event returns the Event to the caller, so cancelling one
        that has already fired is reachable from user code.
        """
        el = EventList()
        fn = MagicMock()
        popped = Event(1.0, fn)
        el.add_event(popped)
        el.add_event(Event(2.0, fn))

        assert el.pop_event() is popped
        popped.cancel()

        assert el._n_canceled == 0
        assert len(el) == 1

    def test_adding_popped_event_again_restores_counting(self):
        el = EventList()
        event = Event(1.0, MagicMock())
        el.add_event(event)

        el.pop_event()
        el.add_event(event)
        event.cancel()

        assert el._n_canceled == 1
        assert len(el) == 0

    def test_clear_resets_count(self):
        el = EventList()
        event = Event(1.0, MagicMock())
        el.add_event(event)
        event.cancel()

        el.clear()

        assert el._n_canceled == 0
        assert len(el) == 0

    def test_cancel_after_clear_does_not_corrupt_count(self):
        """Canceling an event that was removed by clear() must not affect the list."""
        el = EventList()
        fn = MagicMock()
        stale = Event(1.0, fn)
        el.add_event(stale)
        el.clear()

        live = [Event(float(i), fn) for i in range(10)]
        for e in live:
            el.add_event(e)

        stale.cancel()

        assert el._n_canceled == 0
        assert len(el) == 10

    def test_is_empty_when_only_tombstones_remain(self):
        el = EventList()
        fn = MagicMock()
        events = [Event(i, fn) for i in range(3)]
        for e in events:
            el.add_event(e)
        for e in events:
            e.cancel()

        assert el.is_empty()
        assert len(el._events) == 3

    def test_cancel_after_list_is_collected(self):
        el = EventList()
        event = Event(1.0, MagicMock())
        el.add_event(event)

        del el
        gc.collect()

        event.cancel()  # must not raise
        assert event.CANCELED

    def test_pickle_roundtrip_keeps_counting(self):
        el = EventList()
        events = [Event(i, _picklable_handler) for i in range(4)]
        for e in events:
            el.add_event(e)
        events[0].cancel()

        restored = pickle.loads(pickle.dumps(el))  # noqa: S301

        assert len(restored) == 3
        live = next(e for e in restored._events if not e.CANCELED)
        live.cancel()
        assert restored._n_canceled == _recount(restored)


# ---------------------------------------------------------------------------
# EventGenerator
# ---------------------------------------------------------------------------
@pytest.fixture
def setup():
    """Create a model and mock function."""
    model = Model()
    return model, MagicMock()


class TestEventListDuplicateAdd:
    """Adding one Event twice to the same list corrupts the canceled count."""

    def test_duplicate_add_is_refused(self):
        """A second add of a queued event raises rather than double-queueing it."""
        event_list = EventList()
        event = Event(1.0, _noop, Priority.DEFAULT)
        event_list.add_event(event)

        with pytest.raises(ValueError, match="already in this event list"):
            event_list.add_event(event)

        assert len(event_list) == 1

    def test_len_and_is_empty_stay_consistent_while_draining(self):
        """len(), is_empty() and pop_event() must agree. See #3973.

        One event in two heap slots was counted once by cancel() and twice by
        pop_event(), so len() overcounted, is_empty() claimed events remained
        after the heap was drained, and the canceled total went negative --
        which also disables compaction, since a negative count never exceeds
        the ratio.
        """
        event_list = EventList()
        events = [Event(float(i), _noop, Priority.DEFAULT) for i in range(4)]
        for event in events:
            event_list.add_event(event)

        with pytest.raises(ValueError):
            event_list.add_event(events[0])

        events[0].cancel()
        live = sum(1 for e in event_list._events if not e.CANCELED)
        assert len(event_list) == live

        drained = 0
        while not event_list.is_empty():
            event_list.pop_event()
            drained += 1

        assert drained == live
        assert len(event_list) == 0
        assert event_list._n_canceled >= 0

    def test_ownership_still_transfers_between_lists(self):
        """Handing an event to a second list stays allowed, as documented."""
        first, second = EventList(), EventList()
        event = Event(1.0, _noop, Priority.DEFAULT)

        first.add_event(event)
        second.add_event(event)

        assert len(second) == 1

    def test_a_popped_event_can_be_added_again(self):
        """pop_event() releases ownership, so rescheduling the event is fine."""
        event_list = EventList()
        event = Event(1.0, _noop, Priority.DEFAULT)
        event_list.add_event(event)

        popped = event_list.pop_event()
        event_list.add_event(popped)

        assert len(event_list) == 1


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

    def test_rejects_nonpositive_callable_interval(self, setup):
        """A callable interval must follow the same > 0 rule as a fixed interval.

        ``Schedule(interval=0)`` is rejected at construction, but a callable that
        returns 0 was previously accepted, which would schedule the next event at
        the current time. The runtime check must reject both 0 and negatives.
        """
        model, fn = setup

        gen = EventGenerator(model, fn, Schedule(interval=lambda m: 0))
        with pytest.raises(ValueError, match="> 0"):
            gen._get_interval()

        gen = EventGenerator(model, fn, Schedule(interval=lambda m: -1))
        with pytest.raises(ValueError, match="> 0"):
            gen._get_interval()

        # A positive callable interval is returned unchanged.
        gen = EventGenerator(model, fn, Schedule(interval=lambda m: 2.0))
        assert gen._get_interval() == 2.0


class TestEventGeneratorStart:
    def test_start_with_schedule_start(self, setup):
        """Test start uses schedule.start time."""
        model, fn = setup
        gen = model.schedule_recurring(fn, Schedule(interval=1.0, start=5.0))

        assert gen.state is EventGeneratorState.ACTIVE
        assert gen.is_active

        model.run_for(4.9)
        fn.assert_not_called()
        model.run_for(0.1)
        fn.assert_called_once()

    def test_start_without_schedule_start(self, setup):
        """Test start defaults to now + interval when schedule.start is None."""
        model, fn = setup
        model.schedule_recurring(fn, Schedule(interval=2.0))

        model.run_for(1.9)
        fn.assert_not_called()
        model.run_for(0.1)
        fn.assert_called_once()

    def test_start_with_schedule_start_in_past_raises(self, setup):
        """Starting a generator whose schedule.start has passed must not rewind time."""
        model, fn = setup
        model.run_until(5.0)
        generators = set(model._event_generators)

        with pytest.raises(ValueError, match="in the past"):
            model.schedule_recurring(fn, Schedule(interval=1.0, start=1.0))
        assert model._event_generators == generators

    def test_starts_once(self, setup):
        """A generator runs one schedule; a fresh run is a new generator."""
        model, fn = setup
        gen = model.schedule_recurring(fn, Schedule(interval=1.0))

        with pytest.raises(ValueError, match="schedule_recurring"):
            gen._start()

    def test_new_generator_is_pending(self, setup):
        model, fn = setup
        gen = EventGenerator(model, fn, Schedule(interval=1.0))

        assert gen.state is EventGeneratorState.PENDING
        assert not gen.is_active
        assert gen.pause() is False
        with pytest.raises(ValueError, match="PENDING"):
            gen.resume()


class TestEventGeneratorCancel:
    def test_cancel(self, setup):
        model, fn = setup
        gen = model.schedule_recurring(fn, Schedule(interval=1.0))

        assert gen.cancel() is True
        assert gen.state is EventGeneratorState.CANCELED
        assert not gen.is_active
        assert gen not in model._event_generators

        model.run_for(5.0)
        fn.assert_not_called()

    def test_cancel_is_final(self, setup):
        model, fn = setup
        gen = model.schedule_recurring(fn, Schedule(interval=1.0))
        gen.cancel()

        assert gen.cancel() is False
        assert gen.pause() is False
        with pytest.raises(ValueError, match="CANCELED"):
            gen.resume()
        model.run_for(5.0)
        fn.assert_not_called()

    def test_cancel_while_paused(self, setup):
        model, fn = setup
        gen = model.schedule_recurring(fn, Schedule(interval=1.0))
        gen.pause()

        assert gen.cancel() is True
        model.run_for(5.0)
        fn.assert_not_called()
        assert (gen.is_active, gen.is_paused) == (False, False)

    def test_cancel_from_within_callback(self, setup):
        """Canceling from the callback must not reschedule a next event."""
        model, _ = setup
        calls = []
        gen = None

        def cb():
            calls.append(model.time)
            gen.cancel()

        gen = model.schedule_recurring(cb, Schedule(interval=1.0, start=1.0))
        model.run_for(5.0)

        assert calls == [1.0]
        assert gen.state is EventGeneratorState.CANCELED
        assert gen.next_scheduled_time is None

    def test_finished_is_final(self, setup):
        model, fn = setup
        gen = model.schedule_recurring(fn, Schedule(interval=1.0, count=2))
        model.run_for(5.0)

        assert gen.state is EventGeneratorState.FINISHED
        assert gen.cancel() is False
        with pytest.raises(ValueError, match="FINISHED"):
            gen.resume()
        model.run_for(5.0)
        assert fn.call_count == 2


class TestEventGeneratorExecution:
    def test_recurring_execution(self, setup):
        model, fn = setup
        gen = model.schedule_recurring(fn, Schedule(interval=2.0, start=0.0))

        model.run_for(7.0)
        assert fn.call_count == 4  # t=0, 2, 4, 6
        assert gen.execution_count == 4

    def test_schedule_end(self, setup):
        """Test schedule.end stops execution."""
        model, fn = setup
        gen = model.schedule_recurring(fn, Schedule(interval=1.0, start=0.0, end=2.5))

        model.run_for(5.0)
        assert fn.call_count == 3  # t=0, 1, 2
        assert gen.state is EventGeneratorState.FINISHED

    def test_first_event_after_end_does_not_fire(self, setup):
        """A generator whose first event would fall after schedule.end never fires."""
        model, fn = setup
        model.run_for(10.0)
        gen = model.schedule_recurring(fn, Schedule(interval=5.0, end=12.0))

        assert gen.state is EventGeneratorState.FINISHED
        assert gen not in model._event_generators

        model.run_for(20.0)
        fn.assert_not_called()

    def test_schedule_count(self, setup):
        """Test schedule.count limits executions."""
        model, fn = setup
        gen = model.schedule_recurring(fn, Schedule(interval=1.0, start=0.0, count=3))

        model.run_for(10.0)
        assert fn.call_count == 3
        assert gen.execution_count == 3
        assert gen.state is EventGeneratorState.FINISHED

    def test_count_reached_before_end(self, setup):
        """Test count reached before end."""
        model, fn = setup
        model.schedule_recurring(
            fn, Schedule(interval=1.0, start=0.0, end=100, count=2)
        )
        model.run_for(10.0)
        assert fn.call_count == 2

    def test_last_execution_computes_no_interval(self, setup):
        """Once the count is used up, the interval isn't asked for another time."""
        model, fn = setup
        asked = []

        def interval(m):
            asked.append(m.time)
            return 1.0

        gen = model.schedule_recurring(fn, Schedule(interval=interval, count=2))
        model.run_for(5.0)

        assert asked == [0.0, 1.0]
        assert gen.state is EventGeneratorState.FINISHED

    def test_callable_interval(self, setup):
        """Test callable interval evaluated each time."""
        model, fn = setup
        intervals = iter([1.0, 2.0, 1.0, 1.0])
        schedule = Schedule(interval=lambda m: next(intervals), start=0.0)
        model.schedule_recurring(fn, schedule)

        model.run_for(4.5)
        assert fn.call_count == 4  # t=0, 1, 3, 4

    def test_functools_partial(self, setup):
        """Test using functools.partial for arguments."""
        model, fn = setup
        partial_func = partial(fn, "a", k="v")
        model.schedule_recurring(partial_func, Schedule(interval=1.0, start=0.0))

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

        model.schedule_recurring(
            func_a, Schedule(interval=1.0, start=1.0), Priority.LOW
        )
        model.schedule_recurring(
            func_b, Schedule(interval=1.0, start=1.0), Priority.HIGH
        )

        model.run_for(1.5)
        assert order == ["H", "L"]


class TestEventGeneratorIntrospection:
    def test_next_scheduled_time(self, setup):
        model, fn = setup
        gen = EventGenerator(model, fn, Schedule(interval=2.0))

        assert not gen.is_active
        assert gen.next_scheduled_time is None

        gen._start()
        assert gen.is_active
        assert gen.next_scheduled_time == model.time + 2.0

        gen.cancel()
        assert not gen.is_active
        assert gen.next_scheduled_time is None

    def test_is_paused_through_lifecycle(self, setup):
        """is_paused tells a paused generator apart from a running or ended one."""
        model, fn = setup
        gen = model.schedule_recurring(fn, Schedule(interval=1.0))
        assert (gen.is_active, gen.is_paused) == (True, False)

        gen.pause()
        assert (gen.is_active, gen.is_paused) == (True, True)
        assert gen.state is EventGeneratorState.PAUSED

        gen.resume()
        assert (gen.is_active, gen.is_paused) == (True, False)

        # canceling a paused generator clears the paused state too
        gen.pause()
        gen.cancel()
        assert (gen.is_active, gen.is_paused) == (False, False)

    def test_is_paused_inside_callback(self, setup):
        """is_paused gives a direct answer inside the generator's own callback."""
        model, _ = setup
        seen = []
        gen = None

        def cb():
            # while firing, nothing is scheduled yet
            seen.append((gen.is_paused, gen.next_scheduled_time))
            gen.pause()
            seen.append((gen.is_paused, gen.next_scheduled_time))

        gen = model.schedule_recurring(cb, Schedule(interval=1.0))
        model.run_for(3.0)

        assert seen == [(False, None), (True, None)]  # fired once, then paused
        assert gen.is_active
        assert gen.is_paused


class TestEventGeneratorExceptions:
    """Exceptions reach the caller; only one computing the next time ends the generator."""

    def test_failed_call_counts_toward_count(self, setup):
        """A call that raised still counts, so count stays an upper bound on calls."""
        model, fn = setup
        fn.side_effect = [RuntimeError("boom"), None, None]
        gen = model.schedule_recurring(fn, Schedule(interval=1.0, start=1.0, count=2))

        with pytest.raises(RuntimeError):
            model.run_until(1)
        model.run_until(10)

        assert fn.call_count == 2
        assert not gen.is_active

    def test_cancel_in_failing_callback_is_respected(self, setup):
        """A callback that cancels its generator and then raises is not rescheduled."""
        model, fn = setup
        gen = None

        def cancel_and_raise():
            gen.cancel()
            raise RuntimeError("boom")

        fn.side_effect = cancel_and_raise
        gen = model.schedule_recurring(fn, Schedule(interval=1.0, start=1.0))

        with pytest.raises(RuntimeError):
            model.run_for(5)
        model.run_for(5)

        assert fn.call_count == 1
        assert gen.state is EventGeneratorState.CANCELED
        assert gen.next_scheduled_time is None

    def test_interval_error_fails_the_generator(self, setup):
        model, fn = setup
        intervals = iter([1.0])

        def interval(m):
            return next(intervals)  # StopIteration on the second call

        gen = model.schedule_recurring(fn, Schedule(interval=interval))

        with pytest.raises(StopIteration):
            model.run_for(5)

        assert gen.state is EventGeneratorState.FAILED
        assert not gen.is_active
        assert gen.next_scheduled_time is None
        assert gen not in model._event_generators
        model.run_for(5)
        assert fn.call_count == 1

    @pytest.mark.parametrize("bad", [-0.5, 0, math.inf, math.nan])
    def test_invalid_callable_interval_fails_the_generator(self, setup, bad):
        model, fn = setup
        gen = model.schedule_recurring(fn, Schedule(start=1, interval=lambda m: bad))

        with pytest.raises(ValueError, match="finite and > 0"):
            model.run_for(3)
        assert gen.state is EventGeneratorState.FAILED

    def test_interval_error_at_start(self, setup):
        model, fn = setup

        def interval(m):
            raise KeyError("no interval")

        with pytest.raises(KeyError):
            model.schedule_recurring(fn, Schedule(interval=interval))
        assert not any(g.function() is fn for g in model._event_generators)

    def test_both_raise(self, setup):
        """The interval's error surfaces, with the callback's as its context."""
        model, fn = setup
        fn.side_effect = RuntimeError("callback")
        calls = iter([1.0])

        def interval(m):
            if next(calls, None) is None:
                raise KeyError("interval")
            return 1.0

        gen = model.schedule_recurring(fn, Schedule(interval=interval))
        with pytest.raises(KeyError) as excinfo:
            model.run_for(5)

        assert isinstance(excinfo.value.__context__, RuntimeError)
        assert gen.state is EventGeneratorState.FAILED


class TestEventGeneratorPauseResume:
    def test_full_lifecycle(self, setup):
        model, fn = setup
        gen = model.schedule_recurring(fn, Schedule(interval=1.0))

        model.run_for(1.0)
        assert fn.call_count == 1

        assert gen.pause() is True
        model.run_for(10.0)
        assert fn.call_count == 1

        assert gen.resume() is True
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
        gen = model.schedule_recurring(fn, Schedule(interval=1.0))
        assert gen.pause() is True
        assert gen.pause() is False
        assert gen.pause() is False
        model.run_for(5.0)
        assert fn.call_count == 0

    def test_resume_idempotent(self, setup):
        """Calling resume while running should do nothing."""
        model, fn = setup
        gen = model.schedule_recurring(fn, Schedule(interval=1.0))
        assert gen.resume() is False
        assert gen.resume() is False
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

        gen = model.schedule_recurring(fn, Schedule(interval=1.0))
        model.run_for(5.0)
        assert call_count["n"] == 1

    def test_pause_from_within_callback(self, setup):
        """Pausing from the callback must not reschedule until resume."""
        model, _ = setup
        calls = []
        gen = None

        def cb():
            calls.append(model.time)
            if len(calls) == 1:
                gen.pause()

        gen = model.schedule_recurring(cb, Schedule(interval=1.0, start=1.0))
        model.run_for(5.0)
        assert calls == [1.0]
        assert gen.next_scheduled_time is None

        gen.resume()
        model.run_for(2.0)
        assert calls == [1.0, 6.0, 7.0]

    def test_pause_resume_from_within_callback(self, setup):
        """Pausing and resuming from the callback must not double-schedule."""
        model, _ = setup
        calls = []
        gen = None

        def cb():
            calls.append(model.time)
            if len(calls) == 1:
                gen.pause()
                gen.resume()

        gen = model.schedule_recurring(cb, Schedule(interval=1.0))
        model.run_for(4.0)

        # resume() schedules a new event too, so this must not double up either.
        assert calls == [1.0, 2.0, 3.0, 4.0]

    def test_pause_resume_in_the_last_callback_respects_count(self, setup):
        """The count is used up before the call, so resume() finishes instead."""
        model, _ = setup
        calls = []
        resumed = []
        gen = None

        def cb():
            calls.append(model.time)
            gen.pause()
            resumed.append(gen.resume())

        gen = model.schedule_recurring(cb, Schedule(interval=1.0, count=1))
        model.run_for(5.0)

        assert calls == [1.0]
        assert resumed == [False]
        assert gen.execution_count == 1
        assert gen.state is EventGeneratorState.FINISHED

    def test_pause_resume_then_raise_in_the_last_callback(self, setup):
        model, _ = setup
        calls = []
        gen = None

        def cb():
            calls.append(model.time)
            gen.pause()
            gen.resume()
            raise RuntimeError("after resume")

        gen = model.schedule_recurring(cb, Schedule(interval=1.0, count=1))
        with pytest.raises(RuntimeError):
            model.run_for(5.0)
        model.run_for(5.0)

        assert calls == [1.0]
        assert gen.state is EventGeneratorState.FINISHED

    def test_pause_in_the_last_callback_finishes(self, setup):
        """A spent generator isn't left PAUSED."""
        model, _ = setup
        gen = None

        def cb():
            gen.pause()

        gen = model.schedule_recurring(cb, Schedule(interval=1.0, count=1))
        model.run_for(5.0)

        assert gen.state is EventGeneratorState.FINISHED
        assert not gen.is_active

    def test_resume_passing_end_finishes(self, setup):
        model, fn = setup
        gen = model.schedule_recurring(fn, Schedule(interval=2.0, end=3.0))
        gen.pause()
        model.run_for(2.0)

        assert gen.resume() is False
        assert gen.state is EventGeneratorState.FINISHED

    def test_paused_generators_resume_as_a_group(self, setup):
        """Keep the ones that actually paused, since some may already be final."""
        model, fn = setup
        running = model.schedule_recurring(fn, Schedule(interval=1.0))
        ended = model.schedule_recurring(fn, Schedule(interval=1.0))
        ended.cancel()

        paused = [gen for gen in (running, ended) if gen.pause()]
        assert paused == [running]
        assert all(gen.resume() for gen in paused)

    def test_next_scheduled_time_updates(self, setup):
        """next_scheduled_time should reflect pause/resume state."""
        model, fn = setup
        gen = model.schedule_recurring(fn, Schedule(interval=2.0))
        assert gen.next_scheduled_time is not None

        gen.pause()
        assert gen.next_scheduled_time is None

        gen.resume()
        assert gen.next_scheduled_time is not None

    def test_resume_schedules_from_current_time(self, setup):
        """Resume should schedule next execution relative to current time."""
        model, fn = setup
        gen = model.schedule_recurring(fn, Schedule(interval=2.0))

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
            "_state": EventGeneratorState.PENDING,
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

        gen = model.schedule_recurring(temp_func, Schedule(interval=1.0))

        model.run_for(1.0)
        assert call_count[0] == 1
        assert gen.is_active

        del temp_func
        gc.collect()

        model.run_for(1.0)
        assert gen.state is EventGeneratorState.CANCELED
        assert call_count[0] == 1
