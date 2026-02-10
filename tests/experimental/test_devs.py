"""Tests for experimental Simulator classes."""

from functools import partial
from unittest.mock import MagicMock, Mock

import pytest

from mesa import Model
from mesa.experimental.devs.eventlist import (
    EventGenerator,
    EventList,
    Priority,
    Schedule,
    SimulationEvent,
)
from mesa.experimental.devs.simulator import ABMSimulator, DEVSimulator


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
        simulator.event_list.add_event(SimulationEvent(1.0, Mock(), Priority.DEFAULT))


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
    assert model.steps == 3
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
    """Tests for SimulationEvent class."""
    some_test_function = MagicMock()

    time = 10
    event = SimulationEvent(
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
        SimulationEvent(
            time, None, priority=Priority.DEFAULT, function_args=[], function_kwargs={}
        )

    # check calling with arguments
    some_test_function = MagicMock()
    event = SimulationEvent(
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

    event = SimulationEvent(time, some_test_function, priority=Priority.DEFAULT)
    del some_test_function
    event.execute()

    # cancel
    some_test_function = MagicMock()
    event = SimulationEvent(
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
    event1 = SimulationEvent(
        10,
        some_test_function,
        priority=Priority.DEFAULT,
        function_args=[],
        function_kwargs={},
    )
    event2 = SimulationEvent(
        10,
        some_test_function,
        priority=Priority.DEFAULT,
        function_args=[],
        function_kwargs={},
    )
    assert event1 < event2  # based on just unique_id as tiebraker

    event1 = SimulationEvent(
        11,
        some_test_function,
        priority=Priority.DEFAULT,
        function_args=[],
        function_kwargs={},
    )
    event2 = SimulationEvent(
        10,
        some_test_function,
        priority=Priority.DEFAULT,
        function_args=[],
        function_kwargs={},
    )
    assert event1 > event2

    event1 = SimulationEvent(
        10,
        some_test_function,
        priority=Priority.DEFAULT,
        function_args=[],
        function_kwargs={},
    )
    event2 = SimulationEvent(
        10,
        some_test_function,
        priority=Priority.HIGH,
        function_args=[],
        function_kwargs={},
    )
    assert event1 > event2


def test_simulation_event_pickle():
    """Test pickling and unpickling of SimulationEvent."""

    # Test with regular function
    def test_fn():
        return "test"

    event = SimulationEvent(
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

    new_event = SimulationEvent.__new__(SimulationEvent)
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

    new_event = SimulationEvent.__new__(SimulationEvent)
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
    event = SimulationEvent(
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
    assert len(event_list) == 1
    assert event.CANCELED

    # peak ahead
    event_list = EventList()
    for i in range(10):
        event = SimulationEvent(
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
        event = SimulationEvent(
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
        event = SimulationEvent(
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
    event = SimulationEvent(
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


@pytest.fixture
def setup():
    """Create a model with simulator and mock function."""
    model = Model()
    simulator = DEVSimulator()
    simulator.setup(model)
    return model, simulator, MagicMock()


class TestSchedule:
    """Tests for Schedule dataclass."""

    def test_defaults(self):
        """Test default values."""
        s = Schedule()
        assert s.interval == 1.0
        assert s.start is None
        assert s.end is None
        assert s.count is None

    def test_custom_values(self):
        """Test custom values."""
        s = Schedule(interval=7, start=100, end=500, count=10)
        assert s.interval == 7
        assert s.start == 100
        assert s.end == 500
        assert s.count == 10

    def test_callable_interval(self):
        """Test callable interval."""
        s = Schedule(interval=lambda m: m.time * 2)
        assert callable(s.interval)


class TestEventGenerator:
    """Tests for EventGenerator."""

    def test_init(self, setup):
        """Test initialization with Schedule."""
        model, _sim, fn = setup
        schedule = Schedule(interval=5.0, start=10, end=100, count=5)
        gen = EventGenerator(model, fn, schedule)

        assert gen.model is model
        assert gen.schedule is schedule
        assert gen.priority == Priority.DEFAULT
        assert not gen.is_active
        assert gen.execution_count == 0

    def test_init_with_priority(self, setup):
        """Test initialization with custom priority."""
        model, _sim, fn = setup
        gen = EventGenerator(model, fn, Schedule(), priority=Priority.HIGH)
        assert gen.priority == Priority.HIGH

    def test_start_with_schedule_start(self, setup):
        """Test start uses schedule.start time."""
        model, sim, fn = setup
        gen = EventGenerator(model, fn, Schedule(interval=1.0, start=5.0))

        assert gen.start() is gen
        assert gen.is_active

        sim.run_for(4.9)
        fn.assert_not_called()
        sim.run_for(0.1)
        fn.assert_called_once()

    def test_start_without_schedule_start(self, setup):
        """Test start defaults to now + interval when schedule.start is None."""
        model, sim, fn = setup
        gen = EventGenerator(model, fn, Schedule(interval=2.0))
        gen.start()

        sim.run_for(1.9)
        fn.assert_not_called()
        sim.run_for(0.1)
        fn.assert_called_once()

    def test_start_when_active_is_noop(self, setup):
        """Test that starting when active does nothing."""
        model, sim, fn = setup
        gen = EventGenerator(model, fn, Schedule(interval=1.0))
        gen.start()
        event_count = len(sim.event_list)

        gen.start()  # No-op
        assert len(sim.event_list) == event_count

    def test_stop(self, setup):
        """Test immediate stop."""
        model, sim, fn = setup
        gen = EventGenerator(model, fn, Schedule(interval=1.0))

        assert gen.start().stop() is gen
        assert not gen.is_active

        sim.run_for(5.0)
        fn.assert_not_called()

    def test_schedule_end(self, setup):
        """Test schedule.end stops execution."""
        model, sim, fn = setup
        gen = EventGenerator(model, fn, Schedule(interval=1.0, start=0.0, end=2.5))
        gen.start()

        sim.run_for(5.0)
        assert fn.call_count == 3  # t=0, 1, 2
        assert not gen.is_active

    def test_schedule_count(self, setup):
        """Test schedule.count limits executions."""
        model, sim, fn = setup
        gen = EventGenerator(model, fn, Schedule(interval=1.0, start=0.0, count=3))
        gen.start()

        sim.run_for(10.0)
        assert fn.call_count == 3
        assert gen.execution_count == 3
        assert not gen.is_active

    def test_schedule_end_and_count(self, setup):
        """Test count reached before end."""
        model, sim, fn = setup
        gen = EventGenerator(
            model, fn, Schedule(interval=1.0, start=0.0, end=100, count=2)
        )
        gen.start()

        sim.run_for(10.0)
        assert fn.call_count == 2

    def test_recurring_execution(self, setup):
        """Test recurring execution and count tracking."""
        model, sim, fn = setup
        gen = EventGenerator(model, fn, Schedule(interval=2.0, start=0.0))
        gen.start()

        sim.run_for(7.0)
        assert fn.call_count == 4  # t=0, 2, 4, 6
        assert gen.execution_count == 4

    def test_callable_interval(self, setup):
        """Test callable interval evaluated each time."""
        model, sim, fn = setup
        intervals = iter([1.0, 2.0, 1.0, 1.0])
        schedule = Schedule(interval=lambda m: next(intervals), start=0.0)
        gen = EventGenerator(model, fn, schedule)
        gen.start()

        sim.run_for(4.5)
        assert fn.call_count == 4  # t=0, 1, 3, 4

    def test_functools_partial(self, setup):
        """Test using functools.partial for arguments."""
        model, sim, fn = setup
        gen = EventGenerator(
            model, partial(fn, "a", k="v"), Schedule(interval=1.0, start=0.0)
        )
        gen.start()

        sim.run_for(0.5)
        fn.assert_called_once_with("a", k="v")

    def test_priority_ordering(self, setup):
        """Test priority affects execution order at same time."""
        model, sim, _ = setup
        order = []

        low = EventGenerator(
            model,
            lambda: order.append("L"),
            Schedule(interval=1.0, start=1.0),
            Priority.LOW,
        )
        high = EventGenerator(
            model,
            lambda: order.append("H"),
            Schedule(interval=1.0, start=1.0),
            Priority.HIGH,
        )

        low.start()
        high.start()

        sim.run_for(1.5)
        assert order == ["H", "L"]
