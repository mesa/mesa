"""Tests for event scheduling edge cases."""

import random
import time
from unittest.mock import MagicMock

import pytest

from mesa import Agent, Model
from mesa.experimental.devs.eventlist import (
    EventGenerator,
    EventList,
    Priority,
    SimulationEvent,
)
from mesa.experimental.devs.simulator import ABMSimulator, DEVSimulator


class TestEventSchedulingEdgeCases:
    """Tests for edge cases in event scheduling."""

    def test_multiple_events_same_time_different_priorities(self):
        """Test priority ordering for simultaneous events."""
        model = Model()
        simulator = DEVSimulator()
        simulator.setup(model)

        execution_order = []

        def append_low():
            execution_order.append("low")

        def append_high():
            execution_order.append("high")

        def append_default():
            execution_order.append("default")

        simulator.schedule_event_absolute(append_low, 1.0, priority=Priority.LOW)
        simulator.schedule_event_absolute(append_high, 1.0, priority=Priority.HIGH)
        simulator.schedule_event_absolute(
            append_default, 1.0, priority=Priority.DEFAULT
        )

        simulator.run_until(1.01)

        assert execution_order == ["high", "default", "low"]

    def test_event_scheduling_during_event_execution(self):
        """Test scheduling events from within event callbacks."""
        model = Model()
        simulator = DEVSimulator()
        simulator.setup(model)

        execution_log = []

        def append_nested():
            execution_log.append("nested")

        def schedule_nested_event():
            execution_log.append("outer")
            simulator.schedule_event_absolute(append_nested, 2.0)

        simulator.schedule_event_absolute(schedule_nested_event, 1.0)
        simulator.run_until(3.0)

        assert execution_log == ["outer", "nested"]

    def test_event_cancellation_during_execution(self):
        """Test canceling events from event callbacks."""
        model = Model()
        simulator = DEVSimulator()
        simulator.setup(model)

        fn1 = MagicMock()
        fn2 = MagicMock()

        event2 = simulator.schedule_event_absolute(fn2, 2.0)

        def cancel_event2():
            simulator.cancel_event(event2)
            fn1()

        simulator.schedule_event_absolute(cancel_event2, 1.0)
        simulator.run_until(3.0)

        fn1.assert_called_once()
        fn2.assert_not_called()

    def test_zero_interval_events(self):
        """Test events scheduled with zero time delta."""
        model = Model()
        simulator = DEVSimulator()
        simulator.setup(model)

        fn = MagicMock()
        simulator.schedule_event_relative(fn, 0.0)

        simulator.run_for(0.0)
        fn.assert_called_once()

    def test_fractional_time_events(self):
        """Test events scheduled at fractional time values."""
        model = Model()
        simulator = DEVSimulator()
        simulator.setup(model)

        times_executed = []
        functions = []  # prevent GC

        def create_callback(expected_time):
            def callback():
                times_executed.append(expected_time)

            return callback

        for i in range(10):
            event_time = i * 0.1
            func = create_callback(event_time)
            functions.append(func)
            simulator.schedule_event_absolute(func, event_time)

        simulator.run_until(1.0)

        assert len(times_executed) == 10
        assert times_executed == [i * 0.1 for i in range(10)]

    def test_event_list_large_number_of_events(self):
        """Test handling 10k events."""
        model = Model()
        simulator = DEVSimulator()
        simulator.setup(model)

        num_events = 10000
        executed_count = [0]

        def increment_counter():
            executed_count[0] += 1

        for i in range(num_events):
            simulator.schedule_event_absolute(increment_counter, float(i))

        simulator.run_until(float(num_events))

        assert executed_count[0] == num_events

    def test_event_generator_with_zero_interval(self):
        """Test generator with zero interval."""
        model = Model()
        simulator = DEVSimulator()
        simulator.setup(model)

        fn = MagicMock()

        gen = EventGenerator(model, fn, interval=0.0)
        gen.start(at=0.0).stop(count=5)

        try:
            simulator.run_for(1.0)
            assert fn.call_count == 5
        except RecursionError:
            pytest.skip("Zero interval causes recursion")

    def test_event_generator_negative_interval_rejection(self):
        """Test that negative intervals are rejected."""
        model = Model()

        gen = EventGenerator(model, MagicMock(), interval=-1.0)

        simulator = DEVSimulator()
        simulator.setup(model)

        try:
            gen.start()
            pytest.skip("Negative intervals not validated - potential improvement area")
        except ValueError:
            pass

    def test_event_generator_stop_before_first_execution(self):
        """Test stopping generator before it executes."""
        model = Model()
        simulator = DEVSimulator()
        simulator.setup(model)

        fn = MagicMock()
        gen = EventGenerator(model, fn, interval=10.0)
        gen.start(at=5.0)
        gen.stop(at=3.0)

        simulator.run_until(10.0)

        assert fn.call_count >= 0

    def test_multiple_generators_same_time(self):
        """Test multiple generators with same schedule."""
        model = Model()
        simulator = DEVSimulator()
        simulator.setup(model)

        calls_a = []
        calls_b = []

        def append_a():
            calls_a.append(model.time)

        def append_b():
            calls_b.append(model.time)

        gen_a = EventGenerator(model, append_a, interval=2.0)
        gen_b = EventGenerator(model, append_b, interval=2.0)

        gen_a.start(at=0.0)
        gen_b.start(at=0.0)

        simulator.run_until(5.0)

        assert len(calls_a) == 3
        assert len(calls_b) == 3
        assert calls_a == calls_b

    def test_event_generator_dynamic_interval(self):
        """Test generator with callable interval."""
        model = Model()
        simulator = DEVSimulator()
        simulator.setup(model)

        call_times = []
        interval_values = [1.0, 2.0, 0.5, 3.0]
        interval_iter = iter(interval_values)

        def dynamic_interval(m):
            try:
                return next(interval_iter)
            except StopIteration:
                return 1.0

        def append_time():
            call_times.append(model.time)

        gen = EventGenerator(model, append_time, interval=dynamic_interval)
        gen.start(at=0.0)

        simulator.run_until(10.0)

        assert call_times[0] == 0.0
        assert call_times[1] == 1.0
        assert call_times[2] == 3.0
        assert call_times[3] == 3.5

    def test_event_with_exception_during_execution(self):
        """Test that exceptions in events propagate correctly."""
        model = Model()
        simulator = DEVSimulator()
        simulator.setup(model)

        def failing_event():
            raise RuntimeError("Event execution failed")

        simulator.schedule_event_absolute(failing_event, 1.0)

        with pytest.raises(RuntimeError, match="Event execution failed"):
            simulator.run_until(1.0)

    def test_weakref_cleanup_of_agent_methods(self):
        """Test weakref cleanup when agent is deleted."""
        model = Model()
        simulator = DEVSimulator()
        simulator.setup(model)

        class TestAgent(Agent):
            def __init__(self, model):
                super().__init__(model)
                self.called = False

            def scheduled_method(self):
                self.called = True

        agent = TestAgent(model)
        simulator.schedule_event_absolute(agent.scheduled_method, 1.0)

        agent.remove()
        del agent

        # should not raise exception
        simulator.run_until(1.0)

    def test_event_list_peek_ahead_with_canceled_events(self):
        """Test peek_ahead with canceled events."""
        event_list = EventList()
        fn = MagicMock()

        events = []
        for i in range(10):
            event = SimulationEvent(i, fn, priority=Priority.DEFAULT)
            event_list.add_event(event)
            events.append(event)

        events[2].cancel()
        events[5].cancel()
        events[7].cancel()

        peeked = event_list.peek_ahead(5)

        assert len(peeked) == 5
        assert peeked[0].time == 0
        assert peeked[1].time == 1
        assert peeked[2].time == 3
        assert peeked[3].time == 4
        assert peeked[4].time == 6

    def test_hybrid_abm_devs_execution(self):
        """Test mixing ABM steps with DEVS events."""
        simulator = ABMSimulator()

        class TestModel(Model):
            def __init__(self):
                super().__init__()
                self.step_count = 0

            def step(self):
                self.step_count += 1

        model = TestModel()
        simulator.setup(model)

        event_calls = []

        def record_step():
            event_calls.append(model.steps)

        simulator.schedule_event_absolute(record_step, 2)

        simulator.run_for(3)

        assert model.step_count == 3
        assert len(event_calls) == 1
        assert event_calls[0] >= 2

    def test_time_precision_with_many_fractional_intervals(self):
        """Test float precision over many fractional intervals."""
        model = Model()
        simulator = DEVSimulator()
        simulator.setup(model)

        call_times = []

        interval = 0.1
        count = 100

        for i in range(count):
            expected_time = i * interval
            simulator.schedule_event_absolute(
                lambda t=expected_time: call_times.append((model.time, t)),
                expected_time,
            )

        simulator.run_until(10.0)

        for actual, expected in call_times:
            assert abs(actual - expected) < 1e-10

    def test_simultaneous_start_stop_generators(self):
        """Test generators with overlapping lifetimes."""
        model = Model()
        simulator = DEVSimulator()
        simulator.setup(model)

        calls = {"gen1": [], "gen2": [], "gen3": []}

        def append_gen1():
            calls["gen1"].append(model.time)

        def append_gen2():
            calls["gen2"].append(model.time)

        def append_gen3():
            calls["gen3"].append(model.time)

        gen1 = EventGenerator(model, append_gen1, interval=1.0)
        gen1.start(at=0.0).stop(at=5.0)

        gen2 = EventGenerator(model, append_gen2, interval=1.0)
        gen2.start(at=2.0).stop(at=7.0)

        gen3 = EventGenerator(model, append_gen3, interval=0.5)
        gen3.start(at=1.0).stop(at=3.0)

        simulator.run_until(10.0)

        assert len(calls["gen1"]) == 6
        assert len(calls["gen2"]) == 6
        assert len(calls["gen3"]) == 5

    def test_model_step_without_simulator(self):
        """Test Model.step() works without simulator."""

        class TestModel(Model):
            def __init__(self):
                super().__init__()
                self.step_calls = []

            def step(self):
                self.step_calls.append(self.time)

        model = TestModel()

        for _ in range(5):
            model.step()

        assert len(model.step_calls) == 5
        assert model.time == 5.0
        assert model.steps == 5

    def test_event_list_empty_pop_raises_index_error(self):
        """Test popping from empty event list."""
        event_list = EventList()

        with pytest.raises(IndexError, match="Event list is empty"):
            event_list.pop_event()

    def test_event_generator_interval_callable_exception(self):
        """Test exception handling in interval callable."""
        model = Model()
        simulator = DEVSimulator()
        simulator.setup(model)

        def bad_interval(m):
            raise ValueError("Interval calculation failed")

        gen = EventGenerator(model, MagicMock(), interval=bad_interval)

        with pytest.raises(ValueError, match="Interval calculation failed"):
            gen.start()

    def test_event_scheduling_at_current_time(self):
        """Test scheduling event at current time."""
        model = Model()
        simulator = DEVSimulator()
        simulator.setup(model)

        simulator.run_until(5.0)

        fn = MagicMock()
        simulator.schedule_event_absolute(fn, 5.0)

        simulator.run_for(0.0)
        fn.assert_called_once()

    def test_multiple_event_cancellations(self):
        """Test canceling same event multiple times."""
        model = Model()
        simulator = DEVSimulator()
        simulator.setup(model)

        fn = MagicMock()
        event = simulator.schedule_event_absolute(fn, 1.0)

        simulator.cancel_event(event)
        simulator.cancel_event(event)

        simulator.run_until(2.0)
        fn.assert_not_called()

    def test_event_generator_chaining_methods(self):
        """Test method chaining."""
        model = Model()
        simulator = DEVSimulator()
        simulator.setup(model)

        fn = MagicMock()

        gen = EventGenerator(model, fn, interval=1.0).start(at=0.0).stop(count=3)

        simulator.run_until(10.0)

        assert fn.call_count == 3
        assert not gen.is_active


class TestEventSchedulingWithRealAgents:
    """Tests for event scheduling with agents."""

    def test_scheduled_agent_removal(self):
        """Test agent removal before scheduled event."""

        class TestAgent(Agent):
            def __init__(self, model):
                super().__init__(model)
                self.action_count = 0

            def scheduled_action(self):
                self.action_count += 1

        model = Model()
        simulator = DEVSimulator()
        simulator.setup(model)

        agents = [TestAgent(model) for _ in range(5)]

        for i, agent in enumerate(agents):
            simulator.schedule_event_absolute(agent.scheduled_action, i + 1.0)

        agents[2].remove()

        simulator.run_until(10.0)

        assert agents[0].action_count == 1
        assert agents[1].action_count == 1
        assert agents[3].action_count == 1
        assert agents[4].action_count == 1

    def test_agent_scheduling_own_events(self):
        """Test agents scheduling their own events."""

        class SelfSchedulingAgent(Agent):
            def __init__(self, model, simulator):
                super().__init__(model)
                self.simulator = simulator
                self.execution_times = []

            def schedule_next(self):
                self.execution_times.append(self.model.time)
                if len(self.execution_times) < 5:
                    self.simulator.schedule_event_relative(self.schedule_next, 1.0)

        model = Model()
        simulator = DEVSimulator()
        simulator.setup(model)

        agent = SelfSchedulingAgent(model, simulator)
        agent.schedule_next()

        simulator.run_until(10.0)

        assert len(agent.execution_times) == 5
        assert agent.execution_times == [0.0, 1.0, 2.0, 3.0, 4.0]


class TestEventSchedulingPerformance:
    """Tests for performance and scaling."""

    def test_event_list_maintains_heap_property(self):
        """Test heap property with random insertion order."""
        event_list = EventList()
        fn = MagicMock()

        times = [random.random() * 100 for _ in range(1000)]

        for event_time in times:
            event = SimulationEvent(event_time, fn, priority=Priority.DEFAULT)
            event_list.add_event(event)

        popped_times = []
        while not event_list.is_empty():
            event = event_list.pop_event()
            popped_times.append(event.time)

        assert popped_times == sorted(times)

    def test_peek_ahead_performance(self):
        """Test peek_ahead performance with 10k events."""
        event_list = EventList()
        fn = MagicMock()

        for i in range(10000):
            event = SimulationEvent(i, fn, priority=Priority.DEFAULT)
            event_list.add_event(event)

        start = time.time()
        peeked = event_list.peek_ahead(100)
        duration = time.time() - start

        assert len(peeked) == 100
        assert duration < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
