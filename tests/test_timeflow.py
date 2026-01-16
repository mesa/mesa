"""Tests for unified time and event scheduling API."""

import pytest

from mesa import Agent, Model
from mesa.experimental.devs.eventlist import Priority
from mesa.timeflow import RunControl, Scheduler, scheduled


class TestScheduler:
    """Test the Scheduler class."""

    def test_schedule_at_absolute_time(self):
        """Test scheduling an event at an absolute time."""
        model = Model()
        scheduler = Scheduler(model)

        results = []

        def callback():
            results.append(model.time)

        event = scheduler.schedule_at(callback, time=10.0)

        assert event.time == 10.0
        assert not event.CANCELED
        assert len(scheduler.event_list) == 1

    def test_schedule_at_past_raises_error(self):
        """Test that scheduling in the past raises ValueError."""
        model = Model()
        model.time = 10.0
        scheduler = Scheduler(model)

        def callback():
            pass

        with pytest.raises(ValueError, match="Cannot schedule event in the past"):
            scheduler.schedule_at(callback, time=5.0)

    def test_schedule_after_relative_time(self):
        """Test scheduling an event after a delay."""
        model = Model()
        model.time = 5.0
        scheduler = Scheduler(model)

        results = []

        def callback():
            results.append(model.time)

        event = scheduler.schedule_after(callback, delay=3.0)

        assert event.time == 8.0
        assert len(scheduler.event_list) == 1

    def test_schedule_with_priority(self):
        """Test that events with different priorities execute in correct order."""
        model = Model()
        scheduler = Scheduler(model)

        execution_order = []

        def low_priority():
            execution_order.append("low")

        def high_priority():
            execution_order.append("high")

        def default_priority():
            execution_order.append("default")

        # Schedule all at same time with different priorities
        scheduler.schedule_at(low_priority, time=10.0, priority=Priority.LOW)
        scheduler.schedule_at(high_priority, time=10.0, priority=Priority.HIGH)
        scheduler.schedule_at(default_priority, time=10.0, priority=Priority.DEFAULT)

        # Execute events
        while not scheduler.event_list.is_empty():
            event = scheduler.event_list.pop_event()
            model.time = event.time
            event.execute()

        assert execution_order == ["high", "default", "low"]

    def test_schedule_with_args_and_kwargs(self):
        """Test scheduling events with arguments."""
        model = Model()
        scheduler = Scheduler(model)

        results = []

        def callback(x, y, z=None):
            results.append((x, y, z))

        scheduler.schedule_at(callback, time=5.0, args=[1, 2], kwargs={"z": 3})

        event = scheduler.event_list.pop_event()
        event.execute()

        assert results == [(1, 2, 3)]

    def test_cancel_event(self):
        """Test canceling a scheduled event."""
        model = Model()
        scheduler = Scheduler(model)

        executed = []

        def callback():
            executed.append(True)

        event = scheduler.schedule_at(callback, time=10.0)
        assert len(scheduler.event_list) == 1

        scheduler.cancel(event)

        assert event.CANCELED
        # Event is still physically in the list (lazy deletion)
        assert len(scheduler.event_list) == 1

        # But trying to pop will raise IndexError since it skips canceled events
        with pytest.raises(IndexError, match="Event list is empty"):
            scheduler.event_list.pop_event()

        # Verify callback was never executed
        assert executed == []

    def test_clear_events(self):
        """Test clearing all scheduled events."""
        model = Model()
        scheduler = Scheduler(model)

        def callback():
            pass

        scheduler.schedule_at(callback, time=10.0)
        scheduler.schedule_at(callback, time=20.0)
        scheduler.schedule_at(callback, time=30.0)

        assert len(scheduler.event_list) == 3

        scheduler.clear()

        assert scheduler.event_list.is_empty()


class TestRunControl:
    """Test the RunControl class."""

    def test_run_until(self):
        """Test running until a specific time."""
        model = Model()
        scheduler = Scheduler(model)
        run_control = RunControl(model, scheduler)

        execution_times = []

        def callback():
            execution_times.append(model.time)

        scheduler.schedule_at(callback, time=5.0)
        scheduler.schedule_at(callback, time=10.0)
        scheduler.schedule_at(callback, time=15.0)

        run_control.run_until(12.0)

        assert execution_times == [5.0, 10.0]
        assert model.time == 12.0
        # One event should remain scheduled
        assert len(scheduler.event_list) == 1

    def test_run_for(self):
        """Test running for a specific duration."""
        model = Model()
        model.time = 5.0
        scheduler = Scheduler(model)
        run_control = RunControl(model, scheduler)

        execution_times = []

        def callback():
            execution_times.append(model.time)

        scheduler.schedule_at(callback, time=8.0)
        scheduler.schedule_at(callback, time=12.0)

        run_control.run_for(5.0)

        assert execution_times == [8.0]
        assert model.time == 10.0

    def test_run_while(self):
        """Test running while a condition is true."""
        model = Model()
        scheduler = Scheduler(model)
        run_control = RunControl(model, scheduler)

        execution_count = [0]

        def callback():
            execution_count[0] += 1

        def condition(m):
            return execution_count[0] < 3

        scheduler.schedule_at(callback, time=1.0)
        scheduler.schedule_at(callback, time=2.0)
        scheduler.schedule_at(callback, time=3.0)
        scheduler.schedule_at(callback, time=4.0)

        run_control.run_while(condition)

        assert execution_count[0] == 3

    def test_run_next_event(self):
        """Test executing events one at a time."""
        model = Model()
        scheduler = Scheduler(model)
        run_control = RunControl(model, scheduler)

        execution_times = []

        def callback():
            execution_times.append(model.time)

        scheduler.schedule_at(callback, time=5.0)
        scheduler.schedule_at(callback, time=10.0)

        result = run_control.run_next_event()
        assert result is True
        assert execution_times == [5.0]
        assert model.time == 5.0

        result = run_control.run_next_event()
        assert result is True
        assert execution_times == [5.0, 10.0]
        assert model.time == 10.0

        result = run_control.run_next_event()
        assert result is False  # No more events

    def test_run_until_advances_time_with_no_events(self):
        """Test that run_until advances time even with no events."""
        model = Model()
        scheduler = Scheduler(model)
        run_control = RunControl(model, scheduler)

        run_control.run_until(50.0)

        assert model.time == 50.0


class TestScheduledDecorator:
    """Test the @scheduled decorator."""

    def test_scheduled_default_interval(self):
        """Test @scheduled with default interval of 1.0."""

        class TestModel(Model):
            def __init__(self):
                super().__init__()
                self.step_count = 0

            @scheduled
            def step(self):
                self.step_count += 1

        model = TestModel()
        model.run_until(5.0)

        # Should execute at t=1, 2, 3, 4, 5
        assert model.step_count == 5

    def test_scheduled_custom_interval(self):
        """Test @scheduled with custom interval."""

        class TestModel(Model):
            def __init__(self):
                super().__init__()
                self.update_count = 0

            @scheduled(interval=2.5)
            def periodic_update(self):
                self.update_count += 1

        model = TestModel()
        model.run_until(10.0)

        # Should execute at t=2.5, 5.0, 7.5, 10.0
        assert model.update_count == 4

    def test_scheduled_with_priority(self):
        """Test @scheduled respects priority."""

        class TestModel(Model):
            def __init__(self):
                super().__init__()
                self.execution_order = []

            @scheduled(priority=Priority.LOW)
            def low_priority_step(self):
                self.execution_order.append("low")

            @scheduled(priority=Priority.HIGH)
            def high_priority_step(self):
                self.execution_order.append("high")

        model = TestModel()
        model.run_until(1.0)

        # High priority should execute first
        assert model.execution_order == ["high", "low"]

    def test_multiple_scheduled_methods(self):
        """Test model with multiple @scheduled methods at different intervals."""

        class TestModel(Model):
            def __init__(self):
                super().__init__()
                self.hourly_count = 0
                self.daily_count = 0

            @scheduled(interval=1.0)
            def hourly(self):
                self.hourly_count += 1

            @scheduled(interval=24.0)
            def daily(self):
                self.daily_count += 1

        model = TestModel()
        model.run_until(48.0)

        assert model.hourly_count == 48
        assert model.daily_count == 2

    def test_scheduled_not_triggered_on_private_methods(self):
        """Test that @scheduled on private methods doesn't break."""

        class TestModel(Model):
            def __init__(self):
                super().__init__()
                self.count = 0

            @scheduled
            def _private_method(self):
                self.count += 1

        model = TestModel()
        model.run_until(5.0)

        # Private methods should be skipped
        assert model.count == 0


class TestModelIntegration:
    """Test integration of timeflow with Model class."""

    def test_model_schedule_at(self):
        """Test model.schedule_at method."""
        model = Model()

        results = []

        def callback():
            results.append(model.time)

        model.schedule_at(callback, time=15.0)
        model.run_until(20.0)

        assert results == [15.0]
        assert model.time == 20.0

    def test_model_schedule_after(self):
        """Test model.schedule_after method."""
        model = Model()
        model.time = 10.0

        results = []

        def callback():
            results.append(model.time)

        model.schedule_after(callback, delay=5.0)
        model.run_until(20.0)

        assert results == [15.0]

    def test_model_cancel_event(self):
        """Test model.cancel_event method."""
        model = Model()

        executed = []

        def callback():
            executed.append(True)

        event = model.schedule_at(callback, time=10.0)
        model.cancel_event(event)
        model.run_until(20.0)

        assert executed == []

    def test_model_run_methods(self):
        """Test model run methods."""
        model = Model()

        times = []

        def callback():
            times.append(model.time)

        model.schedule_at(callback, time=5.0)
        model.schedule_at(callback, time=10.0)

        model.run_for(7.0)
        assert times == [5.0]

        model.run_until(15.0)
        assert times == [5.0, 10.0]

    def test_hybrid_step_and_events(self):
        """Test combining step() with scheduled events."""

        class HybridModel(Model):
            def __init__(self):
                super().__init__()
                self.step_count = 0
                self.drought_happened = False

                # Schedule one-off event
                self.schedule_at(self.drought, time=50.0)

            @scheduled
            def step(self):
                self.step_count += 1

            def drought(self):
                self.drought_happened = True

        model = HybridModel()
        model.run_until(100.0)

        assert model.step_count == 100
        assert model.drought_happened is True

    def test_agent_self_scheduling(self):
        """Test agents scheduling their own events."""

        class SchedulingAgent(Agent):
            def __init__(self, model):
                super().__init__(model)
                self.released = False

            def get_arrested(self, sentence):
                self.model.schedule_after(self.release, delay=sentence)

            def release(self):
                self.released = True

        model = Model()
        agent = SchedulingAgent(model)

        agent.get_arrested(sentence=10)
        model.run_until(15.0)

        assert agent.released is True

    def test_pure_event_driven_no_step(self):
        """Test pure event-driven simulation without step."""

        class EventDrivenModel(Model):
            def __init__(self):
                super().__init__()
                self.arrival_times = []
                self.schedule_at(self.customer_arrival, time=1.0)

            def customer_arrival(self):
                self.arrival_times.append(self.time)
                if len(self.arrival_times) < 5:
                    # Schedule next arrival
                    next_time = self.time + 2.0
                    self.schedule_at(self.customer_arrival, time=next_time)

        model = EventDrivenModel()
        model.run_until(20.0)

        assert model.arrival_times == [1.0, 3.0, 5.0, 7.0, 9.0]

    def test_run_while_with_model_running_flag(self):
        """Test using run_while with model.running flag."""

        class StoppingModel(Model):
            def __init__(self):
                super().__init__()
                self.step_count = 0

            @scheduled
            def step(self):
                self.step_count += 1
                if self.step_count >= 10:
                    self.running = False

        model = StoppingModel()

        def condition(m):
            return m.running

        model.run_while(condition)

        assert model.step_count == 10
        assert model.running is False


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_event_list_run_until(self):
        """Test run_until with no scheduled events."""
        model = Model()
        model.run_until(100.0)

        assert model.time == 100.0

    def test_schedule_at_current_time(self):
        """Test scheduling an event at current time."""
        model = Model()
        model.time = 10.0

        executed = []

        def callback():
            executed.append(model.time)

        model.schedule_at(callback, time=10.0)
        model.run_until(10.0)

        assert executed == [10.0]

    def test_multiple_events_same_time_same_priority(self):
        """Test execution order for events at same time and priority."""
        model = Model()

        execution_order = []

        def callback1():
            execution_order.append(1)

        def callback2():
            execution_order.append(2)

        def callback3():
            execution_order.append(3)

        # Schedule with same time and priority
        model.schedule_at(callback1, time=5.0)
        model.schedule_at(callback2, time=5.0)
        model.schedule_at(callback3, time=5.0)

        model.run_until(10.0)

        # All should execute, order determined by unique_id
        assert len(execution_order) == 3
        assert set(execution_order) == {1, 2, 3}

    def test_run_next_event_empty_list(self):
        """Test run_next_event returns False when no events."""
        model = Model()

        result = model.run_next_event()

        assert result is False

    def test_scheduled_with_no_parentheses(self):
        """Test @scheduled decorator without parentheses."""

        class TestModel(Model):
            def __init__(self):
                super().__init__()
                self.count = 0

            @scheduled
            def step(self):
                self.count += 1

        model = TestModel()
        model.run_until(3.0)

        assert model.count == 3
