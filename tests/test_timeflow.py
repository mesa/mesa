"""Tests for unified time and event scheduling API."""

import random

import pytest

from mesa import Agent, Model
from mesa.experimental.devs.eventlist import Priority, RecurringEvent
from mesa.timeflow import RunControl, Scheduler


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
        # Note: Model.__init__ schedules step, so we use a fresh scheduler here
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


class TestCoreScheduleMethod:
    """Test the core schedule() method with all its options."""

    def test_schedule_one_off_immediate(self):
        """Test scheduling a one-off event at current time."""
        model = Model()
        scheduler = Scheduler(model)

        executed = []

        def callback():
            executed.append(model.time)

        # No start_at or start_after means immediate (current time)
        event = scheduler.schedule(callback)

        assert event.time == 0.0
        assert not isinstance(event, RecurringEvent)

    def test_schedule_one_off_start_at(self):
        """Test scheduling a one-off event at absolute time."""
        model = Model()
        scheduler = Scheduler(model)

        executed = []

        def callback():
            executed.append(model.time)

        event = scheduler.schedule(callback, start_at=25.0)

        assert event.time == 25.0
        assert not isinstance(event, RecurringEvent)

    def test_schedule_one_off_start_after(self):
        """Test scheduling a one-off event after delay."""
        model = Model()
        model.time = 10.0
        scheduler = Scheduler(model)

        executed = []

        def callback():
            executed.append(model.time)

        event = scheduler.schedule(callback, start_after=5.0)

        assert event.time == 15.0

    def test_schedule_recurring_fixed_interval(self):
        """Test scheduling a recurring event with fixed interval."""
        model = Model()
        scheduler = Scheduler(model)
        run_control = RunControl(model, scheduler)

        execution_times = []

        def callback():
            execution_times.append(model.time)

        event = scheduler.schedule(callback, interval=3.0)

        assert isinstance(event, RecurringEvent)

        run_control.run_until(10.0)

        # Should execute at t=3, 6, 9
        assert execution_times == [3.0, 6.0, 9.0]

    def test_schedule_recurring_with_start_at(self):
        """Test recurring event with explicit start time."""
        model = Model()
        scheduler = Scheduler(model)
        run_control = RunControl(model, scheduler)

        execution_times = []

        def callback():
            execution_times.append(model.time)

        scheduler.schedule(callback, start_at=5.0, interval=2.0)

        run_control.run_until(12.0)

        # Should execute at t=5, 7, 9, 11
        assert execution_times == [5.0, 7.0, 9.0, 11.0]

    def test_schedule_recurring_callable_interval(self):
        """Test recurring event with callable interval (variable timing)."""
        model = Model()
        scheduler = Scheduler(model)
        run_control = RunControl(model, scheduler)

        execution_times = []
        interval_sequence = iter([2.0, 3.0, 1.0, 5.0])

        def get_interval(m):
            return next(interval_sequence, 100.0)  # Large default to stop

        def callback():
            execution_times.append(model.time)

        scheduler.schedule(callback, interval=get_interval)

        run_control.run_until(15.0)

        # First at 2.0, then +3.0=5.0, then +1.0=6.0, then +5.0=11.0
        assert execution_times == [2.0, 5.0, 6.0, 11.0]

    def test_schedule_recurring_with_count(self):
        """Test recurring event with count limit."""
        model = Model()
        scheduler = Scheduler(model)
        run_control = RunControl(model, scheduler)

        execution_times = []

        def callback():
            execution_times.append(model.time)

        scheduler.schedule(callback, interval=2.0, count=3)

        run_control.run_until(20.0)

        # Should only execute 3 times: t=2, 4, 6
        assert execution_times == [2.0, 4.0, 6.0]

    def test_schedule_recurring_with_end_at(self):
        """Test recurring event with end_at limit."""
        model = Model()
        scheduler = Scheduler(model)
        run_control = RunControl(model, scheduler)

        execution_times = []

        def callback():
            execution_times.append(model.time)

        scheduler.schedule(callback, interval=3.0, end_at=10.0)

        run_control.run_until(20.0)

        # Should execute at t=3, 6, 9 (not 12, which is > end_at)
        assert execution_times == [3.0, 6.0, 9.0]

    def test_schedule_recurring_with_end_after(self):
        """Test recurring event with end_after limit (from first execution)."""
        model = Model()
        scheduler = Scheduler(model)
        run_control = RunControl(model, scheduler)

        execution_times = []

        def callback():
            execution_times.append(model.time)

        # Start at 5, run for 7 time units after first execution
        scheduler.schedule(callback, start_at=5.0, interval=2.0, end_after=7.0)

        run_control.run_until(20.0)

        # First execution at t=5, end_after=7 means stop after t=12
        # So: t=5, 7, 9, 11 (next would be 13, which is > 5+7=12)
        assert execution_times == [5.0, 7.0, 9.0, 11.0]

    def test_schedule_conflicting_start_params_raises(self):
        """Test that specifying both start_at and start_after raises error."""
        model = Model()
        scheduler = Scheduler(model)

        def callback():
            pass

        with pytest.raises(
            ValueError, match="Cannot specify both start_at and start_after"
        ):
            scheduler.schedule(callback, start_at=10.0, start_after=5.0)

    def test_schedule_conflicting_end_params_raises(self):
        """Test that specifying multiple end conditions raises error."""
        model = Model()
        scheduler = Scheduler(model)

        def callback():
            pass

        with pytest.raises(ValueError, match="Can only specify one of"):
            scheduler.schedule(callback, interval=1.0, count=5, end_at=10.0)

        with pytest.raises(ValueError, match="Can only specify one of"):
            scheduler.schedule(callback, interval=1.0, count=5, end_after=10.0)

        with pytest.raises(ValueError, match="Can only specify one of"):
            scheduler.schedule(callback, interval=1.0, end_at=10.0, end_after=5.0)


class TestRecurringEventControl:
    """Test pause/resume/stop methods on RecurringEvent."""

    def test_recurring_event_pause(self):
        """Test pausing a recurring event."""
        model = Model()
        scheduler = Scheduler(model)
        run_control = RunControl(model, scheduler)

        execution_times = []

        def callback():
            execution_times.append(model.time)

        event = scheduler.schedule(callback, interval=2.0)

        # Run until t=5, should execute at t=2, 4
        run_control.run_until(5.0)
        assert execution_times == [2.0, 4.0]

        # Pause the event
        event.pause()

        # Run more - no new executions
        run_control.run_until(10.0)
        assert execution_times == [2.0, 4.0]

    def test_recurring_event_resume(self):
        """Test resuming a paused recurring event."""
        model = Model()
        scheduler = Scheduler(model)
        run_control = RunControl(model, scheduler)

        execution_times = []

        def callback():
            execution_times.append(model.time)

        event = scheduler.schedule(callback, interval=2.0)

        # Run until t=5
        run_control.run_until(5.0)
        assert execution_times == [2.0, 4.0]

        # Pause
        event.pause()
        run_control.run_until(8.0)
        assert execution_times == [2.0, 4.0]

        # Resume - should schedule next at current_time + interval
        event.resume()
        run_control.run_until(15.0)

        # After resume at t=8, next execution at t=10, then t=12, t=14
        assert execution_times == [2.0, 4.0, 10.0, 12.0, 14.0]

    def test_recurring_event_stop(self):
        """Test permanently stopping a recurring event."""
        model = Model()
        scheduler = Scheduler(model)
        run_control = RunControl(model, scheduler)

        execution_times = []

        def callback():
            execution_times.append(model.time)

        event = scheduler.schedule(callback, interval=2.0)

        run_control.run_until(5.0)
        assert execution_times == [2.0, 4.0]

        # Stop permanently
        event.cancel()

        run_control.run_until(10.0)
        assert execution_times == [2.0, 4.0]

        # Cannot resume after stop
        event.resume()
        run_control.run_until(15.0)
        assert execution_times == [2.0, 4.0]  # Still no new executions


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


class TestModelIntegration:
    """Test integration of timeflow with Model class."""

    def test_model_has_step_event(self):
        """Test that Model automatically schedules step as heartbeat."""
        model = Model()

        assert hasattr(model, "step_event")
        assert isinstance(model.step_event, RecurringEvent)

    def test_model_step_heartbeat(self):
        """Test that step() is called automatically as heartbeat."""

        class CountingModel(Model):
            def __init__(self):
                super().__init__()
                self.step_count = 0

            def step(self):
                self.step_count += 1

        model = CountingModel()
        model.run_for(5)

        assert model.step_count == 5

    def test_model_schedule_method(self):
        """Test the unified model.schedule() method."""
        model = Model()

        execution_times = []

        def callback():
            execution_times.append(model.time)

        # One-off
        model.schedule(callback, start_at=5.0)

        # Recurring
        model.schedule(callback, interval=10.0, count=2)

        model.run_until(30.0)

        # One-off at 5, recurring at 10, 20
        assert 5.0 in execution_times
        assert 10.0 in execution_times
        assert 20.0 in execution_times

    def test_model_schedule_at(self):
        """Test model.schedule_at method."""
        model = Model()

        results = []

        def callback():
            results.append(model.time)

        model.schedule_at(callback, time=15.0)
        model.run_until(20.0)

        assert 15.0 in results
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

        assert 15.0 in results

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
        assert 5.0 in times
        assert 10.0 not in times

        model.run_until(15.0)
        assert 10.0 in times

    def test_hybrid_step_and_events(self):
        """Test combining step() with scheduled events."""

        class HybridModel(Model):
            def __init__(self):
                super().__init__()
                self.step_count = 0
                self.drought_happened = False

                # Schedule one-off event
                self.schedule_at(self.drought, time=50.0)

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

    def test_pure_event_driven_model(self):
        """Test pure event-driven simulation (step does nothing special)."""

        class EventDrivenModel(Model):
            def __init__(self):
                super().__init__()
                self.arrival_times = []
                # Cancel the default step heartbeat
                self.step_event.cancel()
                # Bootstrap event chain
                self.schedule_at(self.customer_arrival, time=1.0)

            def step(self):
                pass  # Not used

            def customer_arrival(self):
                self.arrival_times.append(self.time)
                if len(self.arrival_times) < 5:
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

            def step(self):
                self.step_count += 1
                if self.step_count >= 10:
                    self.running = False

        model = StoppingModel()
        model.run_while(lambda m: m.running)

        assert model.step_count == 10
        assert model.running is False

    def test_deprecated_direct_step_call(self):
        """Test that calling model.step() directly shows deprecation warning."""
        model = Model()

        with pytest.warns(PendingDeprecationWarning, match="deprecated"):
            model.step()


class TestStochasticIntervals:
    """Test stochastic/variable interval scheduling."""

    def test_exponential_arrivals(self):
        """Test Poisson process with exponential inter-arrival times."""
        model = Model()
        # Cancel default step
        model.step_event.cancel()

        rng = random.Random(42)
        arrival_times = []

        def arrival():
            arrival_times.append(model.time)

        # Schedule with exponential intervals
        model.schedule(
            arrival,
            interval=lambda m: rng.expovariate(1.0),  # rate = 1
            count=10,
        )

        model.run_until(100.0)

        assert len(arrival_times) == 10
        # Verify times are increasing
        for i in range(1, len(arrival_times)):
            assert arrival_times[i] > arrival_times[i - 1]

    def test_variable_interval_based_on_model_state(self):
        """Test interval that depends on model state."""

        class AdaptiveModel(Model):
            def __init__(self):
                super().__init__()
                self.activity_rate = 2.0
                self.step_event.cancel()
                self.check_times = []

                self.schedule(
                    self.check,
                    interval=lambda m: m.activity_rate,
                    count=5,
                )

            def check(self):
                self.check_times.append(self.time)
                # Slow down over time
                self.activity_rate += 1.0

        model = AdaptiveModel()
        model.run_until(50.0)

        # First interval is 2, then 3, then 4, then 5, then 6
        # Times: 2, 5, 9, 14, 20
        assert model.check_times == [2.0, 5.0, 9.0, 14.0, 20.0]


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_event_list_run_until(self):
        """Test run_until with no scheduled events."""
        model = Model()
        # Cancel the default step
        model.step_event.cancel()

        model.run_until(100.0)

        assert model.time == 100.0

    def test_schedule_at_current_time(self):
        """Test scheduling an event at current time."""
        model = Model()
        model.step_event.cancel()
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
        model.step_event.cancel()

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
        model.step_event.cancel()

        result = model.run_next_event()

        assert result is False

    def test_pause_resume_maintains_count(self):
        """Test that pause/resume maintains execution count."""
        model = Model()
        model.step_event.cancel()

        execution_times = []

        def callback():
            execution_times.append(model.time)

        event = model.schedule(callback, interval=2.0, count=5)

        # Run until 2 executions
        model.run_until(5.0)
        assert len(execution_times) == 2  # t=2, 4

        # Pause and resume
        event.pause()
        model.run_until(8.0)
        event.resume()
        model.run_until(20.0)

        # Should only have 5 total executions due to count limit
        assert len(execution_times) == 5

    def test_end_after_measures_from_first_execution(self):
        """Test that end_after is measured from first execution, not schedule time."""
        model = Model()
        model.step_event.cancel()

        execution_times = []

        def callback():
            execution_times.append(model.time)

        # Start at t=10, run for 5 time units after first execution
        model.schedule(callback, start_at=10.0, interval=2.0, end_after=5.0)

        model.run_until(30.0)

        # First at 10, end_after=5 means stop after t=15
        # So: 10, 12, 14 (16 would be > 15)
        assert execution_times == [10.0, 12.0, 14.0]
