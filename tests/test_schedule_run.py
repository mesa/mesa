"""Tests for Model public event scheduling and time advancement API."""
# ruff: noqa: D101, D107

import gc
from functools import partial

import pytest

from mesa import Agent, Model
from mesa.time import Schedule


class StepAgent(Agent):
    def __init__(self, model):
        super().__init__(model)
        self.steps_taken = 0

    def step(self):
        """Run one step."""
        self.steps_taken += 1


class SimpleModel(Model):
    def __init__(self, n=3):
        super().__init__()
        StepAgent.create_agents(self, n)

    def step(self):
        """Run one step."""
        self.agents.shuffle_do("step")


# --- run_for / run_until ---
class TestRunFor:
    def test_single_unit(self):
        """Test single unit."""
        model = SimpleModel()
        model.run_for(1)
        assert model.time == 1.0

    def test_multiple_units(self):
        """Test multiple units."""
        model = SimpleModel()
        model.run_for(10)
        assert model.time == 10.0

    def test_agents_activated(self):
        """Test agents activated."""
        model = SimpleModel(n=5)
        model.run_for(3)
        for agent in model.agents:
            assert agent.steps_taken == 3

    def test_equivalent_to_step(self):
        """run_for(1) should produce the same result as step()."""
        m1, m2 = SimpleModel(n=3), SimpleModel(n=3)
        for _ in range(5):
            m1.step()
            m2.run_for(1)
        assert m1.time == m2.time == 5.0

    def test_sequential_calls(self):
        """Test sequential calls."""
        model = SimpleModel()
        model.run_for(5)
        model.run_for(5)
        assert model.time == 10.0


class TestRunUntil:
    def test_basic(self):
        """Test basic."""
        model = SimpleModel()
        model.run_until(5.0)
        assert model.time == 5.0

    def test_already_past(self):
        """Test already past."""
        model = SimpleModel()
        model.run_for(10)
        with pytest.warns(RuntimeWarning):
            model.run_until(5.0)  # already past t=5
            assert model.time == 10

    def test_sequential(self):
        """Test sequential."""
        model = SimpleModel()
        model.run_until(3.0)
        model.run_until(7.0)
        assert model.time == 7.0


# --- schedule_event ---
class TestScheduleEvent:
    def test_at(self):
        """Test at."""
        model = SimpleModel()
        log = []

        def fire():
            """Record that the event fired."""
            log.append("fired")

        model.schedule_event(fire, at=2.5)
        model.run_for(3)
        assert "fired" in log

    def test_after(self):
        """Test after."""
        model = SimpleModel()
        log = []

        def fire():
            """Record that the event fired."""
            log.append("fired")

        model.run_for(5)
        model.schedule_event(fire, after=2.0)
        model.run_for(3)
        assert "fired" in log
        assert model.time == 8.0

    def test_not_yet_reached(self):
        """Test not yet reached."""
        model = SimpleModel()
        log = []

        def fire():
            """Record that the event fired."""
            log.append("fired")

        model.schedule_event(fire, at=10.0)
        model.run_for(3)
        assert log == []

    def test_cancel(self):
        """Test cancel."""
        model = SimpleModel()
        log = []

        def fire():
            """Record that the event fired."""
            log.append("fired")

        event = model.schedule_event(fire, at=2.0)
        event.cancel()
        model.run_for(5)
        assert log == []

    def test_at_and_after_exclusive(self):
        """Test at and after exclusive."""
        model = SimpleModel()

        def noop():
            """Do nothing."""

        with pytest.raises(ValueError):
            model.schedule_event(noop, at=1.0, after=1.0)
        with pytest.raises(ValueError):
            model.schedule_event(noop)

    def test_inline_lambda_with_strong_reference(self):
        """Test inline lambda with strong reference."""
        model = SimpleModel()
        log = []

        def callback():
            """Handle the callback."""
            log.append("fired")

        model.schedule_event(callback, at=1.0)
        model.run_for(2.0)
        assert log == ["fired"]

    def test_partial_callback_with_strong_reference(self):
        """Test partial callback with strong reference."""
        model = SimpleModel()
        log = []

        def fire(label):
            """Record that the event fired."""
            log.append(label)

        callback = partial(fire, "x")
        model.schedule_event(callback, at=1.0)
        model.run_for(2.0)
        assert log == ["x"]

    def test_rejects_past_time(self):
        """schedule_event should not allow scheduling in the past."""
        model = SimpleModel()
        model.run_until(10)

        def noop():
            """Do nothing."""

        with pytest.raises(ValueError, match="Cannot schedule event in the past"):
            model.schedule_event(noop, at=5)


# --- schedule_recurring ---
class TestScheduleRecurring:
    def test_fixed_interval(self):
        """Test fixed interval."""
        model = SimpleModel()
        log = []

        def record():
            """Record the current time."""
            log.append(model.time)

        model.schedule_recurring(record, Schedule(interval=2.0, start=2.0))
        model.run_for(10)
        assert log == [2.0, 4.0, 6.0, 8.0, 10.0]

    def test_fire_and_forget_survives_gc(self):
        """Generator must work even when user doesn't save the return value."""
        model = SimpleModel()
        log = []

        def record():
            """Record the current time."""
            log.append(model.time)

        model.schedule_recurring(record, Schedule(interval=2.0, start=2.0))
        gc.collect()  # Force GC — would kill the generator without the fix
        model.run_for(10)
        assert log == [2.0, 4.0, 6.0, 8.0, 10.0]

    def test_stop_generator(self):
        """Test stop generator."""
        model = SimpleModel()
        log = []

        def record():
            """Record the current time."""
            log.append(model.time)

        gen = model.schedule_recurring(record, Schedule(interval=1.0, start=1.0))
        model.run_for(3)
        gen.stop()
        model.run_for(3)
        assert len(log) == 3

    def test_with_count(self):
        """Test with count."""
        model = SimpleModel()
        log = []

        def record():
            """Record the current time."""
            log.append(model.time)

        model.schedule_recurring(record, Schedule(interval=1.0, start=1.0, count=3))
        model.run_for(10)
        assert len(log) == 3

    def test_rejects_past_start(self):
        """schedule_recurring should not allow start time in the past."""
        model = SimpleModel()
        model.run_until(10)

        def noop():
            """Do nothing."""

        with pytest.raises(
            ValueError, match="Cannot start recurring schedule in the past"
        ):
            model.schedule_recurring(noop, Schedule(interval=1.0, start=3.0))


class TestEdgeCases:
    def test_schedule_event_at_zero(self):
        """Event scheduled at t=0 should fire during the first run."""
        model = SimpleModel()
        log = []

        def fire():
            """Record that the event fired."""
            log.append("fired")

        model.schedule_event(fire, at=0.0)
        model.run_for(1)
        assert "fired" in log

    def test_event_and_recurring_interact(self):
        """One-off and recurring events coexist correctly."""
        model = SimpleModel()
        log = []

        def one_off():
            """Record the one-off event."""
            log.append(("once", model.time))

        def recurring():
            """Record the recurring event."""
            log.append(("repeat", model.time))

        model.schedule_event(one_off, at=2.5)
        model.schedule_recurring(recurring, Schedule(interval=2.0, start=2.0))
        model.run_for(6)

        one_off_events = [t for label, t in log if label == "once"]
        recurring_events = [t for label, t in log if label == "repeat"]

        assert one_off_events == [2.5]
        assert recurring_events == [2.0, 4.0, 6.0]
