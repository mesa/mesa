"""Tests for Model public event scheduling and time advancement API."""
# ruff: noqa: D101 D102 D107

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
        self.steps_taken += 1


class SimpleModel(Model):
    def __init__(self, n=3):
        super().__init__()
        StepAgent.create_agents(self, n)

    def step(self):
        self.agents.shuffle_do("step")


# --- run_for / run_until ---
class TestRunFor:
    def test_single_unit(self):
        model = SimpleModel()
        model.run_for(1)
        assert model.time == 1.0

    def test_multiple_units(self):
        model = SimpleModel()
        model.run_for(10)
        assert model.time == 10.0

    def test_agents_activated(self):
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
        model = SimpleModel()
        model.run_for(5)
        model.run_for(5)
        assert model.time == 10.0


class TestRunUntil:
    def test_basic(self):
        model = SimpleModel()
        model.run_until(5.0)
        assert model.time == 5.0

    def test_already_past(self):
        model = SimpleModel()
        model.run_for(10)
        with pytest.warns(RuntimeWarning):
            model.run_until(5.0)  # already past t=5
            assert model.time == 10

    def test_sequential(self):
        model = SimpleModel()
        model.run_until(3.0)
        model.run_until(7.0)
        assert model.time == 7.0


# --- schedule_event ---
class TestScheduleEvent:
    def test_at(self):
        model = SimpleModel()
        log = []

        def fire():
            log.append("fired")

        model.schedule_event(fire, at=2.5)
        model.run_for(3)
        assert "fired" in log

    def test_after(self):
        model = SimpleModel()
        log = []

        def fire():
            log.append("fired")

        model.run_for(5)
        model.schedule_event(fire, after=2.0)
        model.run_for(3)
        assert "fired" in log
        assert model.time == 8.0

    def test_not_yet_reached(self):
        model = SimpleModel()
        log = []

        def fire():
            log.append("fired")

        model.schedule_event(fire, at=10.0)
        model.run_for(3)
        assert log == []

    def test_cancel(self):
        model = SimpleModel()
        log = []

        def fire():
            log.append("fired")

        event = model.schedule_event(fire, at=2.0)
        event.cancel()
        model.run_for(5)
        assert log == []

    def test_at_and_after_exclusive(self):
        model = SimpleModel()

        def noop():
            pass

        with pytest.raises(ValueError):
            model.schedule_event(noop, at=1.0, after=1.0)
        with pytest.raises(ValueError):
            model.schedule_event(noop)

    def test_inline_lambda_with_strong_reference(self):
        model = SimpleModel()
        log = []

        def callback():
            log.append("fired")

        model.schedule_event(callback, at=1.0)
        model.run_for(2.0)
        assert log == ["fired"]

    def test_partial_callback_with_strong_reference(self):
        model = SimpleModel()
        log = []

        def fire(label):
            log.append(label)

        callback = partial(fire, "x")
        model.schedule_event(callback, at=1.0)
        model.run_for(2.0)
        assert log == ["x"]


# --- schedule_recurring ---
class TestScheduleRecurring:
    def test_fixed_interval(self):
        model = SimpleModel()
        log = []

        def record():
            log.append(model.time)

        model.schedule_recurring(record, Schedule(interval=2.0, start=2.0))
        model.run_for(10)
        assert log == [2.0, 4.0, 6.0, 8.0, 10.0]

    def test_fire_and_forget_survives_gc(self):
        """Generator must work even when user doesn't save the return value."""
        model = SimpleModel()
        log = []

        def record():
            log.append(model.time)

        model.schedule_recurring(record, Schedule(interval=2.0, start=2.0))
        gc.collect()  # Force GC — would kill the generator without the fix
        model.run_for(10)
        assert log == [2.0, 4.0, 6.0, 8.0, 10.0]

    def test_stop_generator(self):
        model = SimpleModel()
        log = []

        def record():
            log.append(model.time)

        gen = model.schedule_recurring(record, Schedule(interval=1.0, start=1.0))
        model.run_for(3)
        gen.stop()
        model.run_for(3)
        assert len(log) == 3

    def test_with_count(self):
        model = SimpleModel()
        log = []

        def record():
            log.append(model.time)

        model.schedule_recurring(record, Schedule(interval=1.0, start=1.0, count=3))
        model.run_for(10)
        assert len(log) == 3


# --- schedule validation guards ---
class TestScheduleValidation:
    def test_rejects_nonpositive_fixed_interval(self):
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
        """Test that Schedule raises an error if start is greater than end."""
        with pytest.raises(ValueError):
            Schedule(interval=1.0, start=10, end=5)
