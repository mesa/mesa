"""Tests for mesa.experimental.actions."""
# ruff: noqa: D101 D102 D103 D107

import pytest

from mesa import Model
from mesa.experimental.actions import Action, ActionAgent, linear, step


# --- Helpers ---
class SimpleModel(Model):
    def __init__(self):
        super().__init__()


def make_agent(model=None):
    model = model or SimpleModel()
    return ActionAgent(model)


def effect_tracker():
    """Returns (callback, results_list) to track on_effect calls."""
    results = []

    def callback(agent, completion):
        results.append(round(completion, 4))

    return callback, results


# --- Reward curves ---
class TestRewardCurves:
    def test_linear(self):
        assert linear(0.0) == 0.0
        assert linear(0.5) == 0.5
        assert linear(1.0) == 1.0

    def test_step(self):
        assert step(0.0) == 0.0
        assert step(0.5) == 0.0
        assert step(0.99) == 0.0
        assert step(1.0) == 1.0


# --- Action ---
class TestAction:
    def test_defaults(self):
        a = Action("test")
        assert a.name == "test"
        assert a.duration == 1.0
        assert a.priority == 0.0
        assert a.interruptible is True
        assert a.progress == 0.0
        assert a.reward_curve is linear

    def test_effective_completion(self):
        a = Action("test", reward_curve=step)
        a.progress = 0.5
        assert a.effective_completion == 0.0
        a.progress = 1.0
        assert a.effective_completion == 1.0

    def test_remaining_time(self):
        a = Action("test", duration=10.0)
        assert a.remaining_time == 10.0
        a.progress = 0.3
        assert a.remaining_time == pytest.approx(7.0)

    def test_repr(self):
        a = Action("forage")
        assert "forage" in repr(a)


# --- ActionAgent basics ---
class TestActionAgent:
    def test_not_busy_initially(self):
        agent = make_agent()
        assert not agent.is_busy
        assert agent.current_action is None

    def test_start_action(self):
        agent = make_agent()
        agent.start_action(Action("work", duration=5.0))
        assert agent.is_busy
        assert agent.current_action.name == "work"

    def test_start_while_busy_raises(self):
        agent = make_agent()
        agent.start_action(Action("work", duration=5.0))
        with pytest.raises(ValueError, match="already performing"):
            agent.start_action(Action("other", duration=1.0))


# --- Completion ---
class TestCompletion:
    def test_completion_fires_on_effect(self):
        model = SimpleModel()
        agent = make_agent(model)
        cb, results = effect_tracker()

        agent.start_action(Action("work", duration=5.0, on_effect=cb))
        model.run_for(5)

        assert results == [1.0]
        assert not agent.is_busy

    def test_completion_clears_state(self):
        model = SimpleModel()
        agent = make_agent(model)
        action = Action("work", duration=3.0)
        agent.start_action(action)
        model.run_for(3)

        assert agent.current_action is None
        assert action.progress == 1.0
        assert action._event is None
        assert action._started_at is None

    def test_instantaneous_action(self):
        model = SimpleModel()
        agent = make_agent(model)
        cb, results = effect_tracker()

        agent.start_action(Action("instant", duration=0, on_effect=cb))
        assert results == [1.0]
        assert not agent.is_busy

    def test_no_on_effect_is_fine(self):
        model = SimpleModel()
        agent = make_agent(model)
        agent.start_action(Action("silent", duration=2.0))
        model.run_for(2)
        assert not agent.is_busy


# --- Interruption ---
class TestInterruption:
    def test_interrupt_applies_partial_reward(self):
        model = SimpleModel()
        agent = make_agent(model)
        cb, results = effect_tracker()

        agent.start_action(Action("work", duration=10.0, on_effect=cb))
        model.run_for(5)  # 50% progress
        agent.interrupt_for(Action("urgent", duration=1.0))

        assert results == [pytest.approx(0.5)]  # linear: 50% progress = 50% reward
        assert agent.current_action.name == "urgent"

    def test_interrupt_with_step_curve(self):
        model = SimpleModel()
        agent = make_agent(model)
        cb, results = effect_tracker()

        agent.start_action(
            Action("build", duration=10.0, reward_curve=step, on_effect=cb)
        )
        model.run_for(8)  # 80% progress
        agent.interrupt_for(Action("urgent", duration=1.0))

        assert results == [0.0]  # step: < 100% = no reward

    def test_not_interruptible(self):
        model = SimpleModel()
        agent = make_agent(model)
        cb, results = effect_tracker()

        agent.start_action(
            Action("critical", duration=10.0, interruptible=False, on_effect=cb)
        )
        model.run_for(5)
        agent.interrupt_for(Action("nope", duration=1.0))

        # Should still be doing the original action
        assert agent.current_action.name == "critical"
        assert results == []

    def test_interrupt_when_idle_starts_action(self):
        agent = make_agent()
        agent.interrupt_for(Action("fresh", duration=3.0))
        assert agent.current_action.name == "fresh"


# --- Cancellation ---
class TestCancellation:
    def test_cancel_applies_partial_reward(self):
        model = SimpleModel()
        agent = make_agent(model)
        cb, results = effect_tracker()

        agent.start_action(Action("work", duration=4.0, on_effect=cb))
        model.run_for(1)  # 25% progress
        agent.cancel_action()

        assert results == [pytest.approx(0.25)]
        assert not agent.is_busy

    def test_cancel_when_idle_is_noop(self):
        agent = make_agent()
        agent.cancel_action()  # Should not raise
        assert not agent.is_busy


# --- Custom reward curves ---
class TestCustomCurves:
    def test_quadratic_curve(self):
        model = SimpleModel()
        agent = make_agent(model)
        cb, results = effect_tracker()

        def quadratic(p):
            return p**2

        agent.start_action(
            Action("study", duration=10.0, reward_curve=quadratic, on_effect=cb)
        )
        model.run_for(5)  # 50% progress
        agent.cancel_action()

        assert results == [pytest.approx(0.25)]  # 0.5^2 = 0.25


# --- Agent removal ---
class TestRemoval:
    def test_remove_cancels_action(self):
        model = SimpleModel()
        agent = make_agent(model)
        action = Action("work", duration=10.0)
        agent.start_action(action)
        agent.remove()

        assert action._event is None or action._event.CANCELED


# --- Integration: full sequence ---
class TestIntegration:
    def test_start_interrupt_complete(self):
        """Start action A, interrupt with B halfway, let B complete."""
        model = SimpleModel()
        agent = make_agent(model)
        cb_a, results_a = effect_tracker()
        cb_b, results_b = effect_tracker()

        agent.start_action(Action("A", duration=10.0, on_effect=cb_a))
        model.run_for(5)  # A at 50%
        agent.interrupt_for(Action("B", duration=4.0, on_effect=cb_b))
        model.run_for(4)  # B completes

        assert results_a == [pytest.approx(0.5)]  # A interrupted at 50%
        assert results_b == [1.0]  # B completed
        assert not agent.is_busy

    def test_multiple_interrupts(self):
        """Chain of interruptions with increasing priority."""
        model = SimpleModel()
        agent = make_agent(model)
        cb, results = effect_tracker()

        agent.start_action(Action("low", duration=10.0, priority=1, on_effect=cb))
        model.run_for(2)  # 20%
        agent.interrupt_for(Action("mid", duration=10.0, priority=5, on_effect=cb))
        model.run_for(3)  # 30%
        agent.interrupt_for(Action("high", duration=2.0, priority=10, on_effect=cb))
        model.run_for(2)  # completes

        assert len(results) == 3
        assert results[0] == pytest.approx(0.2)  # low interrupted at 20%
        assert results[1] == pytest.approx(0.3)  # mid interrupted at 30%
        assert results[2] == 1.0  # high completed
