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


# ---------------------------------------------------------------------------
# Phase 2 – reschedule_on_interrupt=False (explicit discard, default behavior)
# ---------------------------------------------------------------------------


class TestRescheduleDiscard:
    def test_interrupted_action_not_in_queue(self):
        """Default reschedule_on_interrupt=False: interrupted action is discarded."""
        model = SimpleModel()
        agent = make_agent(model)

        work = Action("work", duration=10.0)  # reschedule_on_interrupt=False by default
        agent.start_action(work)
        model.run_for(5)
        agent.interrupt_for(Action("urgent", duration=1.0))

        assert work not in agent.action_queue

    def test_on_effect_called_once_on_interrupt(self):
        """on_effect is called exactly once (at interruption) and never again."""
        model = SimpleModel()
        agent = make_agent(model)
        cb, results = effect_tracker()

        agent.start_action(Action("work", duration=10.0, on_effect=cb))
        model.run_for(4)  # 40%
        agent.interrupt_for(Action("urgent", duration=1.0))
        model.run_for(1)  # urgent completes

        # only the 40% partial reward, urgent has no on_effect
        assert results == [pytest.approx(0.4)]


# ---------------------------------------------------------------------------
# Phase 2 – reschedule_on_interrupt="remainder"
# ---------------------------------------------------------------------------


class TestRescheduleRemainder:
    def test_interrupted_action_pushed_to_front_of_queue(self):
        """Interrupted remainder action is at index 0 of the queue."""
        model = SimpleModel()
        agent = make_agent(model)

        forage = Action("forage", duration=10.0, reschedule_on_interrupt="remainder")
        agent.start_action(forage)
        model.run_for(5)  # 50% done
        agent.interrupt_for(Action("flee", duration=1.0))

        assert agent.action_queue[0] is forage

    def test_progress_preserved_in_queue(self):
        """Progress at interruption point is stored on the re-queued action."""
        model = SimpleModel()
        agent = make_agent(model)

        forage = Action("forage", duration=10.0, reschedule_on_interrupt="remainder")
        agent.start_action(forage)
        model.run_for(6)  # 60% done
        agent.interrupt_for(Action("flee", duration=1.0))

        assert forage.progress == pytest.approx(0.6)

    def test_resumes_from_remaining_duration(self):
        """After fleeing completes, forage auto-resumes and only runs remaining time."""
        model = SimpleModel()
        agent = make_agent(model)
        cb, results = effect_tracker()

        forage = Action(
            "forage",
            duration=10.0,
            reschedule_on_interrupt="remainder",
            on_effect=cb,
        )
        agent.start_action(forage)
        model.run_for(5)  # 50% done, 5 time units remaining
        agent.interrupt_for(Action("flee", duration=2.0))
        # flee completes at +2 → forage re-queued action auto-resumes
        # forage now needs 5 more time units to finish
        model.run_for(2)  # flee finishes, forage auto-starts with 5 remaining
        assert agent.current_action is forage  # resuming
        model.run_for(5)  # finish remaining 50%

        assert results[0] == pytest.approx(0.5)  # partial at interruption
        assert results[1] == pytest.approx(1.0)  # full completion
        assert not agent.is_busy

    def test_remainder_placed_before_pre_existing_queue_entries(self):
        """Re-queued remainder action goes to front, ahead of any queued actions."""
        model = SimpleModel()
        agent = make_agent(model)

        forage = Action("forage", duration=10.0, reschedule_on_interrupt="remainder")
        queued_first = Action("queued", duration=1.0)

        agent.start_action(forage)
        agent.action_queue.append(queued_first)  # manually enqueue
        model.run_for(3)
        agent.interrupt_for(Action("flee", duration=1.0))

        # forage (interrupted) must be before queued_first
        assert agent.action_queue[0] is forage
        assert agent.action_queue[1] is queued_first


# ---------------------------------------------------------------------------
# Phase 2 – reschedule_on_interrupt="full"
# ---------------------------------------------------------------------------


class TestRescheduleFull:
    def test_interrupted_action_pushed_to_front_with_reset_progress(self):
        """Full-reschedule action is in queue at index 0 with progress=0."""
        model = SimpleModel()
        agent = make_agent(model)

        build = Action("build", duration=10.0, reschedule_on_interrupt="full")
        agent.start_action(build)
        model.run_for(7)  # 70% done
        agent.interrupt_for(Action("emergency", duration=1.0))

        assert agent.action_queue[0] is build
        assert build.progress == pytest.approx(0.0)  # reset

    def test_partial_reward_given_at_interruption(self):
        """on_effect fires with partial reward at the moment of interruption."""
        model = SimpleModel()
        agent = make_agent(model)
        cb, results = effect_tracker()

        build = Action(
            "build",
            duration=10.0,
            reschedule_on_interrupt="full",
            on_effect=cb,
        )
        agent.start_action(build)
        model.run_for(6)  # 60% progress
        agent.interrupt_for(Action("emergency", duration=1.0))

        assert results == [pytest.approx(0.6)]

    def test_restarts_from_full_duration_after_resume(self):
        """Full-reschedule action runs its entire duration again when resumed."""
        model = SimpleModel()
        agent = make_agent(model)
        cb, results = effect_tracker()

        build = Action(
            "build",
            duration=4.0,
            reschedule_on_interrupt="full",
            on_effect=cb,
        )
        agent.start_action(build)
        model.run_for(2)  # 50% done
        agent.interrupt_for(Action("emergency", duration=1.0))
        model.run_for(1)  # emergency completes, build auto-resumes from 0
        model.run_for(4)  # build needs full 4 time units

        assert results[-1] == pytest.approx(1.0)
        assert not agent.is_busy


# ---------------------------------------------------------------------------
# Phase 2 – Action queue mechanics
# ---------------------------------------------------------------------------


class TestActionQueue:
    def test_queue_empty_initially(self):
        agent = make_agent()
        assert agent.action_queue == []

    def test_queue_drains_after_completion(self):
        """After current action completes, the next queued action auto-starts."""
        model = SimpleModel()
        agent = make_agent(model)

        a1 = Action("first", duration=3.0)
        a2 = Action("second", duration=2.0)
        agent.start_action(a1)
        agent.action_queue.append(a2)

        model.run_for(3)  # a1 completes
        assert agent.current_action is a2
        model.run_for(2)  # a2 completes
        assert not agent.is_busy

    def test_queue_order_is_fifo(self):
        """Queued actions run in the order they were appended."""
        model = SimpleModel()
        agent = make_agent(model)
        order = []

        def track(name):
            def cb(agent, c):
                order.append(name)

            return cb

        agent.start_action(Action("first", duration=1.0, on_effect=track("first")))
        agent.action_queue.append(
            Action("second", duration=1.0, on_effect=track("second"))
        )
        agent.action_queue.append(
            Action("third", duration=1.0, on_effect=track("third"))
        )

        model.run_for(3)
        assert order == ["first", "second", "third"]

    def test_cancel_drains_queue(self):
        """cancel_action with default clear_queue=False auto-starts next queued action."""
        model = SimpleModel()
        agent = make_agent(model)

        a1 = Action("first", duration=10.0)
        a2 = Action("second", duration=2.0)
        agent.start_action(a1)
        agent.action_queue.append(a2)
        agent.cancel_action()

        assert agent.current_action is a2

    def test_cancel_with_clear_queue_true(self):
        """cancel_action(clear_queue=True) leaves agent idle with empty queue."""
        model = SimpleModel()
        agent = make_agent(model)
        cb, results = effect_tracker()

        agent.start_action(Action("first", duration=5.0))
        agent.action_queue.append(Action("second", duration=2.0, on_effect=cb))
        agent.cancel_action(clear_queue=True)

        assert not agent.is_busy
        assert agent.action_queue == []
        model.run_for(5)
        assert results == []  # second never ran

    def test_remove_clears_queue(self):
        """Removing an agent discards all queued actions."""
        model = SimpleModel()
        agent = make_agent(model)
        agent.start_action(Action("work", duration=5.0))
        agent.action_queue.append(Action("more", duration=3.0))
        agent.remove()

        assert agent.action_queue == []


# ---------------------------------------------------------------------------
# Phase 2 – request_action (priority-based preemption)
# ---------------------------------------------------------------------------


class TestRequestAction:
    def test_starts_immediately_when_idle(self):
        agent = make_agent()
        action = Action("work", duration=5.0)
        agent.request_action(action)
        assert agent.current_action is action

    def test_higher_priority_preempts(self):
        """New action with strictly higher priority interrupts the running action."""
        model = SimpleModel()
        agent = make_agent(model)
        cb, results = effect_tracker()

        low = Action("low", duration=10.0, priority=1.0, on_effect=cb)
        high = Action("high", duration=2.0, priority=5.0)

        agent.start_action(low)
        model.run_for(5)  # 50% progress on low
        agent.request_action(high)

        assert agent.current_action is high
        assert results == [pytest.approx(0.5)]

    def test_equal_priority_enqueues(self):
        """New action with equal priority is appended to the queue."""
        model = SimpleModel()
        agent = make_agent(model)

        current = Action("current", duration=10.0, priority=3.0)
        incoming = Action("incoming", duration=2.0, priority=3.0)
        agent.start_action(current)
        agent.request_action(incoming)

        assert agent.current_action is current
        assert incoming in agent.action_queue

    def test_lower_priority_enqueues(self):
        """New action with lower priority is appended to the queue."""
        model = SimpleModel()
        agent = make_agent(model)

        current = Action("current", duration=10.0, priority=5.0)
        incoming = Action("incoming", duration=2.0, priority=2.0)
        agent.start_action(current)
        agent.request_action(incoming)

        assert agent.current_action is current
        assert incoming in agent.action_queue

    def test_non_interruptible_blocks_preemption(self):
        """A non-interruptible action cannot be preempted even by higher priority."""
        model = SimpleModel()
        agent = make_agent(model)

        critical = Action("critical", duration=10.0, priority=1.0, interruptible=False)
        urgent = Action("urgent", duration=2.0, priority=99.0)
        agent.start_action(critical)
        agent.request_action(urgent)

        assert agent.current_action is critical
        assert urgent in agent.action_queue

    def test_queued_actions_run_after_current_completes(self):
        """Actions enqueued via request_action execute after the current one finishes."""
        model = SimpleModel()
        agent = make_agent(model)
        cb, results = effect_tracker()

        current = Action("current", duration=3.0, priority=5.0, on_effect=cb)
        low = Action("low", duration=2.0, priority=1.0, on_effect=cb)
        agent.start_action(current)
        agent.request_action(low)  # enqueued (lower priority)

        model.run_for(3)  # current completes, low auto-starts
        assert agent.current_action is low
        model.run_for(2)  # low completes
        assert results == [1.0, 1.0]
        assert not agent.is_busy
