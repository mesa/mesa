"""Tests for mesa.experimental.actions."""

# ruff: noqa: D101, D102, D103, D107
import gc
import math
import pickle
import warnings
import weakref

import pytest

from mesa import Agent, Model
from mesa.experimental.actions import Action, ActionState, HasActions
from mesa.time import Priority

# --- Helpers ---


class ActionAgent(HasActions, Agent):
    """Agent with action support, as models combine it."""


class TrackedAction(Action):
    """Action subclass that records lifecycle events for testing."""

    def __init__(self, agent, duration=5.0, **kwargs):
        super().__init__(agent, duration=duration, **kwargs)
        self.start_count = 0
        self.resume_count = 0
        self.completed = False
        self.interrupted = False
        self.failed = False
        self.canceled = False
        self.interrupt_progress = None

    def on_start(self):
        self.start_count += 1

    def on_resume(self):
        self.resume_count += 1

    def on_complete(self):
        self.completed = True

    def on_interrupt(self):
        self.interrupted = True
        self.interrupt_progress = self.progress

    def on_fail(self):
        self.failed = True

    def on_cancel(self):
        self.canceled = True


def make_model_and_agent():
    model = Model()
    agent = ActionAgent(model)
    return model, agent


# --- Name property ---


class TestName:
    def test_subclass_name(self):
        _model, agent = make_model_and_agent()
        action = TrackedAction(agent)
        assert action.name == "TrackedAction"

    def test_base_class_name(self):
        _model, agent = make_model_and_agent()
        action = Action(agent)
        assert action.name == "Action"

    def test_name_in_repr(self):
        _model, agent = make_model_and_agent()
        action = TrackedAction(agent)
        assert "TrackedAction" in repr(action)

    def test_instance_name_override(self):
        """Instance attribute overrides the property."""
        _model, agent = make_model_and_agent()
        action = Action(agent)
        action.name = "custom_name"
        assert action.name == "custom_name"

    def test_name_via_init(self):
        """Name can be passed via __init__."""
        _model, agent = make_model_and_agent()
        action = Action(agent, name="my_action")
        assert action.name == "my_action"


# --- Basic lifecycle ---


class TestActionLifecycle:
    def test_action_starts_pending(self):
        _model, agent = make_model_and_agent()
        action = TrackedAction(agent)
        assert action.state is ActionState.PENDING
        assert action.progress == 0.0

    def test_start_action(self):
        _model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=5.0)

        agent.start_action(action)

        assert action.state is ActionState.ACTIVE
        assert action.start_count == 1
        assert action.resume_count == 0
        assert agent.current_action is action
        assert agent.is_busy

    def test_action_completes(self):
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=5.0)

        agent.start_action(action)
        model.run_for(5)

        assert action.state is ActionState.COMPLETED
        assert action.completed
        assert action.progress == 1.0
        assert agent.current_action is None
        assert not agent.is_busy

    def test_instantaneous_action(self):
        _model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=0)

        agent.start_action(action)

        assert action.state is ActionState.COMPLETED
        assert action.completed
        assert agent.current_action is None

    def test_on_start_fires_once(self):
        _model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=3.0)

        agent.start_action(action)
        assert action.start_count == 1

    def test_on_complete_fires(self):
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=3.0)

        agent.start_action(action)
        model.run_for(3)

        assert action.completed
        assert not action.interrupted


# --- Interruption ---


class TestInterruption:
    def test_interrupt_updates_progress(self):
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)

        agent.start_action(action)
        model.run_for(3)  # 30% done
        agent.interrupt_action()

        assert action.state is ActionState.INTERRUPTED
        assert action.interrupted
        assert action.interrupt_progress == pytest.approx(0.3)

    def test_interrupt_for_replaces_action(self):
        model, agent = make_model_and_agent()
        first = TrackedAction(agent, duration=10.0)
        second = TrackedAction(agent, duration=5.0)

        agent.start_action(first)
        model.run_for(4)  # 40% done with first
        result = agent.interrupt_for(second)

        assert result is True
        assert first.state is ActionState.INTERRUPTED
        assert first.interrupt_progress == pytest.approx(0.4)
        assert second.state is ActionState.ACTIVE
        assert agent.current_action is second

    def test_non_interruptible_blocks_interrupt(self):
        model, agent = make_model_and_agent()
        first = TrackedAction(agent, duration=10.0, interruptible=False)
        second = TrackedAction(agent, duration=5.0)

        agent.start_action(first)
        model.run_for(3)
        result = agent.interrupt_for(second)

        assert result is False
        assert first.state is ActionState.ACTIVE
        assert agent.current_action is first
        assert second.start_count == 0

    def test_non_interruptible_refuses_interrupt_action(self):
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0, interruptible=False)

        agent.start_action(action)
        model.run_for(5)

        assert agent.interrupt_action() is False
        assert action.state is ActionState.ACTIVE
        assert not action.interrupted

    def test_force_ignores_interruptible_flag(self):
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0, interruptible=False)

        agent.start_action(action)
        model.run_for(5)
        result = agent.interrupt_action(force=True)

        assert result is True
        assert action.state is ActionState.INTERRUPTED
        assert action.interrupt_progress == pytest.approx(0.5)
        assert action.is_resumable

    def test_interrupt_action_idle_returns_false(self):
        _model, agent = make_model_and_agent()
        assert agent.interrupt_action() is False

    def test_interrupt_idle_agent_just_starts(self):
        _model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=5.0)

        result = agent.interrupt_for(action)

        assert result is True
        assert action.state is ActionState.ACTIVE
        assert agent.current_action is action

    def test_interrupt_callback_receives_progress(self):
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=4.0)

        agent.start_action(action)
        model.run_for(1)  # 25%
        agent.interrupt_action()

        assert action.interrupt_progress == pytest.approx(0.25)


# --- on_start vs on_resume ---


class TestStartResume:
    def test_first_start_calls_on_start(self):
        _model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=5.0)

        agent.start_action(action)

        assert action.start_count == 1
        assert action.resume_count == 0

    def test_resume_calls_on_resume_not_on_start(self):
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)

        agent.start_action(action)
        model.run_for(3)
        agent.interrupt_action()

        action.resume()

        assert action.start_count == 1  # Not called again
        assert action.resume_count == 1

    def test_default_on_resume_calls_on_start(self):
        """When on_resume is not overridden, it delegates to on_start."""
        model, agent = make_model_and_agent()

        class StartTracker(Action):
            def __init__(self, agent):
                super().__init__(agent, duration=10.0)
                self.started = []

            def on_start(self):
                self.started.append("start")

            def on_interrupt(self):
                pass

        action = StartTracker(agent)

        agent.start_action(action)
        assert action.started == ["start"]

        model.run_for(3)
        agent.interrupt_action()

        action.resume()
        # Default on_resume calls on_start
        assert action.started == ["start", "start"]

    def test_overridden_on_resume_prevents_on_start(self):
        """When on_resume is overridden, on_start is not called on resume."""
        model, agent = make_model_and_agent()

        class ResumeTracker(Action):
            def __init__(self, agent):
                super().__init__(agent, duration=10.0)
                self.log = []

            def on_start(self):
                self.log.append("start")

            def on_resume(self):
                self.log.append("resume")

            def on_interrupt(self):
                pass

        action = ResumeTracker(agent)

        agent.start_action(action)
        model.run_for(3)
        agent.interrupt_action()

        action.resume()

        assert action.log == ["start", "resume"]

    def test_multiple_resume_cycles(self):
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)

        agent.start_action(action)
        model.run_for(2)
        agent.interrupt_action()

        action.resume()
        model.run_for(3)
        agent.interrupt_action()

        action.resume()
        model.run_for(5)

        assert action.start_count == 1
        assert action.resume_count == 2
        assert action.completed


class TestHookEndsAction:
    """A hook that ends its own action leaves nothing scheduled."""

    def test_interrupt_in_on_start_schedules_no_completion(self):
        done = []

        class InterruptOnStart(Action):
            def on_start(self):
                self.interrupt()

            def on_resume(self):
                pass

            def on_complete(self):
                done.append(self.model.time)

        model, agent = make_model_and_agent()
        action = InterruptOnStart(agent, duration=5.0)
        agent.start_action(action)
        assert action.state is ActionState.INTERRUPTED

        model.run_until(1.0)
        action.resume()
        model.run_until(10.0)

        # resumed at t=1 with 5 left, so it ends at t=6, not at the old t=5
        assert done == [6.0]

    def test_interrupt_in_on_resume_schedules_no_completion(self):
        done = []

        class InterruptOnResume(Action):
            interrupt_now = False

            def on_resume(self):
                if self.interrupt_now:
                    self.interrupt()

            def on_complete(self):
                done.append(self.model.time)

        model, agent = make_model_and_agent()
        action = InterruptOnResume(agent, duration=5.0)
        agent.start_action(action)
        model.run_until(1.0)
        action.interrupt()  # 4 left

        model.run_until(2.0)
        action.interrupt_now = True
        action.resume()
        assert action.state is ActionState.INTERRUPTED

        model.run_until(3.0)
        action.interrupt_now = False
        action.resume()
        model.run_until(10.0)

        assert done == [7.0]

    def test_interrupt_and_resume_in_on_start_leaves_one_completion(self):
        done = []

        class Bounce(Action):
            bounced = False

            def on_start(self):
                if not self.bounced:
                    self.bounced = True
                    self.interrupt()
                    self.resume()

            def on_complete(self):
                done.append(self.model.time)

        model, agent = make_model_and_agent()
        before = len(model._event_list)
        action = agent.start_action(Bounce(agent, duration=5.0))

        # One completion event, plus the wake the interrupt queued.
        assert len(model._event_list) == before + 2
        model.run_until(10.0)
        assert done == [5.0]
        assert action.state is ActionState.COMPLETED

    def test_instant_action_completes_even_if_on_start_cancels_it(self):
        """It is due the moment it starts, so completion wins.

        Refusing to start is what start_requirements are for.
        """
        _model, agent = make_model_and_agent()

        class Abort(TrackedAction):
            def on_start(self):
                super().on_start()
                self.cancel()

        action = agent.start_action(Abort(agent, duration=0))

        assert action.state is ActionState.COMPLETED
        assert action.completed
        assert not action.canceled


# --- Completion wins ---


class TestCompletionWins:
    """At its deadline an action completes, whatever else happens at that time."""

    def at_deadline(self, model, fn, time):
        """Run fn at `time`, before the action's own completion event.

        Returns the list its return value is appended to.
        """
        results = []

        def run():
            results.append(fn())

        self.keep_alive = run  # events hold their callable weakly
        model.schedule_event(run, at=time, priority=Priority.HIGH)
        return results

    def test_interrupt_at_the_deadline_completes(self):
        model, agent = make_model_and_agent()
        agent.energy = 0.0

        class Forage(TrackedAction):
            def on_progress(self, delta):
                self.agent.energy += 30 * delta

        action = agent.start_action(Forage(agent, duration=10.0))
        results = self.at_deadline(model, action.interrupt, 10.0)
        model.run_until(20.0)

        assert results == [False]
        assert action.state is ActionState.COMPLETED
        assert not action.interrupted
        assert agent.energy == 30
        with pytest.raises(ValueError, match="COMPLETED"):
            action.resume()

    def test_due_action_completes_even_if_not_interruptible(self):
        model, agent = make_model_and_agent()
        action = agent.start_action(
            TrackedAction(agent, duration=10.0, interruptible=False)
        )
        model.run_until(5.0)
        results = self.at_deadline(model, action.interrupt, 10.0)
        model.run_until(10.0)

        assert results == [False]
        assert action.completed

    def test_cancel_at_the_deadline_completes(self):
        model, agent = make_model_and_agent()
        action = agent.start_action(TrackedAction(agent, duration=10.0))
        results = self.at_deadline(model, action.cancel, 10.0)
        model.run_until(10.0)

        assert results == [False]
        assert action.completed
        assert not action.canceled

    def test_due_action_does_not_block_its_successor(self):
        model, agent = make_model_and_agent()
        current = agent.start_action(
            TrackedAction(agent, duration=10.0, interruptible=False)
        )
        successor = TrackedAction(agent, duration=2.0)
        results = self.at_deadline(model, lambda: agent.interrupt_for(successor), 10.0)
        model.run_until(10.0)

        assert results == [True]
        assert current.completed
        assert agent.current_action is successor

    def test_interrupt_for_the_due_current_action_completes_it(self):
        model, agent = make_model_and_agent()
        current = agent.start_action(TrackedAction(agent, duration=10.0))
        results = self.at_deadline(model, lambda: agent.interrupt_for(current), 10.0)
        model.run_until(10.0)

        assert results == [False]
        assert current.completed

    def test_progress_is_exactly_one_at_the_deadline(self):
        """Accumulated progress can fall short of 1.0; the deadline can't."""
        model, agent = make_model_and_agent()
        action = agent.start_action(TrackedAction(agent, duration=7.0))
        for _ in range(2):
            model.run_for(0.7)
            action.interrupt()
            action.resume()
        # Summing 0.1 + 0.1 + 0.8 gives 0.9999999999999999 here.
        seen = self.at_deadline(model, lambda: action.progress, 7.0)
        model.run_until(7.0)

        assert seen == [1.0]
        assert action.completed


# --- Live progress ---


class TestLiveProgress:
    def test_progress_live_during_active(self):
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)

        agent.start_action(action)

        model.run_for(3)
        assert action.progress == pytest.approx(0.3)

        model.run_for(2)
        assert action.progress == pytest.approx(0.5)

    def test_remaining_time_live(self):
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)

        agent.start_action(action)
        model.run_for(4)

        assert action.remaining_time == pytest.approx(6.0)

    def test_elapsed_time_live(self):
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)

        agent.start_action(action)
        model.run_for(4)

        assert action.elapsed_time == pytest.approx(4.0)

    def test_progress_frozen_after_interrupt(self):
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)

        agent.start_action(action)
        model.run_for(3)
        agent.interrupt_action()

        assert action.progress == pytest.approx(0.3)

        # Time passes but progress doesn't change
        model.run_for(5)
        assert action.progress == pytest.approx(0.3)

    def test_progress_live_after_resume(self):
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)

        agent.start_action(action)
        model.run_for(4)  # 40%
        agent.interrupt_action()

        action.resume()
        model.run_for(3)  # 40% + 30% = 70%

        assert action.progress == pytest.approx(0.7)


# --- Pause and resume ---


class TestPauseResume:
    def test_resume_continues_from_progress(self):
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)

        agent.start_action(action)
        model.run_for(3)  # 30% done
        agent.interrupt_action()

        assert action.progress == pytest.approx(0.3)
        assert action.remaining_time == pytest.approx(7.0)
        assert action.is_resumable

        action.resume()
        assert action.state is ActionState.ACTIVE

        model.run_for(7)
        assert action.state is ActionState.COMPLETED
        assert action.completed
        assert action.progress == 1.0

    def test_multiple_interrupt_resume_cycles(self):
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)

        # First attempt: 20%
        agent.start_action(action)
        model.run_for(2)
        agent.interrupt_action()
        assert action.progress == pytest.approx(0.2)

        # Second attempt: 20% + 30% = 50%
        action.resume()
        model.run_for(3)
        agent.interrupt_action()
        assert action.progress == pytest.approx(0.5)

        # Third attempt: 50% + 50% = done
        action.resume()
        model.run_for(5)
        assert action.state is ActionState.COMPLETED
        assert action.progress == 1.0

    def test_completed_action_not_resumable(self):
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=3.0)

        agent.start_action(action)
        model.run_for(3)

        assert not action.is_resumable
        with pytest.raises(ValueError, match="COMPLETED"):
            action.resume()

    def test_is_resumable_property(self):
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=5.0)

        assert not action.is_resumable  # PENDING

        agent.start_action(action)
        assert not action.is_resumable  # ACTIVE

        model.run_for(2)
        agent.interrupt_action()
        assert action.is_resumable  # INTERRUPTED with progress < 1

    def test_resume_respects_remaining_duration_only(self):
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)

        agent.start_action(action)
        model.run_for(8)  # 80%
        agent.interrupt_action()

        action.resume()

        # Should complete after 2 more time units, not 10
        model.run_for(2)
        assert action.state is ActionState.COMPLETED


# --- Error handling ---


class TestActionClearsAgentReference:
    """Verify the Action itself clears agent.current_action."""

    def test_complete_clears_current_action(self):
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=3.0)

        agent.start_action(action)
        model.run_for(3)

        assert agent.current_action is None

    def test_interrupt_clears_current_action(self):
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)

        agent.start_action(action)
        model.run_for(2)
        action.interrupt()

        assert agent.current_action is None

    def test_cancel_clears_current_action(self):
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0, interruptible=False)

        agent.start_action(action)
        model.run_for(2)
        action.cancel()

        assert agent.current_action is None

    def test_interrupt_pending_returns_false(self):
        _model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=5.0)

        assert action.interrupt() is False

    def test_interrupt_completed_returns_false(self):
        _model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=0)

        agent.start_action(action)  # completes instantly
        assert action.interrupt() is False

    def test_interrupt_already_interrupted_returns_false(self):
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)

        agent.start_action(action)
        model.run_for(3)
        action.interrupt()

        assert action.interrupt() is False

    def test_cancel_final_returns_false(self):
        _model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=0)
        agent.start_action(action)  # completes instantly

        assert action.cancel() is False
        assert action.state is ActionState.COMPLETED
        assert not action.canceled


class TestErrorHandling:
    def test_start_while_busy_raises(self):
        _model, agent = make_model_and_agent()
        first = TrackedAction(agent, duration=10.0)
        second = TrackedAction(agent, duration=5.0)

        agent.start_action(first)

        with pytest.raises(ValueError, match="already performing"):
            agent.start_action(second)

    def test_start_wrong_agent_raises(self):
        model, agent1 = make_model_and_agent()
        agent2 = ActionAgent(model)
        action = TrackedAction(agent1, duration=5.0)

        with pytest.raises(ValueError, match="does not match"):
            agent2.start_action(action)

    def test_start_completed_action_raises(self):
        _model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=0)

        agent.start_action(action)  # Completes immediately

        with pytest.raises(ValueError, match="COMPLETED"):
            agent.start_action(action)

    def test_negative_duration_raises(self):
        _model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=-1.0)

        with pytest.raises(ValueError, match="duration"):
            agent.start_action(action)

    def test_failing_duration_leaves_idle_agent_usable(self):
        _model, agent = make_model_and_agent()

        def bad_duration(a):
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            agent.start_action(Action(agent, duration=bad_duration))

        assert agent.current_action is None
        action = agent.start_action(TrackedAction(agent))
        assert agent.current_action is action

    def test_completed_action_does_not_occupy_slot(self):
        _model, agent = make_model_and_agent()
        action = Action(agent, duration=0)
        agent.start_action(action)

        with pytest.raises(ValueError, match="COMPLETED"):
            agent.start_action(action)

        assert agent.current_action is None

    def test_cancel_idle_returns_false(self):
        _model, agent = make_model_and_agent()
        assert agent.interrupt_action() is False

    def test_action_requires_has_actions_agent(self):
        """A plain Agent without the mixin is rejected up front."""
        model = Model()
        plain = Agent(model)

        with pytest.raises(TypeError, match="HasActions"):
            Action(plain)

    def test_mixin_combines_with_other_agent_bases(self):
        """Cooperative __init__ lets the mixin ride any Agent subclass."""
        from mesa.discrete_space import CellAgent, OrthogonalMooreGrid  # noqa: PLC0415

        class Grazer(HasActions, CellAgent):
            pass

        model = Model()
        grid = OrthogonalMooreGrid((3, 3), random=model.random)
        agent = Grazer(model)
        agent.cell = grid[(1, 1)]

        action = agent.start_action(TrackedAction(agent, duration=2.0))
        model.run_for(2)

        assert action.completed
        assert not agent.is_busy
        assert agent.cell is grid[(1, 1)]


# --- Callable duration and priority ---


class TestCallableDurationPriority:
    def test_callable_duration(self):
        model, agent = make_model_and_agent()
        agent.speed = 2.0
        action = TrackedAction(agent, duration=lambda a: 10.0 / a.speed)

        agent.start_action(action)
        assert action.duration == 5.0

        model.run_for(5)
        assert action.state is ActionState.COMPLETED

    def test_callable_priority(self):
        _model, agent = make_model_and_agent()
        agent.threat_level = 8.0
        action = TrackedAction(agent, duration=3.0, priority=lambda a: a.threat_level)

        agent.start_action(action)
        assert action.priority == 8.0

    def test_callable_duration_resolved_once(self):
        """Duration callable is resolved at first start, not on resume."""
        model, agent = make_model_and_agent()
        call_count = 0

        def get_duration(a):
            nonlocal call_count
            call_count += 1
            return 10.0

        action = TrackedAction(agent, duration=get_duration)

        agent.start_action(action)
        assert call_count == 1

        model.run_for(3)
        agent.interrupt_action()

        action.resume()  # Resume
        assert call_count == 1  # Not called again


# --- Subclass callbacks ---


class TestSubclassCallbacks:
    def test_subclass_on_complete(self):
        model, agent = make_model_and_agent()
        agent.energy = 50

        class GainEnergy(Action):
            def on_complete(self):
                self.agent.energy += 30

        action = GainEnergy(agent, duration=3.0)

        agent.start_action(action)
        model.run_for(3)

        assert agent.energy == 80

    def test_subclass_on_progress(self):
        model, agent = make_model_and_agent()

        class ProgressTracker(Action):
            def __init__(self, agent):
                super().__init__(agent, duration=10.0)
                self.received_progress = []

            def on_progress(self, delta):
                self.received_progress.append(delta)

        action = ProgressTracker(agent)

        agent.start_action(action)
        model.run_for(2)
        agent.interrupt_action()

        assert action.received_progress == [pytest.approx(0.2)]

    def test_subclass_on_start(self):
        _model, agent = make_model_and_agent()

        class StartTracker(Action):
            def __init__(self, agent):
                super().__init__(agent, duration=5.0)
                self.started = False

            def on_start(self):
                self.started = True

        action = StartTracker(agent)

        agent.start_action(action)
        assert action.started is True


# --- Agent removal ---


class TestAgentRemoval:
    def test_remove_cancels_the_active_action(self):
        model, agent = make_model_and_agent()
        action = agent.start_action(LoggedAction(agent, duration=10.0))
        model.run_for(3)
        agent.remove()

        assert action.log == [
            ("start", S.ACTIVE),
            ("progress", S.ACTIVE),
            ("stop", S.CANCELED),
            ("cancel", S.CANCELED),
        ]
        assert action.progress == pytest.approx(0.3)
        assert agent.current_action is None
        model.run_for(20)
        assert action.state is ActionState.CANCELED

    def test_remove_never_completes_a_due_action(self):
        model, agent = make_model_and_agent()
        action = agent.start_action(TrackedAction(agent, duration=10.0))

        def remove():
            agent.remove()

        model.schedule_event(remove, at=10.0, priority=Priority.HIGH)
        model.run_for(10)

        assert action.state is ActionState.CANCELED
        assert action.canceled
        assert not action.completed
        assert action.progress == 1.0

    def test_remove_cancels_interrupted_actions_in_start_order(self):
        _model, agent = make_model_and_agent()
        canceled = []

        class Named(Action):
            def on_cancel(self):
                canceled.append(self.name)

        first = agent.start_action(Named(agent, name="first"))
        first.interrupt()
        second = agent.start_action(Named(agent, name="second"))
        second.interrupt()
        agent.start_action(Named(agent, name="third"))
        first.cancel()  # already final, so removal leaves it alone
        canceled.clear()
        pending = Named(agent, name="pending")

        agent.remove()

        assert canceled == ["third", "second"]
        assert second.state is ActionState.CANCELED
        assert pending.state is ActionState.PENDING

    def test_dropped_interrupted_action_is_not_kept_alive(self):
        model, agent = make_model_and_agent()
        action = agent.start_action(TrackedAction(agent))
        action.interrupt()
        model.run_for(1)  # the wake hands the action to on_idle, then lets go
        ref = weakref.ref(action)
        del action
        gc.collect()

        assert ref() is None
        agent.remove()

    def test_no_wake_after_removal(self):
        model = Model()
        agent = IdleRecorder(model)
        agent.start_action(TrackedAction(agent, duration=5.0))
        model.run_for(1)
        agent.remove()
        model.run_for(10)

        assert agent.wakes == []

    def test_hooks_run_while_the_agent_is_in_the_model(self):
        model, agent = make_model_and_agent()
        seen = []

        class Look(Action):
            def on_cancel(self):
                seen.append(self.agent in self.model.agents)

        agent.start_action(Look(agent))
        agent.remove()

        assert seen == [True]
        assert agent not in model.agents

    @pytest.mark.parametrize(
        "call",
        [
            lambda agent, action: agent.start_action(Action(agent)),
            lambda agent, action: action.resume(),
            lambda agent, action: agent.interrupt_for(Action(agent)),
        ],
        ids=["start_action", "resume", "interrupt_for"],
    )
    def test_removed_agent_cannot_start_anything(self, call):
        _model, agent = make_model_and_agent()
        action = agent.start_action(TrackedAction(agent))
        action.interrupt()
        agent.remove()

        with pytest.raises(ValueError, match="was removed"):
            call(agent, action)

    def test_hook_that_removes_the_agent_again(self):
        model, agent = make_model_and_agent()
        calls = []

        class RemovesAgain(Action):
            def on_cancel(self):
                calls.append(self)
                self.agent.remove()

        agent.start_action(RemovesAgain(agent))
        agent.remove()

        assert len(calls) == 1
        assert agent not in model.agents

    def test_on_complete_may_remove_the_agent(self):
        model = Model()
        agent = IdleRecorder(model)

        class Last(Action):
            def on_complete(self):
                self.agent.remove()

        action = agent.start_action(Last(agent, duration=2.0))
        model.run_for(5)

        assert action.state is ActionState.COMPLETED
        assert agent.wakes == []

    def test_reservation_held_across_an_interruption_is_released(self):
        """The Shift example: removed mid-break, the reservation still goes."""
        model, agent = make_model_and_agent()
        station = {"reserved_by": None}

        class Shift(Action):
            def on_start(self):
                station["reserved_by"] = self.agent

            def on_cancel(self):
                station["reserved_by"] = None

        shift = agent.start_action(Shift(agent, duration=8.0))
        model.run_for(2)
        shift.interrupt()  # on a break, still holding the station
        agent.remove()

        assert station["reserved_by"] is None

    def test_remove_all_agents_cancels_their_actions(self):
        model = Model()
        agents = [ActionAgent(model) for _ in range(3)]
        actions = [a.start_action(TrackedAction(a)) for a in agents]

        model.remove_all_agents()

        assert all(action.canceled for action in actions)

    def test_pickle_keeps_tracking(self):
        model, agent = make_model_and_agent()
        interrupted = agent.start_action(TrackedAction(agent, name="kept"))
        interrupted.interrupt()
        agent.kept = interrupted  # referenced, so it survives the round trip

        model2 = pickle.loads(pickle.dumps(model))  # noqa: S301
        agent2 = next(iter(model2.agents))
        agent2.remove()

        assert agent2.kept.state is ActionState.CANCELED
        assert agent2.kept.canceled
        assert interrupted.state is ActionState.INTERRUPTED

    def test_remove_with_explicit_cleanup(self):
        """An override that cancels first, as before, still works."""
        model = Model()

        class CleanAgent(ActionAgent):
            def remove(self):
                self.cancel_action()
                super().remove()

        agent = CleanAgent(model)
        action = TrackedAction(agent, duration=10.0)

        agent.start_action(action)
        model.run_for(3)
        agent.remove()

        assert action.canceled
        assert action.state is ActionState.CANCELED
        assert action.progress == pytest.approx(0.3)


class TestResumeDetection:
    """Verify on_start can distinguish first start from resume."""

    def test_on_start_can_detect_resume(self):
        model, agent = make_model_and_agent()

        class DetectResumeAction(Action):
            def __init__(self, agent):
                super().__init__(agent, duration=10.0)
                self.start_types = []

            def on_start(self):
                self.start_types.append("resume" if self.progress > 0 else "first")

            def on_interrupt(self):
                pass

        action = DetectResumeAction(agent)

        agent.start_action(action)
        model.run_for(3)
        agent.interrupt_action()

        action.resume()
        model.run_for(7)

        assert action.start_types == ["first", "resume"]


class TestEdgeCases:
    def test_double_completion_ignored(self):
        """Calling _do_complete twice doesn't fire on_complete twice."""
        model, agent = make_model_and_agent()

        class CompletionCounter(Action):
            def __init__(self, agent):
                super().__init__(agent, duration=3.0)
                self.complete_count = 0

            def on_complete(self):
                self.complete_count += 1

        action = CompletionCounter(agent)

        agent.start_action(action)
        model.run_for(3)

        # Manually try to complete again
        action._do_complete()

        assert action.complete_count == 1

    def test_remaining_time_before_start(self):
        _model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)

        # Before start, duration hasn't been resolved yet
        assert action.remaining_time == 0.0

    def test_elapsed_time_before_start(self):
        _model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)

        assert action.elapsed_time == 0.0

    def test_repr(self):
        _model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=5.0)

        assert "PENDING" in repr(action)
        assert "0%" in repr(action)

        agent.start_action(action)
        assert "ACTIVE" in repr(action)


# --- Requirements and failure ---


class TestRequirements:
    def test_no_requirements_by_default(self):
        _model, agent = make_model_and_agent()
        assert Action(agent).start_requirements == []

    def test_single_callable_is_wrapped(self):
        _model, agent = make_model_and_agent()
        action = Action(agent, start_requirements=lambda a: True)
        assert len(action.start_requirements) == 1

    def test_iterable_is_copied(self):
        """The action keeps its own list, not the caller's."""
        _model, agent = make_model_and_agent()
        shared = [lambda a: True]
        action = Action(agent, start_requirements=shared)
        shared.append(lambda a: False)
        assert len(action.start_requirements) == 1

    def test_subclass_can_assign_requirements(self):
        _model, agent = make_model_and_agent()

        class Picky(Action):
            def __init__(self, agent):
                super().__init__(agent)
                self.start_requirements = [lambda a: a.ready]

        agent.ready = False
        action = agent.start_action(Picky(agent))
        assert action.has_failed

    def test_requirement_receives_the_agent(self):
        _model, agent = make_model_and_agent()
        seen = []

        def record(a):
            seen.append(a)
            return True

        agent.start_action(Action(agent, start_requirements=record))
        assert seen == [agent]

    def test_all_requirements_must_hold(self):
        _model, agent = make_model_and_agent()
        action = TrackedAction(
            agent, start_requirements=[lambda a: True, lambda a: False]
        )

        agent.start_action(action)

        assert action.has_failed

    def test_action_starts_when_every_requirement_holds(self):
        _model, agent = make_model_and_agent()
        action = TrackedAction(
            agent, start_requirements=[lambda a: True, lambda a: True]
        )

        agent.start_action(action)

        assert action.state is ActionState.ACTIVE
        assert action.start_count == 1

    def test_requirement_is_not_checked_mid_flight(self):
        """Broken and repaired between start and completion is not a failure."""
        model, agent = make_model_and_agent()
        agent.grass = True
        action = TrackedAction(
            agent, duration=5.0, start_requirements=lambda a: a.grass
        )
        agent.start_action(action)

        model.run_for(2)
        agent.grass = False
        model.run_for(1)
        agent.grass = True
        model.run_for(3)

        assert action.state is ActionState.COMPLETED
        assert action.completed


class TestActionFailure:
    def test_failing_requirement_blocks_start(self):
        _model, agent = make_model_and_agent()
        action = TrackedAction(agent, start_requirements=lambda a: False)

        agent.start_action(action)

        assert action.state is ActionState.FAILED
        assert action.failed
        assert action.start_count == 0
        assert action.progress == 0.0
        assert not agent.is_busy

    def test_start_failure_schedules_no_completion(self):
        model, agent = make_model_and_agent()
        before = len(model._event_list)

        action = TrackedAction(agent, start_requirements=lambda a: False)
        agent.start_action(action)

        # Exactly one new event: the agent's idle wake, never a completion.
        assert len(model._event_list) == before + 1
        model.run_for(10)
        assert action.start_count == 0
        assert not action.completed

    def test_failed_action_cannot_be_restarted(self):
        _model, agent = make_model_and_agent()
        action = TrackedAction(agent, start_requirements=lambda a: False)
        agent.start_action(action)

        with pytest.raises(ValueError, match="FAILED"):
            agent.start_action(action)

    def test_start_requirements_rechecked_on_resume(self):
        model, agent = make_model_and_agent()
        agent.safe = True
        action = TrackedAction(agent, duration=5.0, start_requirements=lambda a: a.safe)
        agent.start_action(action)

        model.run_for(2)
        agent.interrupt_action()
        agent.safe = False

        action.resume()

        assert action.state is ActionState.FAILED
        assert action.resume_count == 0
        assert action.progress == pytest.approx(0.4)

    def test_requirement_broken_at_completion_fails(self):
        model, agent = make_model_and_agent()
        agent.grass = True
        action = TrackedAction(
            agent, duration=3.0, completion_requirements=lambda a: a.grass
        )
        agent.start_action(action)

        agent.grass = False
        model.run_for(4)

        assert action.state is ActionState.FAILED
        assert action.failed
        assert not action.completed
        assert action.progress == 1.0  # the duration elapsed, the effect did not apply
        assert agent.current_action is None

    def test_instantaneous_action_checks_both_lists(self):
        """duration=0 completes inside start(), so both gates run in one call."""
        _model, agent = make_model_and_agent()
        agent.ready = True

        action = TrackedAction(
            agent,
            duration=0.0,
            start_requirements=lambda a: a.ready,
            completion_requirements=lambda a: False,
        )
        agent.start_action(action)

        assert action.state is ActionState.FAILED
        assert action.start_count == 1  # it did start, then failed to land


class TestInterruptForWithRequirements:
    def test_returns_false_when_the_new_action_fails(self):
        _model, agent = make_model_and_agent()
        agent.start_action(TrackedAction(agent, duration=5.0))

        assert (
            agent.interrupt_for(
                TrackedAction(agent, start_requirements=lambda a: False)
            )
            is False
        )

    def test_old_action_is_not_rolled_back(self):
        model, agent = make_model_and_agent()
        first = TrackedAction(agent, duration=5.0)
        agent.start_action(first)

        model.run_for(2)
        agent.interrupt_for(TrackedAction(agent, start_requirements=lambda a: False))

        assert first.state is ActionState.INTERRUPTED
        assert agent.current_action is None

    def test_returns_true_when_the_new_action_starts(self):
        _model, agent = make_model_and_agent()
        agent.start_action(TrackedAction(agent, duration=5.0))

        assert agent.interrupt_for(
            TrackedAction(agent, start_requirements=lambda a: True)
        )

    def test_refused_interruption_leaves_the_new_action_untouched(self):
        _model, agent = make_model_and_agent()
        agent.start_action(TrackedAction(agent, duration=5.0, interruptible=False))
        replacement = TrackedAction(agent, start_requirements=lambda a: False)

        assert agent.interrupt_for(replacement) is False
        assert replacement.state is ActionState.PENDING
        assert not replacement.failed


# --- Vocabulary: cancel is final, resume continues ---


class TestCancel:
    def test_cancel_active_is_final(self):
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)
        agent.start_action(action)
        model.run_for(3)

        assert agent.cancel_action() is True

        assert action.state is ActionState.CANCELED
        assert action.canceled
        assert not action.interrupted
        assert action.progress == pytest.approx(0.3)
        assert agent.current_action is None
        assert not action.is_resumable
        with pytest.raises(ValueError, match="CANCELED"):
            action.resume()

        model.run_for(20)
        assert not action.completed

    def test_cancel_ignores_interruptible_flag(self):
        _model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0, interruptible=False)
        agent.start_action(action)

        assert action.cancel() is True
        assert action.state is ActionState.CANCELED

    def test_cancel_interrupted(self):
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)
        agent.start_action(action)
        model.run_for(2)
        action.interrupt()

        assert action.cancel() is True
        assert action.state is ActionState.CANCELED
        assert action.canceled
        assert action.progress == pytest.approx(0.2)

    def test_cancel_pending(self):
        _model, agent = make_model_and_agent()
        action = TrackedAction(agent)

        assert action.cancel() is True
        assert action.state is ActionState.CANCELED
        assert action.canceled
        with pytest.raises(ValueError, match="CANCELED"):
            agent.start_action(action)

    def test_cancel_twice_fires_once(self):
        _model, agent = make_model_and_agent()
        calls = []

        class Counting(Action):
            def on_cancel(self):
                calls.append(self)

        action = agent.start_action(Counting(agent))
        assert action.cancel() is True
        assert action.cancel() is False
        assert len(calls) == 1

    def test_cancel_action_idle_returns_false(self):
        _model, agent = make_model_and_agent()
        assert agent.cancel_action() is False


class TestResume:
    def test_resume_active_returns_false(self):
        _model, agent = make_model_and_agent()
        action = agent.start_action(TrackedAction(agent))

        assert action.resume() is False
        assert action.resume_count == 0

    def test_resume_pending_raises(self):
        _model, agent = make_model_and_agent()
        with pytest.raises(ValueError, match="start_action"):
            TrackedAction(agent).resume()

    def test_resume_while_busy_raises(self):
        _model, agent = make_model_and_agent()
        first = agent.start_action(TrackedAction(agent))
        first.interrupt()
        agent.start_action(TrackedAction(agent))

        with pytest.raises(ValueError, match="already performing"):
            first.resume()
        assert first.state is ActionState.INTERRUPTED

    def test_resume_returns_false_on_failed_requirement(self):
        _model, agent = make_model_and_agent()
        agent.safe = True
        action = agent.start_action(
            TrackedAction(agent, start_requirements=lambda a: a.safe)
        )
        action.interrupt()
        agent.safe = False

        assert action.resume() is False
        assert action.has_failed

    def test_start_action_on_interrupted_points_to_resume(self):
        _model, agent = make_model_and_agent()
        action = agent.start_action(TrackedAction(agent))
        action.interrupt()

        with pytest.raises(ValueError, match=r"action\.resume\(\)"):
            agent.start_action(action)


class TestReadOnlyState:
    def test_state_cannot_be_assigned(self):
        _model, agent = make_model_and_agent()
        action = TrackedAction(agent)
        with pytest.raises(AttributeError):
            action.state = ActionState.COMPLETED

    def test_current_action_cannot_be_assigned(self):
        _model, agent = make_model_and_agent()
        with pytest.raises(AttributeError):
            agent.current_action = TrackedAction(agent)


class TestResolvedValues:
    @pytest.mark.parametrize("duration", [math.nan, math.inf, -1.0])
    def test_invalid_duration_raises_before_anything_changes(self, duration):
        _model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=duration)

        with pytest.raises(ValueError, match="duration"):
            agent.start_action(action)
        assert action.state is ActionState.PENDING
        assert action.start_count == 0
        assert not agent.is_busy

    def test_nan_priority_raises(self):
        _model, agent = make_model_and_agent()
        action = TrackedAction(agent, priority=lambda a: math.nan)

        with pytest.raises(ValueError, match="NaN"):
            agent.start_action(action)
        assert action.state is ActionState.PENDING

    def test_infinite_priority_always_preempts(self):
        _model, agent = make_model_and_agent()
        agent.start_action(TrackedAction(agent, duration=10.0, priority=1e300))

        assert agent.interrupt_for(TrackedAction(agent, priority=math.inf))


class TestInterruptForSlot:
    """interrupt_for checks what it can up front, then asks: is the slot free?"""

    @pytest.mark.parametrize("kind", ["other agent", "final"])
    def test_invalid_new_action_raises_before_interrupting(self, kind):
        model, agent = make_model_and_agent()
        current = agent.start_action(TrackedAction(agent, duration=5.0))
        if kind == "other agent":
            new_action = TrackedAction(ActionAgent(model))
        else:
            new_action = TrackedAction(agent)
            new_action.cancel()

        with pytest.raises(ValueError):
            agent.interrupt_for(new_action)

        assert current.state is ActionState.ACTIVE
        assert not current.interrupted
        model.run_for(5)
        assert current.completed

    def test_raises_even_when_the_policy_would_refuse(self):
        """Ownership is checked first, so a wrong call is never a quiet False."""
        model, agent = make_model_and_agent()
        agent.start_action(TrackedAction(agent, duration=5.0, interruptible=False))

        with pytest.raises(ValueError, match="does not match"):
            agent.interrupt_for(TrackedAction(ActionAgent(model)))

    def test_current_action_is_a_no_op(self):
        model, agent = make_model_and_agent()
        current = agent.start_action(TrackedAction(agent, duration=5.0))
        model.run_for(2)

        assert agent.interrupt_for(current) is True
        assert agent.current_action is current
        assert not current.interrupted
        assert current.start_count == 1
        model.run_for(3)
        assert current.completed

    def test_resumes_an_interrupted_action(self):
        model, agent = make_model_and_agent()
        task = agent.start_action(TrackedAction(agent, duration=10.0))
        model.run_for(4)
        meeting = TrackedAction(agent, duration=3.0)
        agent.interrupt_for(meeting)

        assert agent.interrupt_for(task) is True
        assert meeting.state is ActionState.INTERRUPTED
        assert task.state is ActionState.ACTIVE
        assert task.resume_count == 1
        assert task.remaining_time == pytest.approx(6.0)

    def test_hook_that_fills_the_slot_wins(self):
        _model, agent = make_model_and_agent()
        fallback = TrackedAction(agent, duration=3.0)

        class GoesElsewhere(TrackedAction):
            def on_interrupt(self):
                self.agent.start_action(fallback)

        agent.start_action(GoesElsewhere(agent, duration=10.0))
        new_action = TrackedAction(agent)

        assert agent.interrupt_for(new_action) is False
        assert agent.current_action is fallback
        assert new_action.state is ActionState.PENDING

    def test_hook_that_starts_the_new_action_counts(self):
        _model, agent = make_model_and_agent()
        new_action = TrackedAction(agent)

        class HandsOver(TrackedAction):
            def on_interrupt(self):
                self.agent.start_action(new_action)

        agent.start_action(HandsOver(agent, duration=10.0))

        assert agent.interrupt_for(new_action) is True
        assert agent.current_action is new_action
        assert new_action.start_count == 1

    def test_hook_that_cancels_the_new_action(self):
        _model, agent = make_model_and_agent()
        new_action = TrackedAction(agent)

        class Vetoes(TrackedAction):
            def on_interrupt(self):
                new_action.cancel()

        agent.start_action(Vetoes(agent, duration=10.0))

        assert agent.interrupt_for(new_action) is False
        assert new_action.state is ActionState.CANCELED
        assert not agent.is_busy


# --- Hook contract ---


class LoggedAction(Action):
    """Action that logs every hook with the state it saw."""

    def __init__(self, agent, duration=5.0, **kwargs):
        super().__init__(agent, duration=duration, **kwargs)
        self.log = []

    def _record(self, hook):
        self.log.append((hook, self.state))

    def on_start(self):
        self._record("start")

    def on_resume(self):
        self._record("resume")

    def on_progress(self, delta):
        self._record("progress")

    def on_stop(self):
        self._record("stop")

    def on_interrupt(self):
        self._record("interrupt")

    def on_complete(self):
        self._record("complete")

    def on_fail(self):
        self._record("fail")

    def on_cancel(self):
        self._record("cancel")


S = ActionState


class TestHookSequences:
    """Each exit fires its hooks in a fixed order, each seeing a fixed state."""

    def test_interrupt(self):
        model, agent = make_model_and_agent()
        action = agent.start_action(LoggedAction(agent))
        model.run_for(1)
        action.interrupt()

        assert action.log == [
            ("start", S.ACTIVE),
            ("progress", S.ACTIVE),
            ("stop", S.INTERRUPTED),
            ("interrupt", S.INTERRUPTED),
        ]

    def test_resume_then_complete(self):
        model, agent = make_model_and_agent()
        action = agent.start_action(LoggedAction(agent))
        model.run_for(1)
        action.interrupt()
        action.log.clear()
        action.resume()
        model.run_for(5)

        assert action.log == [
            ("resume", S.ACTIVE),
            ("progress", S.ACTIVE),
            ("stop", S.COMPLETED),
            ("complete", S.COMPLETED),
        ]

    def test_cancel_active(self):
        model, agent = make_model_and_agent()
        action = agent.start_action(LoggedAction(agent))
        model.run_for(1)
        action.cancel()

        assert action.log == [
            ("start", S.ACTIVE),
            ("progress", S.ACTIVE),
            ("stop", S.CANCELED),
            ("cancel", S.CANCELED),
        ]

    @pytest.mark.parametrize("from_state", ["pending", "interrupted"])
    def test_cancel_without_an_attempt_fires_only_on_cancel(self, from_state):
        model, agent = make_model_and_agent()
        action = LoggedAction(agent)
        if from_state == "interrupted":
            agent.start_action(action)
            model.run_for(1)
            action.interrupt()
            action.log.clear()
        action.cancel()

        assert action.log == [("cancel", S.CANCELED)]

    def test_completion_requirement_fails(self):
        model, agent = make_model_and_agent()
        action = agent.start_action(
            LoggedAction(agent, completion_requirements=lambda a: False)
        )
        model.run_for(5)

        assert action.log == [
            ("start", S.ACTIVE),
            ("progress", S.ACTIVE),
            ("stop", S.FAILED),
            ("fail", S.FAILED),
        ]

    def test_start_requirement_fails(self):
        _model, agent = make_model_and_agent()
        action = agent.start_action(
            LoggedAction(agent, start_requirements=lambda a: False)
        )

        assert action.log == [("fail", S.FAILED)]

    def test_instant_action(self):
        _model, agent = make_model_and_agent()
        action = agent.start_action(LoggedAction(agent, duration=0))

        assert action.log == [
            ("start", S.ACTIVE),
            ("progress", S.ACTIVE),
            ("stop", S.COMPLETED),
            ("complete", S.COMPLETED),
        ]

    def test_no_progress_hook_without_work(self):
        """An attempt interrupted the instant it began did no work."""
        _model, agent = make_model_and_agent()
        action = agent.start_action(LoggedAction(agent))
        action.interrupt()

        assert action.log == [
            ("start", S.ACTIVE),
            ("stop", S.INTERRUPTED),
            ("interrupt", S.INTERRUPTED),
        ]


class TestProgressReporting:
    def test_deltas_add_up_to_one(self):
        model, agent = make_model_and_agent()
        deltas = []

        class Recorder(Action):
            def on_progress(self, delta):
                deltas.append(delta)

        action = agent.start_action(Recorder(agent, duration=7.0))
        for _ in range(3):
            model.run_for(0.7)
            action.interrupt()
            action.resume()
        model.run_for(10)

        assert len(deltas) == 4
        assert math.fsum(deltas) == 1.0

    def test_interrupt_and_resume_at_the_same_time_changes_nothing(self):
        """A pause of zero length pays the same as no pause, to the last bit."""

        class Forage(Action):
            def on_progress(self, delta):
                self.agent.energy += 30 * delta

        def run(pause_at):
            model, agent = make_model_and_agent()
            agent.energy = 0.0
            action = agent.start_action(Forage(agent, duration=7.0))
            for t in pause_at:
                model.run_until(t)
                action.interrupt()
                action.resume()
            model.run_until(10.0)
            return agent.energy, action.progress

        assert run([]) == run([2.1]) == (30.0, 1.0)
        assert run([0.7, 1.4]) == run([0.7, 0.7, 1.4, 1.4])

    def test_progress_is_paid_before_the_completion_check(self):
        """A requirement sees all the work, however often it was interrupted."""
        model, agent = make_model_and_agent()
        agent.stock = 0.0

        class Fill(Action):
            def __init__(self, agent):
                super().__init__(
                    agent,
                    duration=4.0,
                    completion_requirements=lambda a: a.stock == pytest.approx(10),
                )

            def on_progress(self, delta):
                self.agent.stock += 10 * delta

        action = agent.start_action(Fill(agent))
        model.run_for(1)
        action.interrupt()
        action.resume()
        model.run_for(5)

        assert action.state is ActionState.COMPLETED

    def test_on_stop_releases_after_the_completion_check(self):
        """The machine example: the requirement still sees the machine held."""
        model = Model()
        worker = ActionAgent(model)
        worker.parts = 0
        machine = {"holder": None}

        class UseMachine(Action):
            def __init__(self, worker):
                super().__init__(
                    worker,
                    duration=8,
                    start_requirements=lambda a: machine["holder"] is None,
                    completion_requirements=lambda a: machine["holder"] is a,
                )

            def on_start(self):
                machine["holder"] = self.agent

            def on_stop(self):
                machine["holder"] = None

            def on_complete(self):
                self.agent.parts += 1

        action = worker.start_action(UseMachine(worker))
        model.run_for(3)
        action.interrupt()
        assert machine["holder"] is None
        action.resume()
        model.run_for(5)

        assert action.state is ActionState.COMPLETED
        assert worker.parts == 1
        assert machine["holder"] is None


GUARDED_CALLS = {
    "cancel_action": lambda agent: agent.cancel_action(),
    "start_action": lambda agent: agent.start_action(Action(agent)),
    "remove": lambda agent: agent.remove(),
}


class TestGuards:
    """Lifecycle calls are refused where they would cut a transition in half."""

    @pytest.mark.parametrize("call", GUARDED_CALLS)
    @pytest.mark.parametrize("place", ["on_progress", "on_stop"])
    def test_refused_in_hooks(self, place, call):
        model, agent = make_model_and_agent()
        errors = []

        class Meddler(Action):
            pass

        def meddle(self, *args):
            try:
                GUARDED_CALLS[call](self.agent)
            except RuntimeError as error:
                errors.append(error)

        setattr(Meddler, place, meddle)
        action = agent.start_action(Meddler(agent, duration=2.0))
        model.run_for(1)
        action.interrupt()

        assert len(errors) == 1
        assert place in str(errors[0])
        assert action.state is ActionState.INTERRUPTED

    @pytest.mark.parametrize(
        "where",
        ["start_requirements", "completion_requirements", "duration", "priority"],
    )
    def test_refused_in_decision_callables(self, where):
        model, agent = make_model_and_agent()

        def meddle(a):
            a.cancel_action()
            return 1.0

        action = Action(agent, **{where: meddle})
        with pytest.raises(RuntimeError, match="Can't start, interrupt"):
            agent.start_action(action)
            model.run_for(2)

    def test_refused_in_should_interrupt(self):
        model = Model()

        class Meddler(ActionAgent):
            def should_interrupt(self, current, incoming):
                current.interrupt()
                return True

        agent = Meddler(model)
        current = agent.start_action(Action(agent, duration=5.0))

        with pytest.raises(RuntimeError, match="should_interrupt"):
            agent.interrupt_for(Action(agent))
        assert current.state is ActionState.ACTIVE

    def test_other_agents_are_not_guarded(self):
        model = Model()
        other = ActionAgent(model)
        theirs = other.start_action(Action(other, duration=5.0))

        class Notifier(Action):
            def on_stop(self):
                theirs.interrupt()

        agent = ActionAgent(model)
        agent.start_action(Notifier(agent, duration=1.0))
        model.run_for(1)

        assert theirs.state is ActionState.INTERRUPTED

    def test_guard_is_lifted_after_the_hook(self):
        model, agent = make_model_and_agent()

        class Fails(Action):
            def on_progress(self, delta):
                raise KeyError("boom")

        agent.start_action(Fails(agent, duration=2.0))
        model.run_for(1)
        with pytest.raises(KeyError):
            agent.interrupt_action()

        assert agent._guarded_in is None


# --- Preemption policy ---


class TestShouldInterrupt:
    def test_higher_priority_preempts(self):
        model, agent = make_model_and_agent()
        low = TrackedAction(agent, duration=10.0, priority=1.0)
        high = TrackedAction(agent, duration=5.0, priority=2.0)

        agent.start_action(low)
        model.run_for(2)

        assert agent.interrupt_for(high) is True
        assert low.state is ActionState.INTERRUPTED
        assert agent.current_action is high

    def test_lower_priority_is_refused(self):
        model, agent = make_model_and_agent()
        high = TrackedAction(agent, duration=10.0, priority=2.0)
        low = TrackedAction(agent, duration=5.0, priority=1.0)

        agent.start_action(high)
        model.run_for(2)

        assert agent.interrupt_for(low) is False
        assert high.state is ActionState.ACTIVE
        assert agent.current_action is high
        assert low.state is ActionState.PENDING
        assert low.start_count == 0

    def test_equal_priority_preempts(self):
        """>= keeps the pre-priority behavior: both default to 0.0."""
        _model, agent = make_model_and_agent()
        first = TrackedAction(agent, duration=10.0)
        second = TrackedAction(agent, duration=5.0)

        agent.start_action(first)

        assert agent.interrupt_for(second) is True

    def test_callable_priority_resolved_before_decision(self):
        _model, agent = make_model_and_agent()
        agent.threat = 5.0
        current = TrackedAction(agent, duration=10.0, priority=3.0)
        incoming = TrackedAction(agent, duration=2.0, priority=lambda a: a.threat)

        agent.start_action(current)

        assert agent.interrupt_for(incoming) is True
        assert incoming.priority == 5.0

    def test_callable_priority_resolved_once(self):
        _model, agent = make_model_and_agent()
        calls = []

        def prio(a):
            calls.append(a)
            return 1.0

        agent.start_action(TrackedAction(agent, duration=10.0))
        agent.interrupt_for(TrackedAction(agent, duration=5.0, priority=prio))

        assert len(calls) == 1

    def test_non_interruptible_refused_despite_priority(self):
        _model, agent = make_model_and_agent()
        agent.start_action(
            TrackedAction(agent, duration=10.0, priority=1.0, interruptible=False)
        )

        assert agent.interrupt_for(TrackedAction(agent, priority=100.0)) is False

    def test_not_consulted_when_idle(self):
        _model, agent = make_model_and_agent()
        consulted = []

        class Watcher(ActionAgent):
            def should_interrupt(self, current, incoming):
                consulted.append((current, incoming))
                return super().should_interrupt(current, incoming)

        watcher = Watcher(agent.model)
        action = TrackedAction(watcher, duration=5.0)

        assert watcher.interrupt_for(action) is True
        assert consulted == []
        assert watcher.current_action is action

    def test_override_encodes_custom_policy(self):
        """A subclass can ignore priorities entirely."""
        model = Model()

        class Stubborn(ActionAgent):
            def should_interrupt(self, current, incoming):
                return incoming.name == "Flee"

        agent = Stubborn(model)
        agent.start_action(TrackedAction(agent, duration=10.0))

        assert agent.interrupt_for(TrackedAction(agent, priority=100.0)) is False
        assert agent.interrupt_for(TrackedAction(agent, name="Flee")) is True

    def test_override_cannot_force_non_interruptible(self):
        """Returning True attempts the interruption; the flag still refuses."""
        model = Model()

        class Pushy(ActionAgent):
            def should_interrupt(self, current, incoming):
                return True

        agent = Pushy(model)
        shielded = TrackedAction(agent, duration=10.0, interruptible=False)
        agent.start_action(shielded)

        assert agent.interrupt_for(TrackedAction(agent, priority=100.0)) is False
        assert shielded.state is ActionState.ACTIVE


# --- Integration: realistic scenarios ---


class TestRealisticScenarios:
    def test_sheep_forage_flee_resume(self):
        """Sheep forages, flees from predator, resumes foraging."""
        model = Model()
        sheep = ActionAgent(model)
        sheep.energy = 50.0
        sheep.alive = True

        class Forage(Action):
            def __init__(self, sheep):
                super().__init__(sheep, duration=5.0)

            def on_progress(self, delta):
                self.agent.energy += 30 * delta

        class Flee(Action):
            def __init__(self, sheep):
                super().__init__(sheep, duration=2.0, interruptible=False)

            def on_complete(self):
                pass  # survived

            def on_interrupt(self):
                self.agent.alive = False

        # Start foraging
        forage = Forage(sheep)
        sheep.start_action(forage)
        model.run_for(3)  # 60% done

        # Predator appears — interrupt and flee
        flee = Flee(sheep)
        result = sheep.interrupt_for(flee)

        assert result is True
        assert sheep.energy == pytest.approx(50.0 + 30 * 0.6)

        # Flee completes
        model.run_for(2)
        assert flee.state is ActionState.COMPLETED
        assert sheep.alive
        assert not sheep.is_busy

        # Resume foraging (remaining 40%)
        forage.resume()
        model.run_for(2)  # 40% of 5.0 = 2.0 time units

        assert forage.state is ActionState.COMPLETED
        # 18 for the first 60%, 12 for the rest: one forage, paid once.
        assert sheep.energy == pytest.approx(50.0 + 30.0)

    def test_sequential_actions(self):
        """Agent performs multiple actions in sequence."""
        model = Model()
        agent = ActionAgent(model)
        agent.log = []

        class LogAction(Action):
            def __init__(self, agent, label):
                super().__init__(agent, duration=2.0)
                self.label = label

            def on_complete(self):
                self.agent.log.append(f"done_{self.label}")

        for i in range(3):
            action = LogAction(agent, i)
            agent.start_action(action)
            model.run_for(2)

        assert agent.log == ["done_0", "done_1", "done_2"]

    def test_flee_non_interruptible_protects(self):
        """A fleeing agent can't be interrupted."""
        model = Model()
        agent = ActionAgent(model)

        flee = Action(agent, duration=3.0, interruptible=False)
        distraction = TrackedAction(agent, duration=1.0)

        agent.start_action(flee)
        model.run_for(1)

        result = agent.interrupt_for(distraction)

        assert result is False
        assert agent.current_action is flee
        assert distraction.start_count == 0

    def test_worker_interrupted_resumes_task(self):
        """Worker on a task, interrupted by meeting, resumes task."""
        model = Model()
        worker = ActionAgent(model)
        worker.log = []

        class Task(Action):
            def on_start(self):
                self.agent.log.append(f"start@{self.agent.model.time}")

            def on_resume(self):
                self.agent.log.append(f"resume@{self.agent.model.time}")

            def on_complete(self):
                self.agent.log.append(f"done@{self.agent.model.time}")

            def on_interrupt(self):
                self.agent.log.append(
                    f"interrupted@{self.agent.model.time}({self.progress:.0%})"
                )

        task = Task(worker, duration=10.0)
        meeting = TrackedAction(worker, duration=3.0)

        # Work on task
        worker.start_action(task)
        model.run_for(4)  # 40%

        # Meeting interrupts
        worker.interrupt_for(meeting)
        model.run_for(3)  # Meeting done

        # Resume task
        task.resume()
        model.run_for(6)  # Remaining 60%

        assert task.state is ActionState.COMPLETED
        assert worker.log == [
            "start@0.0",
            "interrupted@4.0(40%)",
            "resume@7.0",
            "done@13.0",
        ]

    def test_contested_resource_is_claimed_at_the_start(self):
        """The right pattern for a rival resource: claim it, do not re-check it.

        The patch holds one serving. The first sheep claims it in on_start and
        grazes unimpeded; the second cannot start at all and decides what to do
        from on_fail, which is where "go elsewhere" or "wait" would live.
        """
        model = Model()
        patch = {"servings": 1}
        first, second = ActionAgent(model), ActionAgent(model)
        first.energy = second.energy = 0.0
        second.looked_elsewhere = False

        class Graze(Action):
            def __init__(self, sheep):
                super().__init__(
                    sheep,
                    duration=3.0,
                    start_requirements=lambda a: patch["servings"] > 0,
                )

            def on_start(self):
                patch["servings"] -= 1  # claimed, so nobody else can take it

            def on_complete(self):
                self.agent.energy += 30

            def on_fail(self):
                self.agent.looked_elsewhere = True

        first_graze = first.start_action(Graze(first))
        second_graze = second.start_action(Graze(second))

        model.run_for(4)

        assert first_graze.state is ActionState.COMPLETED
        assert first.energy == 30
        assert second_graze.state is ActionState.FAILED
        assert second.looked_elsewhere
        assert second.energy == 0.0

    def test_completion_requirement_for_a_condition_nobody_can_claim(self):
        """Market hours cannot be reserved, so the check belongs at completion."""
        model = Model()
        trader = ActionAgent(model)
        trader.filled = False
        market = {"open": True}

        class Trade(Action):
            def __init__(self, agent):
                super().__init__(
                    agent,
                    duration=5.0,
                    completion_requirements=lambda a: market["open"],
                )

            def on_complete(self):
                self.agent.filled = True

        trade = trader.start_action(Trade(trader))
        model.run_for(2)
        market["open"] = False  # closes while the trade is in flight
        model.run_for(4)

        assert trade.state is ActionState.FAILED
        assert not trader.filled


# --- Wake contract (on_idle) ---


class IdleRecorder(ActionAgent):
    """Agent that records every on_idle wake with the state it saw."""

    def __init__(self, model):
        super().__init__(model)
        self.wakes = []

    def on_idle(self, previous):
        state = previous.state if previous is not None else None
        self.wakes.append((self.model.time, previous, state))


class TestOnIdle:
    def make_recorder(self):
        model = Model()
        return model, IdleRecorder(model)

    def test_wake_fires_after_completion(self):
        model, agent = self.make_recorder()
        action = TrackedAction(agent, duration=5.0)
        agent.start_action(action)

        model.run_for(6)

        assert agent.wakes == [(5.0, action, ActionState.COMPLETED)]

    def test_wake_is_deferred_not_synchronous(self):
        model, agent = self.make_recorder()
        action = TrackedAction(agent)
        agent.start_action(action)
        agent.interrupt_action()

        # The release only queued the wake; nothing has fired yet.
        assert agent.wakes == []

        model.run_for(1)
        assert agent.wakes == [(0.0, action, ActionState.INTERRUPTED)]

    def test_wake_fires_for_cancel(self):
        model, agent = self.make_recorder()
        action = TrackedAction(agent, duration=5.0)
        agent.start_action(action)
        model.run_for(2)

        assert action.cancel()
        model.run_for(1)

        assert agent.wakes == [(2.0, action, ActionState.CANCELED)]

    def test_no_wake_for_canceling_an_interrupted_action(self):
        model, agent = self.make_recorder()
        action = TrackedAction(agent, duration=5.0)
        agent.start_action(action)
        action.interrupt()
        model.run_for(1)  # the interrupt's wake
        agent.wakes.clear()

        action.cancel()
        model.run_for(1)

        assert agent.wakes == []

    def test_wake_fires_for_interrupt(self):
        model, agent = self.make_recorder()
        action = TrackedAction(agent, duration=5.0)
        agent.start_action(action)
        model.run_for(2)

        assert action.interrupt()
        model.run_for(1)

        assert agent.wakes == [(2.0, action, ActionState.INTERRUPTED)]

    def test_wake_fires_for_start_failure(self):
        model, agent = self.make_recorder()
        action = TrackedAction(agent, start_requirements=lambda a: False)
        agent.start_action(action)

        model.run_for(1)

        assert agent.wakes == [(0.0, action, ActionState.FAILED)]

    def test_wake_fires_for_completion_failure(self):
        model, agent = self.make_recorder()
        action = TrackedAction(
            agent, duration=3.0, completion_requirements=lambda a: False
        )
        agent.start_action(action)

        model.run_for(4)

        assert agent.wakes == [(3.0, action, ActionState.FAILED)]

    def test_no_wake_when_interrupt_for_refills_the_slot(self):
        model, agent = self.make_recorder()
        first = TrackedAction(agent, duration=5.0, priority=1.0)
        second = TrackedAction(agent, duration=3.0, priority=2.0)
        agent.start_action(first)

        assert agent.interrupt_for(second)
        model.run_for(4)

        # No idle gap ever existed at t=0; the only wake is second's end.
        assert agent.wakes == [(3.0, second, ActionState.COMPLETED)]

    def test_coalesced_wake_reports_latest_action(self):
        model, agent = self.make_recorder()
        first = TrackedAction(agent)
        agent.start_action(first)
        agent.interrupt_action()
        second = TrackedAction(agent, duration=0.0)
        agent.start_action(second)  # completes instantly, wake still pending

        model.run_for(1)

        assert agent.wakes == [(0.0, second, ActionState.COMPLETED)]

    def test_zero_duration_chain_wakes_once_per_time(self):
        # Without the once-per-time guard this recurses forever: the test
        # hangs rather than fails.
        model = Model()

        class Chain(ActionAgent):
            def __init__(self, model):
                super().__init__(model)
                self.idle_calls = 0

            def on_idle(self, previous):
                self.idle_calls += 1
                self.start_action(Action(self, duration=0.0, name="Step"))

        agent = Chain(model)
        agent.start_action(Action(agent, duration=1.0))
        with pytest.warns(RuntimeWarning) as record:
            model.run_for(2)

        assert agent.idle_calls == 1
        (warning,) = record
        assert str(warning.message).startswith(
            "Chain.on_idle was not called after Step"
        )
        assert warning.filename.endswith("actions.py")

    def test_no_warning_when_on_idle_refills_the_slot(self):
        """An action that ends inside on_idle, followed by another, is fine."""
        model = Model()

        class Retry(ActionAgent):
            def on_idle(self, previous):
                self.start_action(Action(self, start_requirements=lambda a: False))
                self.start_action(Action(self, duration=1.0))

        agent = Retry(model)
        agent.start_action(Action(agent, duration=1.0))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model.run_for(1.5)

        assert agent.is_busy

    def test_agent_chains_actions_across_time(self):
        model = Model()

        class Worker(ActionAgent):
            def __init__(self, model):
                super().__init__(model)
                self.done = 0

            def on_idle(self, previous):
                self.done += 1
                self.start_action(Action(self, duration=1.0))

        agent = Worker(model)
        agent.start_action(Action(agent, duration=1.0))
        model.run_for(3.5)

        # Completions at t=1, 2, 3; each wake starts the next task.
        assert agent.done == 3

    def test_wake_runs_after_all_completions_at_that_time(self):
        model = Model()
        observed = []
        other = TrackedAction(ActionAgent(model), duration=5.0)

        class Nosy(ActionAgent):
            def on_idle(self, previous):
                observed.append(other.state)

        nosy = Nosy(model)
        mine = TrackedAction(nosy, duration=5.0)
        nosy.start_action(mine)
        other.agent.start_action(other)

        model.run_for(6)

        # Both actions end at t=5. The wake is Priority.LOW, so it runs
        # after the other agent's completion even though that completion
        # was queued later.
        assert observed == [ActionState.COMPLETED]

    def test_remove_cancels_pending_wake(self):
        model, agent = self.make_recorder()
        action = TrackedAction(agent)
        agent.start_action(action)
        agent.interrupt_action()

        agent.remove()
        model.run_for(1)

        assert agent.wakes == []
        assert agent._wake_event is None
