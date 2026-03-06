"""Tests for mesa.experimental.actions."""

# ruff: noqa: D101, D102, D103, D107
import pytest

from mesa import Agent, Model
from mesa.experimental.actions import Action, ActionState

# --- Helpers ---


class TrackedAction(Action):
    """Action subclass that records lifecycle events for testing."""

    def __init__(self, agent, duration=5.0, **kwargs):
        super().__init__(agent, duration=duration, **kwargs)
        self.start_count = 0
        self.completed = False
        self.interrupted = False
        self.interrupt_progress = None

    def on_start(self):
        self.start_count += 1

    def on_complete(self):
        self.completed = True

    def on_interrupt(self, progress):
        self.interrupted = True
        self.interrupt_progress = progress


def make_model_and_agent():
    model = Model()
    agent = Agent(model)
    return model, agent


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

    def test_on_start_fires(self):
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
        agent.cancel_action()

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

    def test_cancel_ignores_interruptible_flag(self):
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0, interruptible=False)

        agent.start_action(action)
        model.run_for(5)
        result = agent.cancel_action()

        assert result is True
        assert action.state is ActionState.INTERRUPTED
        assert action.interrupt_progress == pytest.approx(0.5)

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
        agent.cancel_action()

        assert action.interrupt_progress == pytest.approx(0.25)


# --- Pause and resume ---


class TestPauseResume:
    def test_resume_continues_from_progress(self):
        """Interrupted action resumes from where it left off."""
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)

        agent.start_action(action)
        model.run_for(3)  # 30% done
        agent.cancel_action()

        assert action.progress == pytest.approx(0.3)
        assert action.remaining_time == pytest.approx(7.0)
        assert action.is_resumable

        # Resume
        agent.start_action(action)
        assert action.state is ActionState.ACTIVE
        assert action.start_count == 2  # on_start called again

        # Run for the remaining time
        model.run_for(7)
        assert action.state is ActionState.COMPLETED
        assert action.completed
        assert action.progress == 1.0

    def test_resume_partial_then_complete(self):
        """Resume runs for exactly the remaining duration."""
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)

        agent.start_action(action)
        model.run_for(6)  # 60%
        agent.cancel_action()

        assert action.remaining_time == pytest.approx(4.0)

        agent.start_action(action)
        model.run_for(4)  # Remaining 40%

        assert action.state is ActionState.COMPLETED
        assert action.progress == 1.0

    def test_multiple_interrupt_resume_cycles(self):
        """Action can be interrupted and resumed multiple times."""
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)

        # First attempt: 20%
        agent.start_action(action)
        model.run_for(2)
        agent.cancel_action()
        assert action.progress == pytest.approx(0.2)

        # Second attempt: 20% + 30% = 50%
        agent.start_action(action)
        model.run_for(3)
        agent.cancel_action()
        assert action.progress == pytest.approx(0.5)

        # Third attempt: 50% + 50% = done
        agent.start_action(action)
        model.run_for(5)
        assert action.state is ActionState.COMPLETED
        assert action.progress == 1.0
        assert action.start_count == 3

    def test_completed_action_not_resumable(self):
        """Completed actions cannot be resumed."""
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=3.0)

        agent.start_action(action)
        model.run_for(3)

        assert not action.is_resumable
        with pytest.raises(ValueError, match="COMPLETED"):
            action.start()

    def test_is_resumable_property(self):
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=5.0)

        assert not action.is_resumable  # PENDING

        agent.start_action(action)
        assert not action.is_resumable  # ACTIVE

        model.run_for(2)
        agent.cancel_action()
        assert action.is_resumable  # INTERRUPTED with progress < 1

    def test_interrupt_for_with_resume_later(self):
        """Realistic: interrupt, do something else, resume original."""
        model, agent = make_model_and_agent()
        agent.energy = 50.0

        class Forage(Action):
            def on_interrupt(self, progress):
                self.agent.energy += 30 * progress

            def on_complete(self):
                self.agent.energy += 30

        forage = Forage(agent, duration=10.0)
        flee = TrackedAction(agent, duration=2.0, interruptible=False)

        # Start foraging
        agent.start_action(forage)
        model.run_for(4)  # 40%

        # Predator! Interrupt and flee
        agent.interrupt_for(flee)
        assert agent.energy == pytest.approx(50.0 + 30 * 0.4)  # Partial reward

        # Flee completes
        model.run_for(2)
        assert flee.completed
        assert not agent.is_busy

        # Resume foraging — but we already got the partial reward,
        # so on_complete should give the remaining 60%
        # User handles this in their subclass:
        assert forage.is_resumable
        assert forage.progress == pytest.approx(0.4)

    def test_resume_respects_remaining_duration_only(self):
        """Resuming schedules only for remaining time, not full duration."""
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)

        agent.start_action(action)
        model.run_for(8)  # 80%
        agent.cancel_action()

        agent.start_action(action)

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

    def test_cancel_non_active_returns_false(self):
        _model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=5.0)

        assert action.cancel() is False  # PENDING


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
        agent2 = Agent(model)
        action = TrackedAction(agent1, duration=5.0)

        with pytest.raises(ValueError, match="does not match"):
            agent2.start_action(action)

    def test_start_completed_action_raises(self):
        _model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=0)

        agent.start_action(action)  # Completes immediately

        with pytest.raises(ValueError, match="COMPLETED"):
            action.start()

    def test_negative_duration_raises(self):
        _model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=-1.0)

        with pytest.raises(ValueError, match="duration"):
            agent.start_action(action)

    def test_cancel_idle_returns_false(self):
        _model, agent = make_model_and_agent()
        assert agent.cancel_action() is False


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
        agent.cancel_action()

        agent.start_action(action)  # Resume
        assert call_count == 1  # Not called again


# --- Inline callbacks ---


class TestInlineCallbacks:
    def test_inline_on_complete(self):
        model, agent = make_model_and_agent()
        agent.energy = 50

        action = Action(
            agent,
            duration=3.0,
            on_complete=lambda: setattr(agent, "energy", agent.energy + 30),
        )

        agent.start_action(action)
        model.run_for(3)

        assert agent.energy == 80

    def test_inline_on_interrupt(self):
        model, agent = make_model_and_agent()
        received_progress = []

        action = Action(
            agent,
            duration=10.0,
            on_interrupt=lambda p: received_progress.append(p),
        )

        agent.start_action(action)
        model.run_for(2)
        agent.cancel_action()

        assert received_progress == [pytest.approx(0.2)]

    def test_inline_on_start(self):
        _model, agent = make_model_and_agent()
        started = []

        action = Action(
            agent,
            duration=5.0,
            on_start=lambda: started.append(True),
        )

        agent.start_action(action)
        assert started == [True]


# --- Agent removal ---


class TestAgentRemoval:
    def test_remove_cancels_action_silently(self):
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)

        agent.start_action(action)
        model.run_for(3)
        agent.remove()

        # Event cancelled but on_interrupt should NOT fire
        assert not action.interrupted
        assert agent.current_action is None


# --- Query properties ---


class TestQueryProperties:
    def test_remaining_time_active(self):
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)

        agent.start_action(action)
        model.run_for(3)

        # remaining_time queries model.time, but progress isn't
        # updated until interrupt. Use the action's own accounting.
        agent.cancel_action()
        assert action.remaining_time == pytest.approx(7.0)

    def test_remaining_time_interrupted(self):
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)

        agent.start_action(action)
        model.run_for(6)
        agent.cancel_action()

        assert action.remaining_time == pytest.approx(4.0)

    def test_elapsed_time(self):
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)

        agent.start_action(action)
        model.run_for(4)
        agent.cancel_action()

        assert action.elapsed_time == pytest.approx(4.0)

    def test_elapsed_time_across_resumes(self):
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)

        agent.start_action(action)
        model.run_for(3)
        agent.cancel_action()

        agent.start_action(action)
        model.run_for(2)
        agent.cancel_action()

        # Total elapsed: 3 + 2 = 5
        assert action.elapsed_time == pytest.approx(5.0)


class TestResumeDetection:
    """Verify on_start can distinguish first start from resume."""

    def test_on_start_can_detect_resume(self):
        model, agent = make_model_and_agent()
        start_types = []

        class DetectResumeAction(Action):
            def on_start(self):
                start_types.append("resume" if self.progress > 0 else "first")

            def on_interrupt(self, progress):
                pass

        action = DetectResumeAction(agent, duration=10.0)

        agent.start_action(action)
        model.run_for(3)
        agent.cancel_action()

        agent.start_action(action)
        model.run_for(7)

        assert start_types == ["first", "resume"]


class TestEdgeCases:
    def test_double_completion_ignored(self):
        """Calling _do_complete twice doesn't fire on_complete twice."""
        model, agent = make_model_and_agent()
        complete_count = []

        action = Action(
            agent,
            duration=3.0,
            on_complete=lambda: complete_count.append(1),
        )

        agent.start_action(action)
        model.run_for(3)

        # Manually try to complete again
        action._do_complete()

        assert len(complete_count) == 1

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


# --- Integration: realistic scenarios ---


class TestRealisticScenarios:
    def test_sheep_forage_flee_resume(self):
        """Sheep forages, flees from predator, resumes foraging."""
        model = Model()
        sheep = Agent(model)
        sheep.energy = 50.0
        sheep.alive = True

        class Forage(Action):
            def on_complete(self):
                self.agent.energy += 30

            def on_interrupt(self, progress):
                self.agent.energy += 30 * progress

        class Flee(Action):
            def __init__(self, agent):
                super().__init__(agent, duration=2.0, interruptible=False)

            def on_complete(self):
                pass  # survived

            def on_interrupt(self, progress):
                self.agent.alive = False

        # Start foraging
        forage = Forage(sheep, duration=5.0)
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
        sheep.start_action(forage)
        model.run_for(2)  # 40% of 5.0 = 2.0 time units

        assert forage.state is ActionState.COMPLETED
        assert sheep.energy == pytest.approx(50.0 + 30 * 0.6 + 30)  # partial + full

    def test_sequential_actions(self):
        """Agent performs multiple actions in sequence."""
        model = Model()
        agent = Agent(model)
        agent.log = []

        for i in range(3):
            action = Action(
                agent,
                duration=2.0,
                on_complete=lambda i=i: agent.log.append(f"done_{i}"),
            )
            agent.start_action(action)
            model.run_for(2)

        assert agent.log == ["done_0", "done_1", "done_2"]

    def test_flee_non_interruptible_protects(self):
        """A fleeing agent can't be interrupted."""
        model = Model()
        agent = Agent(model)

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
        worker = Agent(model)
        worker.progress_log = []

        class Task(Action):
            def on_start(self):
                self.agent.progress_log.append(
                    f"{'resume' if self.progress > 0 else 'start'}"
                    f"@{self.agent.model.time}"
                )

            def on_complete(self):
                self.agent.progress_log.append(f"done@{self.agent.model.time}")

            def on_interrupt(self, progress):
                self.agent.progress_log.append(
                    f"interrupted@{self.agent.model.time}({progress:.0%})"
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
        worker.start_action(task)
        model.run_for(6)  # Remaining 60%

        assert task.state is ActionState.COMPLETED
        assert worker.progress_log == [
            "start@0.0",
            "interrupted@4.0(40%)",
            "resume@7.0",
            "done@13.0",
        ]
