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
        self.resume_count = 0
        self.completed = False
        self.interrupted = False
        self.interrupt_progress = None

    def on_start(self):
        self.start_count += 1

    def on_resume(self):
        self.resume_count += 1

    def on_complete(self):
        self.completed = True

    def on_interrupt(self, progress):
        self.interrupted = True
        self.interrupt_progress = progress


def make_model_and_agent():
    model = Model()
    agent = Agent(model)
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
        agent.cancel_action()

        agent.start_action(action)

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

            def on_interrupt(self, progress):
                pass

        action = StartTracker(agent)

        agent.start_action(action)
        assert action.started == ["start"]

        model.run_for(3)
        agent.cancel_action()

        agent.start_action(action)
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

            def on_interrupt(self, progress):
                pass

        action = ResumeTracker(agent)

        agent.start_action(action)
        model.run_for(3)
        agent.cancel_action()

        agent.start_action(action)

        assert action.log == ["start", "resume"]

    def test_multiple_resume_cycles(self):
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)

        agent.start_action(action)
        model.run_for(2)
        agent.cancel_action()

        agent.start_action(action)
        model.run_for(3)
        agent.cancel_action()

        agent.start_action(action)
        model.run_for(5)

        assert action.start_count == 1
        assert action.resume_count == 2
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
        agent.cancel_action()

        assert action.progress == pytest.approx(0.3)

        # Time passes but progress doesn't change
        model.run_for(5)
        assert action.progress == pytest.approx(0.3)

    def test_progress_live_after_resume(self):
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)

        agent.start_action(action)
        model.run_for(4)  # 40%
        agent.cancel_action()

        agent.start_action(action)
        model.run_for(3)  # 40% + 30% = 70%

        assert action.progress == pytest.approx(0.7)


# --- Pause and resume ---


class TestPauseResume:
    def test_resume_continues_from_progress(self):
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)

        agent.start_action(action)
        model.run_for(3)  # 30% done
        agent.cancel_action()

        assert action.progress == pytest.approx(0.3)
        assert action.remaining_time == pytest.approx(7.0)
        assert action.is_resumable

        agent.start_action(action)
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

    def test_completed_action_not_resumable(self):
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

    def test_resume_respects_remaining_duration_only(self):
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

    def test_subclass_on_interrupt(self):
        model, agent = make_model_and_agent()

        class ProgressTracker(Action):
            def __init__(self, agent):
                super().__init__(agent, duration=10.0)
                self.received_progress = []

            def on_interrupt(self, progress):
                self.received_progress.append(progress)

        action = ProgressTracker(agent)

        agent.start_action(action)
        model.run_for(2)
        agent.cancel_action()

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
    def test_remove_cancels_action_silently(self):
        """remove() cancels the event but does NOT fire on_interrupt."""
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)

        agent.start_action(action)
        model.run_for(3)
        agent.remove()

        assert not action.interrupted
        assert agent.current_action is None

    def test_remove_with_explicit_cleanup(self):
        """Users can opt into on_interrupt by calling cancel_action first."""
        model = Model()
        agent = Agent(model)
        action = TrackedAction(agent, duration=10.0)

        agent.start_action(action)
        model.run_for(3)

        # Explicit cleanup pattern
        agent.cancel_action()  # Fires on_interrupt
        assert action.interrupted
        assert action.interrupt_progress == pytest.approx(0.3)


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

            def on_interrupt(self, progress):
                pass

        action = DetectResumeAction(agent)

        agent.start_action(action)
        model.run_for(3)
        agent.cancel_action()

        agent.start_action(action)
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


# --- Integration: realistic scenarios ---


class TestRealisticScenarios:
    def test_sheep_forage_flee_resume(self):
        """Sheep forages, flees from predator, resumes foraging."""
        model = Model()
        sheep = Agent(model)
        sheep.energy = 50.0
        sheep.alive = True

        class Forage(Action):
            def __init__(self, sheep):
                super().__init__(sheep, duration=5.0)

            def on_complete(self):
                self.agent.energy += 30

            def on_interrupt(self, progress):
                self.agent.energy += 30 * progress

        class Flee(Action):
            def __init__(self, sheep):
                super().__init__(sheep, duration=2.0, interruptible=False)

            def on_complete(self):
                pass  # survived

            def on_interrupt(self, progress):
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
        sheep.start_action(forage)
        model.run_for(2)  # 40% of 5.0 = 2.0 time units

        assert forage.state is ActionState.COMPLETED
        # Partial (18.0) + full (30.0) = 48.0 added
        assert sheep.energy == pytest.approx(50.0 + 18.0 + 30.0)

    def test_sequential_actions(self):
        """Agent performs multiple actions in sequence."""
        model = Model()
        agent = Agent(model)
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
        worker.log = []

        class Task(Action):
            def on_start(self):
                self.agent.log.append(f"start@{self.agent.model.time}")

            def on_resume(self):
                self.agent.log.append(f"resume@{self.agent.model.time}")

            def on_complete(self):
                self.agent.log.append(f"done@{self.agent.model.time}")

            def on_interrupt(self, progress):
                self.agent.log.append(
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
        assert worker.log == [
            "start@0.0",
            "interrupted@4.0(40%)",
            "resume@7.0",
            "done@13.0",
        ]


# --- HasActions mixin ---


class TestHasActionsMixin:
    """Tests for agent-level action lifecycle hooks via HasActions mixin."""

    def test_on_action_complete_fires_after_action_hook(self):
        """Agent hook fires after Action.on_complete."""
        from mesa.experimental.actions import HasActions

        model = Model()
        call_order = []

        class TrackingAction(Action):
            def on_complete(self):
                call_order.append("action")

        class TrackingAgent(Agent, HasActions):
            def on_action_complete(self, action):
                call_order.append("agent")

        agent = TrackingAgent(model)
        action = TrackingAction(agent, duration=3.0)

        agent.start_action(action)
        model.run_for(3)

        assert call_order == ["action", "agent"]

    def test_on_action_start_fires_after_action_hook(self):
        """Agent hook fires after Action.on_start."""
        from mesa.experimental.actions import HasActions

        model = Model()
        call_order = []

        class TrackingAction(Action):
            def on_start(self):
                call_order.append("action")

        class TrackingAgent(Agent, HasActions):
            def on_action_start(self, action):
                call_order.append("agent")

        agent = TrackingAgent(model)
        action = TrackingAction(agent, duration=3.0)

        agent.start_action(action)

        assert call_order == ["action", "agent"]

    def test_on_action_interrupt_fires_after_action_hook(self):
        """Agent hook fires after Action.on_interrupt."""
        from mesa.experimental.actions import HasActions

        model = Model()
        call_order = []

        class TrackingAction(Action):
            def on_interrupt(self, progress):
                call_order.append("action")

        class TrackingAgent(Agent, HasActions):
            def on_action_interrupt(self, action, progress):
                call_order.append("agent")

        agent = TrackingAgent(model)
        action = TrackingAction(agent, duration=10.0)

        agent.start_action(action)
        model.run_for(3)
        agent.cancel_action()

        assert call_order == ["action", "agent"]

    def test_hooks_not_called_without_mixin(self):
        """Agent without HasActions mixin doesn't receive hooks."""
        model = Model()
        call_log = []

        class TrackingAction(Action):
            def on_start(self):
                call_log.append("action_start")

            def on_complete(self):
                call_log.append("action_complete")

        # Regular agent without HasActions
        agent = Agent(model)
        action = TrackingAction(agent, duration=3.0)

        agent.start_action(action)
        model.run_for(3)

        # Only action hooks should fire, no agent hooks
        assert call_log == ["action_start", "action_complete"]

    def test_instantaneous_action_skips_on_action_start(self):
        """Zero-duration actions skip on_action_start but fire on_action_complete."""
        from mesa.experimental.actions import HasActions

        model = Model()
        call_log = []

        class TrackingAction(Action):
            def on_start(self):
                call_log.append("action_start")

            def on_complete(self):
                call_log.append("action_complete")

        class TrackingAgent(Agent, HasActions):
            def on_action_start(self, action):
                call_log.append("agent_start")

            def on_action_complete(self, action):
                call_log.append("agent_complete")

        agent = TrackingAgent(model)
        action = TrackingAction(agent, duration=0)  # Instantaneous

        agent.start_action(action)

        # on_action_start should NOT be called for instantaneous actions
        assert call_log == ["action_start", "action_complete", "agent_complete"]

    def test_interrupt_passes_progress_correctly(self):
        """Agent's on_action_interrupt receives correct progress value."""
        from mesa.experimental.actions import HasActions

        model = Model()

        class TrackingAgent(Agent, HasActions):
            def __init__(self, model):
                super().__init__(model)
                self.received_progress = None

            def on_action_interrupt(self, action, progress):
                self.received_progress = progress

        agent = TrackingAgent(model)
        action = Action(agent, duration=10.0)

        agent.start_action(action)
        model.run_for(4)  # 40%
        agent.cancel_action()

        assert agent.received_progress == pytest.approx(0.4)

    def test_on_action_start_fires_on_resume(self):
        """Agent hook fires after on_resume too."""
        from mesa.experimental.actions import HasActions

        model = Model()
        call_log = []

        class TrackingAction(Action):
            def on_start(self):
                call_log.append("action_start")

            def on_resume(self):
                call_log.append("action_resume")

            def on_interrupt(self, progress):
                pass

        class TrackingAgent(Agent, HasActions):
            def on_action_start(self, action):
                call_log.append("agent_start")

        agent = TrackingAgent(model)
        action = TrackingAction(agent, duration=10.0)

        agent.start_action(action)
        model.run_for(3)
        agent.cancel_action()

        # Resume the action
        agent.start_action(action)

        assert call_log == [
            "action_start",
            "agent_start",
            "action_resume",
            "agent_start",
        ]

    def test_cancel_fires_on_action_interrupt(self):
        """cancel() also fires on_action_interrupt."""
        from mesa.experimental.actions import HasActions

        model = Model()

        class TrackingAgent(Agent, HasActions):
            def __init__(self, model):
                super().__init__(model)
                self.interrupt_called = False

            def on_action_interrupt(self, action, progress):
                self.interrupt_called = True

        agent = TrackingAgent(model)
        action = Action(agent, duration=10.0, interruptible=False)  # Non-interruptible

        agent.start_action(action)
        model.run_for(5)
        action.cancel()  # Force cancel

        assert agent.interrupt_called

    def test_sheep_forage_rest_pattern(self):
        """Demonstrate the Sheep/Forage/Rest pattern from the PR description."""
        from mesa.experimental.actions import HasActions

        model = Model()

        class Forage(Action):
            def on_complete(self):
                self.agent.energy += 30

        class Rest(Action):
            def on_complete(self):
                self.agent.energy = min(100, self.agent.energy + 20)

        class Sheep(Agent, HasActions):
            def __init__(self, model):
                super().__init__(model)
                self.energy = 50
                self.action_log = []

            def on_action_complete(self, action):
                self.action_log.append(action.__class__.__name__)
                self.decide_next()

            def decide_next(self):
                if self.energy < 70:
                    self.start_action(Rest(self, duration=2.0))
                else:
                    self.start_action(Forage(self, duration=3.0))

        sheep = Sheep(model)

        # Start first action manually
        sheep.start_action(Forage(sheep, duration=3.0))

        # Run for 8 time units - should cycle through actions
        model.run_for(8)

        # Verify action chaining worked
        assert len(sheep.action_log) >= 2
        assert sheep.energy >= 50  # Should have gained energy

    def test_agent_receives_action_reference(self):
        """Agent hooks receive the action as an argument."""
        from mesa.experimental.actions import HasActions

        model = Model()

        class NamedAction(Action):
            def __init__(self, agent, name):
                super().__init__(agent, duration=2.0)
                self.action_name = name

        class TrackingAgent(Agent, HasActions):
            def __init__(self, model):
                super().__init__(model)
                self.received_actions = []

            def on_action_complete(self, action):
                self.received_actions.append(action.action_name)

        agent = TrackingAgent(model)

        agent.start_action(NamedAction(agent, "first"))
        model.run_for(2)

        agent.start_action(NamedAction(agent, "second"))
        model.run_for(2)

        assert agent.received_actions == ["first", "second"]
