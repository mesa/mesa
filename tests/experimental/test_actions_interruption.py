"""Tests for mesa.experimental.actions interruption and resumption behavior.

Provides comprehensive coverage for action lifecycle transitions, interrupt
callbacks, progress tracking, and resumable actions—critical for agent
interaction patterns that depend on interruptible task scheduling.
"""

# ruff: noqa: D101, D102, D103, D107
import pytest # type: ignore

from mesa import Agent, Model
from mesa.experimental.actions import Action, ActionState


class TrackedAction(Action):
    """Action subclass that tracks lifecycle events for testing."""

    def __init__(self, agent, duration=5.0, **kwargs):
        super().__init__(agent, duration=duration, **kwargs)
        self.on_start_called = 0
        self.on_resume_called = 0
        self.on_complete_called = 0
        self.on_interrupt_called = 0
        self.interrupt_progress = None

    def on_start(self):
        self.on_start_called += 1

    def on_resume(self):
        self.on_resume_called += 1
        super().on_resume()

    def on_complete(self):
        self.on_complete_called += 1

    def on_interrupt(self, progress):
        self.on_interrupt_called += 1
        self.interrupt_progress = progress


def make_model_and_agent():
    """Factory for creating a test model and agent."""
    model = Model()
    agent = Agent(model)
    return model, agent


class TestActionInterruption:
    """Tests for action interruption behavior."""

    def test_interrupt_active_action_returns_true(self):
        """Interrupting an active action should return True."""
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)
        action.start()

        assert action.state is ActionState.ACTIVE
        success = action.interrupt()

        assert success is True
        assert action.state is ActionState.INTERRUPTED

    def test_interrupt_inactive_action_returns_false(self):
        """Interrupting a non-active action should return False."""
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)

        assert action.state is ActionState.PENDING
        success = action.interrupt()

        assert success is False
        assert action.state is ActionState.PENDING

    def test_interrupt_calls_on_interrupt_callback(self):
        """Interrupting an action should call on_interrupt with progress."""
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)
        action.start()

        # Manually set progress to simulate partial completion
        action._progress = 0.5
        action._event = None  # Cancel the scheduled event

        action.interrupt()

        assert action.on_interrupt_called == 1
        assert abs(action.interrupt_progress - 0.5) < 0.01

    def test_interrupt_non_interruptible_action_returns_false(self):
        """Interrupting a non-interruptible action should fail."""
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0, interruptible=False)
        action.start()

        assert action.interruptible is False
        success = action.interrupt()

        assert success is False
        assert action.state is ActionState.ACTIVE

    def test_interrupt_clears_agent_reference(self):
        """Interrupting an action should clear agent.current_action."""
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)
        action.start()
        agent.current_action = action

        action.interrupt()

        assert agent.current_action is None

    def test_interrupt_cancels_scheduled_event(self):
        """Interrupting should cancel the scheduled completion event."""
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)
        action.start()

        assert action._event is not None
        action.interrupt()

        assert action._event is None


class TestActionResumption:
    """Tests for resuming interrupted actions."""

    def test_resumable_action_after_interrupt(self):
        """A recently interrupted action should be resumable."""
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)
        action.start()
        action.interrupt()

        assert action.state is ActionState.INTERRUPTED
        assert action.is_resumable is True

    def test_non_resumable_action_when_completed(self):
        """A completed action should not be resumable."""
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)
        action.start()
        action._do_complete()

        assert action.state is ActionState.COMPLETED
        assert action.is_resumable is False

    def test_resume_calls_on_resume_not_on_start(self):
        """Resuming an action should call on_resume."""
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)
        action.start()

        assert action.on_start_called == 1
        assert action.on_resume_called == 0

        action.interrupt()
        action.start()  # Resume

        # on_resume is called when resuming (default on_resume calls on_start)
        assert action.on_resume_called == 1  # Called on resume

    def test_resume_preserves_progress(self):
        """Resuming should preserve progress from before interruption."""
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)
        action.start()

        # Simulate 30% completion
        action._progress = 0.3
        action._event = None

        action.interrupt()

        assert abs(action.progress - 0.3) < 0.01

        action.start()  # Resume

        assert abs(action.progress - 0.3) < 0.01

    def test_resume_invalid_state_raises_error(self):
        """Resuming an action not in PENDING or INTERRUPTED state should raise."""
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)
        action.start()
        action._do_complete()

        # Action is now COMPLETED, not resumable
        with pytest.raises(ValueError, match="Cannot start action in COMPLETED state"):
            action.start()


class TestActionCancel:
    """Tests for action cancellation (forced interrupt)."""

    def test_cancel_non_interruptible_action_succeeds(self):
        """Canceling should work even on non-interruptible actions."""
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0, interruptible=False)
        action.start()

        assert action.interruptible is False
        success = action.cancel()

        assert success is True
        assert action.state is ActionState.INTERRUPTED

    def test_cancel_calls_on_interrupt(self):
        """Canceling should call on_interrupt with progress."""
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)
        action.start()

        action._progress = 0.7
        action._event = None

        action.cancel()

        assert action.on_interrupt_called == 1
        assert abs(action.interrupt_progress - 0.7) < 0.01

    def test_cancel_inactive_action_returns_false(self):
        """Canceling a non-active action should return False."""
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)

        success = action.cancel()

        assert success is False
        assert action.state is ActionState.PENDING


class TestActionProgressTracking:
    """Tests for progress tracking during action execution."""

    def test_progress_starts_at_zero(self):
        """A new action should have progress 0.0."""
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)

        assert action.progress == 0.0

    def test_progress_frozen_on_interrupt(self):
        """Interrupting should freeze progress at that moment."""
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)
        action.start()

        # Simulate 40% elapsed
        action._progress = 0.4
        action._event = None

        action.interrupt()

        # Progress should stay at 0.4
        assert abs(action.progress - 0.4) < 0.01

    def test_remaining_time_computed_live(self):
        """Remaining time should be computed based on duration and progress."""
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)
        action.start()

        action._progress = 0.25
        action._event = None

        # 10 * (1 - 0.25) = 7.5
        assert abs(action.remaining_time - 7.5) < 0.01

    def test_elapsed_time_computed_correctly(self):
        """Elapsed time should be progress * duration."""
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=20.0)
        action.start()

        action._progress = 0.6
        action._event = None

        # 20 * 0.6 = 12.0
        assert abs(action.elapsed_time - 12.0) < 0.01


class TestActionEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_instantaneous_action_completes_immediately(self):
        """An action with duration 0 should complete immediately."""
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=0.0)
        action.start()

        assert action.state is ActionState.COMPLETED
        assert action.on_complete_called == 1

    def test_action_with_callable_duration(self):
        """Actions should support dynamic duration via callable."""
        model, agent = make_model_and_agent()

        def get_duration(agent):
            return 15.0

        action = TrackedAction(agent, duration=get_duration)
        action.start()

        assert action.duration == 15.0

    def test_action_with_callable_priority(self):
        """Actions should support dynamic priority via callable."""
        model, agent = make_model_and_agent()

        def get_priority(agent):
            return 3.0

        action = TrackedAction(agent, priority=get_priority)
        action.start()

        assert action.priority == 3.0

    def test_negative_duration_raises_error(self):
        """Creating an action with negative duration should raise."""
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=-5.0)

        with pytest.raises(ValueError, match="duration must be >= 0"):
            action.start()

    def test_repr_format(self):
        """Action.__repr__ should include state, progress, and duration."""
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=10.0)
        action.start()

        repr_str = repr(action)

        assert "TrackedAction" in repr_str
        assert "ACTIVE" in repr_str
        assert "duration=10.0" in repr_str
