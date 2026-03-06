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
        self.started = False
        self.completed = False
        self.interrupted = False
        self.interrupt_progress = None

    def on_start(self):
        self.started = True

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
        assert action.started
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
        assert action.started

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
        assert not second.started

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

    def test_interrupt_on_interrupt_callback_receives_progress(self):
        model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=4.0)

        agent.start_action(action)
        model.run_for(1)  # 25%
        agent.cancel_action()

        assert action.interrupt_progress == pytest.approx(0.25)


# --- Error handling ---


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

    def test_start_non_pending_raises(self):
        _model, agent = make_model_and_agent()
        action = TrackedAction(agent, duration=0)

        agent.start_action(action)  # Completes immediately

        with pytest.raises(ValueError, match="PENDING"):
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
        agent.energy = 50
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

        # Event should be cancelled but on_interrupt should NOT fire
        # (agent is being destroyed, not making a decision)
        assert not action.interrupted
        assert agent.current_action is None


# --- Integration: realistic scenarios ---


class TestRealisticScenarios:
    def test_sheep_forage_then_flee(self):
        """Sheep forages, gets interrupted by predator, flees."""
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

        # Predator appears
        flee = Flee(sheep)
        result = sheep.interrupt_for(flee)

        assert result is True
        assert sheep.energy == pytest.approx(50.0 + 30 * 0.6)  # Partial forage reward
        assert sheep.current_action is flee

        # Complete the flee
        model.run_for(2)
        assert flee.state is ActionState.COMPLETED
        assert sheep.alive
        assert not sheep.is_busy

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
        assert not distraction.started
