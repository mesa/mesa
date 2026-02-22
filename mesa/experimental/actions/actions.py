"""mesa.experimental.actions: Action support for Mesa agents.

Adds the ability for agents to perform actions that take time, can be
interrupted, and give proportional reward based on a reward curve.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mesa.time import Event

from mesa import Agent, Model

# --- Reward curves ---
# Any callable [0, 1] → [0, 1] works. These are the two essentials.


def linear(progress: float) -> float:
    """x% time = x% reward. Default."""
    return progress


def step(progress: float) -> float:
    """All or nothing. No partial reward."""
    return 1.0 if progress >= 1.0 else 0.0


class Action:
    """An action an agent can perform over time.

    Attributes:
        name: Human-readable identifier.
        duration: Time to complete.
        priority: Used to decide whether to interrupt. Higher = more important.
        reward_curve: Maps progress [0,1] to effective completion [0,1].
        on_effect: Callback(agent, completion) called on completion or interruption.
        interruptible: Whether this action can be interrupted.
        progress: Current execution progress, 0.0 to 1.0.
    """

    def __init__(
        self,
        name: str,
        duration: float = 1.0,
        priority: float = 0.0,
        reward_curve: Callable[[float], float] = linear,
        on_effect: Callable[[ActionAgent, float], None] | None = None,
        interruptible: bool = True,
    ):
        """Initialise an Action."""
        self.name = name
        self.duration = duration
        self.priority = priority
        self.reward_curve = reward_curve
        self.on_effect = on_effect
        self.interruptible = interruptible

        # Runtime state
        self.progress: float = 0.0
        self._started_at: float | None = None
        self._event: Event | None = None

    @property
    def effective_completion(self) -> float:
        """Reward earned at current progress, based on reward curve."""
        return self.reward_curve(self.progress)

    @property
    def remaining_time(self) -> float:
        """Time remaining to complete this action."""
        return self.duration * (1.0 - self.progress)

    def __repr__(self) -> str:
        """String representation."""
        return f"Action({self.name!r}, progress={self.progress:.0%})"


class ActionAgent(Agent):
    """An Agent that can perform Actions.

    Extends the base Agent with the ability to start, interrupt,
    and cancel timed actions. Actions integrate with Mesa's event
    scheduling system for precise timing.

    Attributes:
        current_action: The Action currently being performed, or None.
    """

    def __init__(self, model: Model, *args, **kwargs):
        """Initialise an ActionAgent."""
        super().__init__(model, *args, **kwargs)
        self.current_action: Action | None = None

    @property
    def is_busy(self) -> bool:
        """Whether the agent is currently performing an action."""
        return self.current_action is not None

    def start_action(self, action: Action) -> None:
        """Start performing an action.

        Args:
            action: The Action to perform.

        Raises:
            ValueError: If the agent is already performing an action.
        """
        if self.current_action is not None:
            raise ValueError(
                f"Agent {self.unique_id} is already performing {self.current_action.name!r}. "
                f"Use interrupt_for() or cancel_action() first."
            )

        self.current_action = action
        action.progress = 0.0
        action._started_at = self.model.time

        if action.duration <= 0:
            self._complete_action()
            return

        action._event = self.model.schedule_event(
            self._complete_action, after=action.duration
        )

    def interrupt_for(self, action: Action) -> None:
        """Interrupt the current action and start a new one.

        The interrupted action's on_effect is called with partial
        completion based on its reward curve. If no current action,
        simply starts the new one.

        Args:
            action: The Action to perform instead.
        """
        if self.current_action is not None:
            if not self.current_action.interruptible:
                return
            self._interrupt_current()

        self.start_action(action)

    def cancel_action(self) -> None:
        """Cancel the current action.

        The action's on_effect is called with partial completion
        based on its reward curve.
        """
        if self.current_action is not None:
            self._interrupt_current()

    def remove(self) -> None:
        """Remove the agent, canceling any current action."""
        if self.current_action and self.current_action._event:
            self.current_action._event.cancel()
        self.current_action = None
        super().remove()

    def _update_progress(self) -> None:
        """Update current action's progress based on elapsed time."""
        action = self.current_action
        if action and action._started_at is not None and action.duration > 0:
            elapsed = self.model.time - action._started_at
            action.progress = min(1.0, action.progress + elapsed / action.duration)

    def _interrupt_current(self) -> None:
        """Stop the current action and apply partial effect."""
        action = self.current_action
        if action is None:
            return

        if action._event:
            action._event.cancel()
            action._event = None

        self._update_progress()
        if action.on_effect:
            action.on_effect(self, action.effective_completion)

        action._started_at = None
        self.current_action = None

    def _complete_action(self) -> None:
        """Handle action completion. Called by the scheduled event."""
        action = self.current_action
        if action is None:
            return

        action.progress = 1.0
        action._event = None
        action._started_at = None
        self.current_action = None

        if action.on_effect:
            action.on_effect(self, action.effective_completion)
