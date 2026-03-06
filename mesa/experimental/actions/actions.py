"""mesa.experimental.actions: Action support for Mesa agents.

Adds the ability for agents to perform actions that take time, can be
interrupted, and give proportional reward based on a reward curve.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from mesa.time import Event

from mesa.agent import Agent
from mesa.model import Model

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
        reschedule_on_interrupt: What happens when this action is interrupted:
            False (default) – action is discarded;
            "remainder" – action is re-queued preserving current progress so it
                resumes from where it left off;
            "full" – action is re-queued with progress reset to 0 so it restarts.
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
        reschedule_on_interrupt: Literal[False, "remainder", "full"] = False,
    ):
        """Initialise an Action.

        Args:
            name: Human-readable label.
            duration: Simulation-time units needed to complete the action.
            priority: Importance level used by request_action for preemption.
            reward_curve: Maps progress [0,1] to earned reward [0,1].
            on_effect: Optional callback(agent, completion) fired at completion
                or interruption with the reward earned so far.
            interruptible: If False, interrupt_for and priority preemption refuse
                to cut this action short.
            reschedule_on_interrupt: Queuing policy on interruption.
                False – discard (default); "remainder" – re-queue preserving
                progress; "full" – re-queue with progress reset to 0.
        """
        self.name = name
        self.duration = duration
        self.priority = priority
        self.reward_curve = reward_curve
        self.on_effect = on_effect
        self.interruptible = interruptible
        self.reschedule_on_interrupt = reschedule_on_interrupt

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

    Extends the base Agent with the ability to start, interrupt, resume, and
    cancel timed actions. Actions integrate with Mesa's event scheduling system
    for precise timing.

    Attributes:
        current_action: The Action currently being performed, or None.
        action_queue: Ordered list of pending Actions. Actions with
            reschedule_on_interrupt="remainder" or "full" are inserted at the
            front when interrupted so they resume before other queued work.
            The queue drains automatically whenever the agent becomes free.
    """

    def __init__(self, model: Model, *args, **kwargs):
        """Initialise an ActionAgent."""
        super().__init__(model, *args, **kwargs)
        self.current_action: Action | None = None
        self.action_queue: list[Action] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_busy(self) -> bool:
        """Whether the agent is currently performing an action."""
        return self.current_action is not None

    def start_action(self, action: Action) -> None:
        """Start performing an action from scratch (progress reset to 0).

        Args:
            action: The Action to perform.

        Raises:
            ValueError: If the agent is already busy. Use interrupt_for() or
                cancel_action() first.
        """
        if self.current_action is not None:
            raise ValueError(
                f"Agent {self.unique_id} is already performing {self.current_action.name!r}. "
                f"Use interrupt_for() or cancel_action() first."
            )

        action.progress = 0.0
        self._resume_action(action)

    def interrupt_for(self, action: Action) -> None:
        """Interrupt the current action and start a new one.

        The interrupted action's on_effect is called with partial reward.
        If its reschedule_on_interrupt is "remainder" or "full" it is pushed
        to the front of the queue to resume later; otherwise it is discarded.

        If the current action is not interruptible the call is silently ignored.
        If the agent is idle, action starts immediately.

        Args:
            action: The Action to perform instead.
        """
        if self.current_action is not None:
            if not self.current_action.interruptible:
                return

            interrupted = (
                self.current_action
            )  # capture before _interrupt_current clears it
            self._interrupt_current()

            if interrupted.reschedule_on_interrupt == "remainder":
                # progress already updated by _interrupt_current — resume from here
                self.action_queue.insert(0, interrupted)
            elif interrupted.reschedule_on_interrupt == "full":
                interrupted.progress = 0.0  # restart from scratch when resumed
                self.action_queue.insert(0, interrupted)
            # False → discard

        self.start_action(action)

    def cancel_action(self, clear_queue: bool = False) -> None:
        """Cancel the current action.

        The action's on_effect is called with the partial reward earned so far.
        If clear_queue is False (default), the next queued action starts
        automatically. If clear_queue is True, the queue is emptied and the
        agent becomes fully idle.

        Args:
            clear_queue: If True, also discard all pending queued actions.
        """
        if self.current_action is not None:
            self._interrupt_current()

        if clear_queue:
            self.action_queue.clear()
        else:
            self._drain_queue()

    def request_action(self, action: Action) -> None:
        """Priority-aware action entry point.

        - Idle agent → starts action immediately.
        - New action priority strictly greater than current action's priority
          AND current action is interruptible → preempts current, starts new.
        - Otherwise → appends action to the back of the queue.

        Args:
            action: The Action to request.
        """
        if not self.is_busy:
            self.start_action(action)
        elif (
            action.priority > self.current_action.priority
            and self.current_action.interruptible
        ):
            self.interrupt_for(action)
        else:
            self.action_queue.append(action)

    def remove(self) -> None:
        """Remove the agent, canceling the current action and clearing the queue."""
        if self.current_action and self.current_action._event:
            self.current_action._event.cancel()
        self.current_action = None
        self.action_queue.clear()
        super().remove()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resume_action(self, action: Action) -> None:
        """Schedule action using its current progress (supports remainder resume).

        Unlike start_action this does NOT reset progress, allowing a "remainder"
        action to continue from where it left off.
        """
        self.current_action = action
        action._started_at = self.model.time

        remaining = action.remaining_time
        if remaining <= 0:
            self._complete_action()
            return

        action._event = self.model.schedule_event(
            self._complete_action, after=remaining
        )

    def _update_progress(self) -> None:
        """Update the current action's progress based on elapsed simulation time."""
        action = self.current_action
        if action and action._started_at is not None and action.duration > 0:
            elapsed = self.model.time - action._started_at
            action.progress = min(1.0, action.progress + elapsed / action.duration)

    def _interrupt_current(self) -> None:
        """Stop the current action, update progress, and fire its on_effect callback."""
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
        """Handle action completion. Called by the Mesa-scheduled event."""
        action = self.current_action
        if action is None:
            return

        action.progress = 1.0
        action._event = None
        action._started_at = None
        self.current_action = None

        if action.on_effect:
            action.on_effect(self, action.effective_completion)

        # Automatically start the next queued action, if any.
        self._drain_queue()

    def _drain_queue(self) -> None:
        """Start the next pending action from the queue if the agent is now idle."""
        if self.action_queue and not self.is_busy:
            next_action = self.action_queue.pop(0)
            self._resume_action(next_action)
