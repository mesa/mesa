"""mesa.experimental.actions: Timed, interruptible actions for Mesa agents.

An Action represents something an agent does over time. It integrates with
Mesa's event scheduling system for precise timing and supports interruption
with progress tracking.

Actions are subclassable: override on_start(), on_complete(), and
on_interrupt() to define behavior. For simple cases, pass callables directly.

Example::

    # Subclass approach (complex actions)
    class Forage(Action):
        def __init__(self, sheep):
            super().__init__(sheep, duration=5.0)

        def on_complete(self):
            self.agent.energy += 30

        def on_interrupt(self, progress):
            self.agent.energy += 30 * progress

    sheep.start_action(Forage(sheep))

    # Inline approach (simple actions)
    action = Action(agent, duration=2.0, on_complete=lambda: agent.energy += 10)
    agent.start_action(action)
"""

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mesa.agent import Agent
    from mesa.time import Event


class ActionState(IntEnum):
    """Lifecycle state of an Action."""

    PENDING = auto()
    ACTIVE = auto()
    COMPLETED = auto()
    INTERRUPTED = auto()


class Action:
    """Something an agent does over time.

    Actions have a duration, can be interrupted, and track their own
    lifecycle state. They integrate with Mesa's event scheduler for
    completion timing.

    Subclass and override on_start/on_complete/on_interrupt for complex
    behavior, or pass callables for simple cases.

    Attributes:
        agent: The agent performing this action.
        model: The model (shortcut for agent.model).
        duration: Time to complete. May be a callable(agent) -> float
            for state-dependent duration, resolved at start time.
        priority: Importance level. Higher = more important. May be a
            callable(agent) -> float, resolved at start time.
        interruptible: Whether higher-priority actions can preempt this.
        state: Current lifecycle state (PENDING, ACTIVE, COMPLETED, INTERRUPTED).
        progress: Time fraction completed, 0.0 to 1.0.

    Notes:
        Actions hold a reference to their agent, mirroring how agents
        reference their model. This allows actions to query and modify
        agent state directly in their hooks.
    """

    def __init__(
        self,
        agent: Agent,
        duration: float | Callable[[Agent], float] = 1.0,
        *,
        priority: float | Callable[[Agent], float] = 0.0,
        interruptible: bool = True,
        on_start: Callable[[], None] | None = None,
        on_complete: Callable[[], None] | None = None,
        on_interrupt: Callable[[float], None] | None = None,
    ) -> None:
        """Initialize an Action.

        Args:
            agent: The agent that will perform this action.
            duration: Time to complete. Either a float or a callable
                that receives the agent and returns a float. Resolved
                when start() is called.
            priority: Importance level for interruption decisions. Either
                a float or a callable that receives the agent and returns
                a float. Resolved when start() is called.
            interruptible: If False, interrupt() will fail and return False.
            on_start: Optional callback, called when the action starts.
                Ignored if the subclass overrides on_start().
            on_complete: Optional callback, called when the action finishes.
                Ignored if the subclass overrides on_complete().
            on_interrupt: Optional callback(progress), called when interrupted.
                Ignored if the subclass overrides on_interrupt().
        """
        self.agent = agent
        self.model = agent.model
        self.interruptible = interruptible

        # Store raw values (may be callables, resolved at start)
        self._duration_spec = duration
        self._priority_spec = priority

        # Resolved values (set in start())
        self.duration: float = 0.0
        self.priority: float = 0.0

        # Optional inline callbacks
        self._on_start_fn = on_start
        self._on_complete_fn = on_complete
        self._on_interrupt_fn = on_interrupt

        # Lifecycle state
        self.state: ActionState = ActionState.PENDING
        self.progress: float = 0.0
        self._start_time: float = -1.0
        self._event: Event | None = None

    # --- Lifecycle methods (override in subclasses) ---

    def on_start(self) -> None:
        """Called when the action starts executing.

        Override for setup logic (e.g., logging, animation triggers,
        resource reservation).
        """
        if self._on_start_fn is not None:
            self._on_start_fn()

    def on_complete(self) -> None:
        """Called when the action finishes normally.

        Override to apply the action's full effect (e.g., gaining
        energy, completing a transaction).
        """
        if self._on_complete_fn is not None:
            self._on_complete_fn()

    def on_interrupt(self, progress: float) -> None:
        """Called when the action is interrupted before completion.

        Override to handle partial completion. The progress parameter
        is the raw time fraction (0.0 to 1.0), giving you full control
        over how partial work translates to partial effect.

        Args:
            progress: Fraction of duration completed (elapsed / duration).
        """
        if self._on_interrupt_fn is not None:
            self._on_interrupt_fn(progress)

    # --- Execution (called by Agent, not typically by users) ---

    def start(self) -> Action:
        """Start executing this action.

        Resolves callable duration/priority, schedules the completion
        event, and calls on_start().

        Returns:
            Self, for chaining.

        Raises:
            ValueError: If the action is not in PENDING state.
        """
        if self.state is not ActionState.PENDING:
            raise ValueError(
                f"Cannot start action in {self.state.name} state. "
                f"Only PENDING actions can be started."
            )

        # Resolve callables
        self.duration = (
            self._duration_spec(self.agent)
            if callable(self._duration_spec)
            else self._duration_spec
        )
        self.priority = (
            self._priority_spec(self.agent)
            if callable(self._priority_spec)
            else self._priority_spec
        )

        if self.duration < 0:
            raise ValueError(f"Action duration must be >= 0, got {self.duration}")

        self._start_time = self.model.time
        self.state = ActionState.ACTIVE
        self.on_start()

        # Instantaneous actions complete immediately
        if self.duration == 0:
            self._do_complete()
            return self

        # Schedule completion event
        self._event = self.model.schedule_event(self._do_complete, after=self.duration)
        return self

    def interrupt(self) -> bool:
        """Interrupt this action.

        Updates progress, fires on_interrupt with the time fraction,
        and cancels the scheduled completion event.

        Returns:
            True if the action was interrupted, False if it could not
            be interrupted (non-interruptible or not active).
        """
        if self.state is not ActionState.ACTIVE:
            return False

        if not self.interruptible:
            return False

        self._update_progress()
        self._cancel_event()

        self.state = ActionState.INTERRUPTED
        self.on_interrupt(self.progress)
        return True

    def cancel(self) -> bool:
        """Cancel this action, ignoring the interruptible flag.

        Like interrupt(), but always succeeds for active actions.
        Useful for cleanup (e.g., agent removal).

        Returns:
            True if the action was cancelled, False if not active.
        """
        if self.state is not ActionState.ACTIVE:
            return False

        self._update_progress()
        self._cancel_event()

        self.state = ActionState.INTERRUPTED
        self.on_interrupt(self.progress)
        return True

    # --- Queries ---

    @property
    def remaining_time(self) -> float:
        """Time remaining until completion. Negative if already done."""
        if self.state is ActionState.ACTIVE and self._start_time >= 0:
            elapsed = self.model.time - self._start_time
            return self.duration - elapsed
        return 0.0

    @property
    def elapsed_time(self) -> float:
        """Time elapsed since the action started."""
        if self._start_time >= 0:
            return self.model.time - self._start_time
        return 0.0

    # --- Internal ---

    def _update_progress(self) -> None:
        """Update progress based on elapsed time."""
        if self.duration > 0 and self._start_time >= 0:
            self.progress = min(
                1.0, (self.model.time - self._start_time) / self.duration
            )
        else:
            self.progress = 1.0

    def _cancel_event(self) -> None:
        """Cancel the scheduled completion event if it exists."""
        if self._event is not None:
            self._event.cancel()
            self._event = None

    def _do_complete(self) -> None:
        """Handle normal completion. Called by the scheduled event."""
        if self.state is not ActionState.ACTIVE:
            return

        self.progress = 1.0
        self._event = None
        self.state = ActionState.COMPLETED
        self.on_complete()

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"Action(state={self.state.name}, "
            f"progress={self.progress:.0%}, "
            f"duration={self.duration})"
        )
