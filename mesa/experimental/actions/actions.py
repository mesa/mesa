"""mesa.experimental.actions: Timed, interruptible actions for Mesa agents.

An Action represents something an agent does over time. It integrates with
Mesa's event scheduling system for precise timing and supports interruption
with progress tracking and optional resumption.

Actions are subclassable: override on_start(), on_complete(),
on_interrupt(), and on_fail() to define behavior.

Example::

    class Forage(Action):
        def __init__(self, sheep):
            super().__init__(sheep, duration=5.0)

        def on_complete(self):
            self.agent.energy += 30

        def on_interrupt(self, progress):
            self.agent.energy += 30 * progress

    sheep.start_action(Forage(sheep))
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
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
    FAILED = auto()


def _as_predicates(
    spec: Callable[[Agent], bool] | Iterable[Callable[[Agent], bool]] | None,
) -> list[Callable[[Agent], bool]]:
    """Normalise a requirement spec into a list the action owns."""
    if spec is None:
        return []
    if callable(spec):
        return [spec]
    return list(spec)


class Action:
    """Something an agent does over time.

    Actions have a duration, can be interrupted, and track their own
    lifecycle state. They integrate with Mesa's event scheduler for
    completion timing.

    Subclass and override on_start/on_complete/on_interrupt for complex
    behavior. All hooks default to doing nothing (pass).

    Attributes:
        agent: The agent performing this action.
        model: The model (shortcut for agent.model).
        name: Human-readable identifier. Defaults to the class name.
        duration: Time to complete. May be a callable(agent) -> float
            for state-dependent duration, resolved at start time.
        priority: Importance level. Higher = more important. May be a
            callable(agent) -> float, resolved at start time.
        interruptible: Whether higher-priority actions can preempt this.
        start_requirements: Predicates that must hold for the action to start.
        completion_requirements: Predicates that must hold for the effect to
            apply. Empty by default.
        state: Current lifecycle state (PENDING, ACTIVE, COMPLETED,
            INTERRUPTED, FAILED).
        progress: Time fraction completed, 0.0 to 1.0. Computed live
            while the action is active.

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
        name: str | None = None,
        priority: float | Callable[[Agent], float] = 0.0,
        interruptible: bool = True,
        start_requirements: Callable[[Agent], bool]
        | Iterable[Callable[[Agent], bool]]
        | None = None,
        completion_requirements: Callable[[Agent], bool]
        | Iterable[Callable[[Agent], bool]]
        | None = None,
    ) -> None:
        """Initialize an Action.

        Args:
            agent: The agent that will perform this action.
            duration: Time to complete. Either a float or a callable
                that receives the agent and returns a float. Resolved
                when start() is called.
            name: Human-readable name. Defaults to the class name.
            priority: Importance level for interruption decisions. Either
                a float or a callable that receives the agent and returns
                a float. Resolved when start() is called.
            interruptible: If False, interrupt() will fail and return False.
            start_requirements: A single callable(agent) -> bool, or an
                iterable of them. All must hold for the action to start.
            completion_requirements: The same, checked instead when the action
                completes, gating the effect rather than the attempt.
        """
        self.agent = agent
        self.model = agent.model
        self.interruptible = interruptible
        self._name: str | None = name

        self.start_requirements: list[Callable[[Agent], bool]] = _as_predicates(
            start_requirements
        )
        self.completion_requirements: list[Callable[[Agent], bool]] = _as_predicates(
            completion_requirements
        )

        # Store raw values (may be callables, resolved at start)
        self._duration_spec = duration
        self._priority_spec = priority

        # Resolved values (set in start())
        self.duration: float = 0.0
        self.priority: float = 0.0

        # Lifecycle state
        self.state: ActionState = ActionState.PENDING
        self._progress: float = 0.0
        self._start_time: float = -1.0
        self._event: Event | None = None

    # --- Properties ---

    @property
    def name(self) -> str:
        """Human-readable name. Returns the class name by default.

        Can be set via __init__(name=...), per-instance assignment
        (``action.name = "dig"``), or overridden in subclasses.
        """
        return self._name if self._name is not None else self.__class__.__name__

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def progress(self) -> float:
        """Time fraction completed, 0.0 to 1.0.

        Computed live while the action is active. For interrupted or
        completed actions, returns the final progress value.
        """
        if (
            self.state is ActionState.ACTIVE
            and self.duration > 0
            and self._start_time >= 0
        ):
            elapsed_this_attempt = self.model.time - self._start_time
            return min(1.0, self._progress + elapsed_this_attempt / self.duration)
        return self._progress

    @property
    def remaining_time(self) -> float:
        """Time remaining until completion.

        Computed live while active. For interrupted actions, returns
        the time that would remain if resumed.
        """
        return self.duration * (1.0 - self.progress)

    @property
    def elapsed_time(self) -> float:
        """Total time spent on this action so far (across all attempts)."""
        return self.duration * self.progress

    @property
    def is_resumable(self) -> bool:
        """Whether this action can be resumed (interrupted, not completed)."""
        return self.state is ActionState.INTERRUPTED and self._progress < 1.0

    @property
    def has_failed(self) -> bool:
        """Whether a requirement did not hold, at start or at completion."""
        return self.state is ActionState.FAILED

    # --- Lifecycle methods (override in subclasses) ---

    def on_start(self) -> None:
        """Called when the action starts executing for the first time.

        Override for setup logic (e.g., logging, animation triggers,
        resource reservation). Not called on resume — see on_resume().
        """

    def on_resume(self) -> None:
        """Called when the action resumes after interruption.

        Override to handle resumption differently from first start
        (e.g., logging "resumed" instead of "started", skipping
        setup that shouldn't happen twice).

        Default implementation calls on_start().
        """
        self.on_start()

    def on_complete(self) -> None:
        """Called when the action finishes normally.

        Override to apply the action's full effect (e.g., gaining
        energy, completing a transaction).
        """

    def on_interrupt(self, progress: float) -> None:
        """Called when the action is interrupted before completion.

        Override to handle partial completion. The progress parameter
        is the raw time fraction (0.0 to 1.0), giving you full control
        over how partial work translates to partial effect.

        Args:
            progress: Fraction of duration completed (elapsed / duration).

        Notes:
            Also called by cancel(). If you need to distinguish
            interruption from cancellation, check self.interruptible:
            a non-interruptible action that receives on_interrupt was
            necessarily cancelled, not interrupted.
        """

    def on_fail(self) -> None:
        """Called when a requirement does not hold.

        Override to release anything the action
        reserved, or to record the attempt.

        Notes:
            Check self.progress to tell the two failures apart: it is 1.0
            when the action ran its full duration and only then found the
            requirement broken, and below 1.0 when the action never got to
            run this attempt.
        """

    # --- Execution (called by Agent, not typically by users) ---

    def start(self) -> Action:
        """Start executing this action (or resume from interruption).

        On first start (PENDING): resolves callable duration/priority,
        starts from progress=0. On resume (INTERRUPTED): continues from
        existing progress with remaining duration.

        If a requirement does not hold the action moves to FAILED instead,
        firing on_fail() rather than on_start() or on_resume(). Inspect
        state after starting rather than assuming the action is ACTIVE.

        Returns:
            Self, for chaining.

        Raises:
            ValueError: If the action is not in PENDING or INTERRUPTED state.
                A FAILED action cannot be restarted; create a new one.
        """
        resuming = self.state is ActionState.INTERRUPTED

        if self.state not in (ActionState.PENDING, ActionState.INTERRUPTED):
            raise ValueError(
                f"Cannot start action in {self.state.name} state. "
                f"Only PENDING or INTERRUPTED actions can be started."
            )

        # Gate entry before duration and priority are resolved, so a failing
        # action neither fires on_start() nor schedules a completion event.
        if not self._requirements_met(self.start_requirements):
            self._fail()
            return self

        # Resolve callables on first start only
        if not resuming:
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

        if resuming:
            self.on_resume()
        else:
            self.on_start()

        # Calculate remaining time
        remaining = self.duration * (1.0 - self._progress)

        # Instantaneous actions (or fully completed) complete immediately
        if remaining <= 0:
            self._do_complete()
            return self

        # Schedule completion event for remaining duration
        self._event = self.model.schedule_event(self._do_complete, after=remaining)
        return self

    def interrupt(self) -> bool:
        """Interrupt this action.

        Updates progress, fires on_interrupt with the time fraction,
        and cancels the scheduled completion event. The action moves
        to INTERRUPTED state and can be resumed later with start().

        Returns:
            True if the action was interrupted, False if it could not
            be interrupted (non-interruptible or not active).
        """
        if self.state is not ActionState.ACTIVE:
            return False

        if not self.interruptible:
            return False

        self._freeze_progress()
        self._cancel_event()

        self.state = ActionState.INTERRUPTED

        if self.agent.current_action is self:
            self.agent.current_action = None

        self.on_interrupt(self._progress)
        return True

    def cancel(self) -> bool:
        """Cancel this action, ignoring the interruptible flag.

        Like interrupt(), but always succeeds for active actions and
        moves to INTERRUPTED state. The action can still be resumed
        if desired.

        Returns:
            True if the action was cancelled, False if not active.
        """
        if self.state is not ActionState.ACTIVE:
            return False

        self._freeze_progress()
        self._cancel_event()

        self.state = ActionState.INTERRUPTED

        if self.agent.current_action is self:
            self.agent.current_action = None

        self.on_interrupt(self._progress)
        return True

    # --- Internal ---

    def _freeze_progress(self) -> None:
        """Snapshot live progress into _progress for storage."""
        if self.duration > 0 and self._start_time >= 0:
            elapsed_this_attempt = self.model.time - self._start_time
            self._progress = min(
                1.0, self._progress + elapsed_this_attempt / self.duration
            )
        else:
            self._progress = 1.0

    def _cancel_event(self) -> None:
        """Cancel the scheduled completion event if it exists."""
        if self._event is not None:
            self._event.cancel()
            self._event = None

    def _requirements_met(self, requirements: list[Callable[[Agent], bool]]) -> bool:
        """Whether every requirement in the given list holds right now."""
        return all(requirement(self.agent) for requirement in requirements)

    def _fail(self) -> None:
        """Move to FAILED, release the agent, and fire on_fail()."""
        self.state = ActionState.FAILED

        if self.agent.current_action is self:
            self.agent.current_action = None

        self.on_fail()

    def _do_complete(self) -> None:
        """Handle normal completion. Called by the scheduled event."""
        if self.state is not ActionState.ACTIVE:
            return

        # Progress reaches 1.0 even if a requirement broke: the full duration
        # elapsed, only the effect is withheld.
        self._progress = 1.0
        self._event = None

        if not self._requirements_met(self.completion_requirements):
            self._fail()
            return

        self.state = ActionState.COMPLETED

        # Clear agent's reference so it's no longer busy
        if self.agent.current_action is self:
            self.agent.current_action = None

        self.on_complete()

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"{self.name}(state={self.state.name}, "
            f"progress={self.progress:.0%}, "
            f"duration={self.duration})"
        )
