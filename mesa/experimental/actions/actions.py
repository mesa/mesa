"""mesa.experimental.actions: Timed, interruptible actions for Mesa agents.

An Action represents something an agent does over time. It integrates with
Mesa's event scheduling system for precise timing and supports interruption
with progress tracking and optional resumption.

Actions are subclassable: override on_start(), on_complete(),
on_interrupt(), and on_fail() to define behavior. The agent-side API
lives on the HasActions mixin; when an action ends, the agent's
on_idle() hook fires, and decisions about what to do next -- chain
another action, resume an interrupted one -- belong there.

Example::

    class Sheep(HasActions, Agent):
        pass

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

from mesa.mesa_logging import create_module_logger
from mesa.time import Priority

if TYPE_CHECKING:
    from mesa.agent import Agent
    from mesa.model import Model
    from mesa.time import Event

_mesa_logger = create_module_logger()


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
            callable(agent) -> float, resolved once: at start, or earlier
            if HasActions.interrupt_for weighs it against a running action.
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
                a float. Resolved once, at start() or when interrupt_for
                first consults the agent's should_interrupt with it.
            interruptible: If False, interrupt() will fail and return False.
            start_requirements: A single callable(agent) -> bool, or an
                iterable of them. All must hold for the action to start.
            completion_requirements: The same, checked instead when the action
                completes, gating the effect rather than the attempt.

        Raises:
            TypeError: If the agent does not inherit from HasActions.
        """
        if not isinstance(agent, HasActions):
            raise TypeError(
                f"Action requires an agent with action support; make "
                f"{type(agent).__name__} inherit from "
                f"mesa.experimental.actions.HasActions."
            )
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

        # Resolved values (set in start(); priority possibly earlier, see
        # _resolve_priority)
        self.duration: float = 0.0
        self.priority: float = 0.0
        self._priority_resolved: bool = False

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
            self._resolve_priority()

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
        self._release_agent()
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
        self._release_agent()
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

    def _release_agent(self) -> None:
        """Clear the agent's slot and schedule its idle wake.

        Completion, interruption, cancellation, and failure all funnel
        through here, so the wake contract has a single path. An action
        that never held the slot (started directly, without
        start_action()) wakes nobody. HasActions.remove() is the one end
        that does not come through here: it abandons the action, and a
        dying agent must not wake.
        """
        if self.agent.current_action is self:
            self.agent.current_action = None
            self.agent._schedule_wake(self)

    def _resolve_priority(self) -> None:
        """Resolve the priority spec against the agent, at most once.

        Called from start(), and from HasActions.interrupt_for before it consults
        should_interrupt, since the incoming action has not started yet and
        its priority would otherwise still read 0.0. A callable spec is
        called at most once either way.
        """
        if self._priority_resolved:
            return
        self.priority = (
            self._priority_spec(self.agent)
            if callable(self._priority_spec)
            else self._priority_spec
        )
        self._priority_resolved = True

    def _requirements_met(self, requirements: list[Callable[[Agent], bool]]) -> bool:
        """Whether every requirement in the given list holds right now."""
        return all(requirement(self.agent) for requirement in requirements)

    def _fail(self) -> None:
        """Move to FAILED, release the agent, and fire on_fail()."""
        self.state = ActionState.FAILED
        self._release_agent()
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
        self._release_agent()
        self.on_complete()

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"{self.name}(state={self.state.name}, "
            f"progress={self.progress:.0%}, "
            f"duration={self.duration})"
        )


class HasActions:
    """Mixin that lets an agent perform Actions.

    An agent performs at most one action at a time, held in
    ``current_action``. Actions are started with start_action(),
    preempted through interrupt_for() under the should_interrupt()
    policy, and force-stopped with cancel_action(). Whenever an action
    ends and nothing replaces it, the on_idle() wake fires.

    Action instances require their agent to inherit from this mixin.
    """

    if TYPE_CHECKING:
        # Supplied by the Agent class this mixin is combined with.
        model: Model
        unique_id: int

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the action slot."""
        self.current_action: Action | None = None
        self._wake_event: Event | None = None
        self._wake_previous: Action | None = None
        self._last_wake_time: float = float("-inf")
        super().__init__(*args, **kwargs)

    def remove(self) -> None:
        """Remove the agent, abandoning its action and pending wake.

        If the agent is currently performing an action, the action's
        scheduled completion event is cancelled silently. The action's
        on_interrupt() callback is NOT fired, because the agent is being
        destroyed — not making a behavioral decision. The action moves
        to no defined end state; it is simply abandoned. A pending
        on_idle wake is cancelled as well: a removed agent never wakes.

        If your action holds external resources (e.g., a Resource slot,
        a reservation, a lock), override remove() and call
        self.cancel_action() before super().remove() to ensure
        on_interrupt() fires and cleanup logic runs:

            def remove(self):
                self.cancel_action()  # Fires on_interrupt for cleanup
                super().remove()

        Notes:
            This is a deliberate design choice. The default silent
            cleanup is safe and avoids callbacks touching agent state
            during teardown. Models that need cleanup should opt in
            explicitly.
        """
        if self.current_action is not None:
            self.current_action._cancel_event()  # Silent cleanup, no callback
            self.current_action = None

        if self._wake_event is not None:
            self._wake_event.cancel()
            self._wake_event = None
            self._wake_previous = None

        super().remove()

    def start_action(self, action: Action) -> Action:
        """Start performing an action.

        The action must be in PENDING or INTERRUPTED state and the agent
        must not be currently performing another action.

        If one of the action's start requirements does not hold, the action moves
        to FAILED instead of starting and the agent stays idle. Check
        action.has_failed rather than assuming the action is running.

        Args:
            action: The Action to perform. Must have been created with
                this agent as its agent.

        Returns:
            The started Action.

        Raises:
            ValueError: If the agent is already performing an action,
                or if the action doesn't belong to this agent.
        """
        if self.current_action is not None:
            raise ValueError(
                f"Agent {self.unique_id} is already performing an action "
                f"({self.current_action!r}). Use interrupt_for() or "
                f"cancel_action() first."
            )

        if action.agent is not self:
            raise ValueError(
                f"Action's agent (id={action.agent.unique_id}) does not match "
                f"this agent (id={self.unique_id})."
            )

        self.current_action = action
        action.start()

        # If the action completed instantly (duration=0), start() already
        # called _do_complete which cleared current_action via the Action.
        return action

    def should_interrupt(self, current: Action, incoming: Action) -> bool:
        """Decide whether an incoming action may preempt the current one.

        Consulted by interrupt_for() whenever the agent is busy, with both
        priorities already resolved. Override to encode preemption policy,
        e.g. comparing action names or agent state instead of priorities.

        Args:
            current: The action the agent is performing.
            incoming: The action that wants to replace it.

        Returns:
            True to attempt the interruption, False to refuse it.

        Notes:
            Returning True cannot override the interruptible flag; use
            cancel_action() to force. This hook decides policy, the flag
            stays a hard property of the action.
        """
        return current.interruptible and incoming.priority >= current.priority

    def interrupt_for(self, new_action: Action) -> bool:
        """Interrupt the current action and start a new one.

        If there is no current action, simply starts the new one. Otherwise
        should_interrupt(current, incoming) decides whether to preempt.

        Args:
            new_action: The Action to perform instead.

        Returns:
            True if the new action was started. False if should_interrupt
            refused, the current action is non-interruptible, or the new
            action failed its start requirements.

        Notes:
            The False cases differ in what they leave behind. A refusal
            changes nothing. A failed requirement does not roll the
            interruption back: the old action is already INTERRUPTED and
            the agent is left idle, since whether to resume it is the
            model's decision.
        """
        if self.current_action is not None:
            new_action._resolve_priority()
            if not self.should_interrupt(self.current_action, new_action):
                return False
            if not self.current_action.interrupt():
                return False
                # interrupt() already cleared current_action

        self.start_action(new_action)
        return not new_action.has_failed

    def cancel_action(self) -> bool:
        """Cancel the current action, ignoring interruptible flag.

        Calls on_interrupt with partial progress. Returns False only if
        there is no current action.

        Returns:
            True if an action was cancelled, False if idle.
        """
        if self.current_action is None:
            return False

        self.current_action.cancel()
        # cancel() already cleared current_action
        return True

    @property
    def is_busy(self) -> bool:
        """Whether the agent is currently performing an action."""
        return self.current_action is not None

    def on_idle(self, previous: Action | None) -> None:
        """Called when the agent's action has ended and nothing replaced it.

        Fires for every way an action can end -- completion, interruption,
        cancellation, failure -- as a zero-delay ``Priority.LOW`` event
        rather than synchronously, so every agent finishing at time t
        decides after all completions at t have run.

        Override it to choose what to do next, typically by starting or
        resuming an action. The default does nothing, and nothing is
        resumed automatically: an interrupted action stays interrupted
        unless this hook (or other model code) restarts it.

        Args:
            previous: The action that just ended, in its final state
                (COMPLETED, INTERRUPTED, or FAILED). None is reserved for
                wakes not caused by an action ending; no built-in trigger
                sends it today.

        Notes:
            At most one wake fires per agent per model time, so a
            zero-duration action started here cannot re-trigger the hook
            in the same instant; a wake dropped by this guard is logged
            at DEBUG level. The wake is skipped when the agent is
            busy again by the time the event runs -- interrupt_for()
            refills the slot synchronously, so no idle gap ever existed.
            If several actions end before the wake runs, they coalesce
            into one call and ``previous`` is the latest of them.
        """

    def _schedule_wake(self, previous: Action | None) -> None:
        """Queue the deferred on_idle event, coalescing repeats.

        Called by Action._release_agent whenever this agent's slot is
        cleared. If a wake is already pending, or one already fired at
        the current model time, no second event is queued -- only
        ``previous`` is brought up to date. A wake dropped by the
        once-per-time guard is logged at DEBUG level.
        """
        self._wake_previous = previous
        if self._wake_event is not None:
            return
        if self._last_wake_time == self.model.time:
            _mesa_logger.debug(
                f"suppressed repeat on_idle wake for agent {self.unique_id} "
                f"at time {self.model.time} (ending action: {previous!r})"
            )
            return
        self._wake_event = self.model.schedule_event(
            self._fire_wake, after=0.0, priority=Priority.LOW
        )

    def _fire_wake(self) -> None:
        """Run the pending wake: call on_idle if the agent is still free."""
        previous = self._wake_previous
        self._wake_event = None
        self._wake_previous = None
        if self.current_action is not None:
            return
        # Recorded only when on_idle actually runs, so a busy skip does not
        # suppress a later release at the same model time.
        self._last_wake_time = self.model.time
        self.on_idle(previous)
