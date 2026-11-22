"""mesa.experimental.actions: Timed, interruptible actions for Mesa agents.

An Action represents something an agent does over time. It integrates with
Mesa's event scheduling system for precise timing and supports interruption
with progress tracking and optional resumption.

The words follow one convention: ``interrupt`` stops an action temporarily,
``resume`` continues it, and ``cancel`` ends it for good. Agents start
actions with ``agent.start_action()``; there is no public ``start()``.

Actions are subclassable: override on_start(), on_complete(),
on_interrupt(), on_fail() and on_cancel() to define behavior. The
agent-side API lives on the HasActions mixin; when an action ends, the
agent's on_idle() hook fires, and decisions about what to do next --
chain another action, resume an interrupted one -- belong there.

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

import math
import warnings
from collections.abc import Callable, Iterable
from enum import IntEnum, auto
from typing import TYPE_CHECKING

from mesa.time import Priority

if TYPE_CHECKING:
    from mesa.agent import Agent
    from mesa.model import Model
    from mesa.time import Event


class ActionState(IntEnum):
    """Lifecycle state of an Action.

    COMPLETED, FAILED and CANCELED are final: an action never leaves them.
    """

    PENDING = auto()
    ACTIVE = auto()
    COMPLETED = auto()
    INTERRUPTED = auto()
    FAILED = auto()
    CANCELED = auto()


_FINAL_STATES = frozenset(
    {ActionState.COMPLETED, ActionState.FAILED, ActionState.CANCELED}
)


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
            for state-dependent duration, resolved at the first start.
        priority: Importance level. Higher = more important. May be a
            callable(agent) -> float, resolved once: at the first start, or
            earlier if HasActions.interrupt_for weighs it against a running
            action.
        interruptible: Whether higher-priority actions can preempt this.
        start_requirements: Predicates that must hold for the action to start.
        completion_requirements: Predicates that must hold for the effect to
            apply. Empty by default.
        state: Current lifecycle state (read-only).
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
                when the action first starts; must be finite and >= 0.
            name: Human-readable name. Defaults to the class name.
            priority: Importance level for interruption decisions. Either
                a float or a callable that receives the agent and returns
                a float. Resolved once, at the first start or when
                interrupt_for first consults the agent's should_interrupt
                with it; must not be NaN.
            interruptible: If False, interrupt() refuses unless forced.
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

        # Store raw values (may be callables, resolved at the first start)
        self._duration_spec = duration
        self._priority_spec = priority

        # Resolved values (set at the first start; priority possibly earlier,
        # see _resolve_priority)
        self.duration: float = 0.0
        self.priority: float = 0.0
        self._priority_resolved: bool = False

        # Lifecycle state
        self._state: ActionState = ActionState.PENDING
        self._progress: float = 0.0
        self._start_time: float = -1.0
        # When the current attempt reaches full duration; see _is_due().
        self._deadline: float = math.inf
        self._event: Event | None = None
        # Counts starts and resumes, so callers can tell whether a hook
        # started a new attempt while they were not looking.
        self._attempts: int = 0

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
    def state(self) -> ActionState:
        """Current lifecycle state. Read-only: use the lifecycle methods."""
        return self._state

    @property
    def progress(self) -> float:
        """Time fraction completed, 0.0 to 1.0.

        Computed live while the action is active, and exactly 1.0 from its
        deadline on. For interrupted or ended actions, returns the progress
        at the moment they stopped.
        """
        if self._state is ActionState.ACTIVE:
            if self._is_due():
                return 1.0
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
        """Whether resume() can continue this action: it is INTERRUPTED."""
        return self._state is ActionState.INTERRUPTED

    @property
    def has_failed(self) -> bool:
        """Whether a requirement did not hold, at start or at completion."""
        return self._state is ActionState.FAILED

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

    def on_cancel(self) -> None:
        """Called when the action is canceled.

        Fires once, whether the action was ACTIVE, INTERRUPTED or still
        PENDING. Override to release what the action held across
        interruptions. Read self.progress for how much work was done.
        """

    # --- Lifecycle ---

    def resume(self) -> bool:
        """Continue an INTERRUPTED action for its remaining duration.

        Start requirements are checked again; if one does not hold, the
        action moves to FAILED and on_fail() fires instead of on_resume().

        Returns:
            True if the action became ACTIVE. False if it already was
            ACTIVE, or if a start requirement failed (it is now FAILED).

        Raises:
            ValueError: If the action is PENDING (use agent.start_action())
                or in a final state, or if its agent is busy with another
                action.
        """
        if self._state is ActionState.ACTIVE:
            return False
        if self._state is not ActionState.INTERRUPTED:
            hint = (
                "Use agent.start_action() to start a PENDING action."
                if self._state is ActionState.PENDING
                else "A final action can't restart; create a new one."
            )
            raise ValueError(
                f"Cannot resume an action in {self._state.name} state. {hint}"
            )
        self.agent._check_idle()
        return self._enter()

    def interrupt(self, *, force: bool = False) -> bool:
        """Stop this action temporarily (ACTIVE -> INTERRUPTED).

        Freezes progress, cancels the scheduled completion and fires
        on_interrupt() with the progress so far. Continue it later with
        resume().

        If the action's deadline has been reached, it completes instead
        (completion wins), even if it is not interruptible.

        Args:
            force: Interrupt even if the action is not interruptible.

        Returns:
            True if the action was interrupted. False if it is not ACTIVE,
            if it was due and completed instead, or if it is not
            interruptible and force is not set.
        """
        if self._state is not ActionState.ACTIVE:
            return False
        if self._is_due():
            self._complete()
            return False
        if not (self.interruptible or force):
            return False

        self._freeze_progress()
        self._cancel_event()
        self._state = ActionState.INTERRUPTED
        self._release_agent()
        self.on_interrupt(self._progress)
        return True

    def cancel(self) -> bool:
        """End this action for good (-> CANCELED) and fire on_cancel().

        Works from ACTIVE, INTERRUPTED and PENDING, and ignores
        ``interruptible``: that flag governs preemption, not the owner
        ending its own action. An ACTIVE action frees the agent's slot. An
        ACTIVE action whose deadline has been reached completes instead
        (completion wins).

        Returns:
            True if the action was canceled. False if it was already final,
            or if it was due and completed instead.
        """
        if self._state in _FINAL_STATES:
            return False

        if self._state is ActionState.ACTIVE:
            if self._is_due():
                self._complete()
                return False
            self._freeze_progress()
            self._cancel_event()
            self._state = ActionState.CANCELED
            self._release_agent()
        else:
            self._state = ActionState.CANCELED
        self.on_cancel()
        return True

    # --- Internal ---

    def _enter(self) -> bool:
        """Start a PENDING action or resume an INTERRUPTED one.

        The caller has checked the owner, the state and that the agent is
        idle. Returns False if a start requirement failed.
        """
        resuming = self._state is ActionState.INTERRUPTED

        # Gate entry before duration and priority are resolved, so a failing
        # action neither fires on_start() nor schedules a completion event.
        if not self._requirements_met(self.start_requirements):
            self._fail()
            # The slot was never claimed, so _release_agent() did not wake.
            self.agent._schedule_wake(self)
            return False

        if not resuming:
            duration = self._resolve_duration()
            self._resolve_priority()
            self.duration = duration

        now = self.model.time
        self._start_time = now
        self._deadline = now + self.duration * (1.0 - self._progress)
        self._state = ActionState.ACTIVE
        self._attempts += 1
        attempt = self._attempts
        self.agent._current_action = self

        # Schedule the completion before the hook runs: a hook that
        # interrupts, cancels or resumes this action then goes through the
        # normal methods, which cancel or replace this event.
        if self._deadline > now:
            self._event = self.model.schedule_event(
                self._do_complete, at=self._deadline
            )

        if resuming:
            self.on_resume()
        else:
            self.on_start()

        # An instant action completes here, unless the hook ended it or
        # started a new attempt of it.
        if (
            self._deadline <= now
            and self._attempts == attempt
            and self._state is ActionState.ACTIVE
        ):
            self._complete()
        return True

    def _is_due(self) -> bool:
        """Whether the current attempt has reached its deadline.

        Decided on time rather than on accumulated progress, which can sit
        just below 1.0 at the deadline after a few interruptions.
        """
        return self.model.time >= self._deadline

    def _freeze_progress(self) -> None:
        """Snapshot live progress into _progress for storage."""
        self._progress = self.progress

    def _cancel_event(self) -> None:
        """Cancel the scheduled completion event if it exists."""
        if self._event is not None:
            self._event.cancel()
            self._event = None

    def _release_agent(self) -> None:
        """Clear the agent's slot and schedule its idle wake.

        Completion, interruption, cancellation, and failure all funnel
        through here, so the wake contract has a single path. An action
        no longer in the slot, because its agent was removed, wakes
        nobody. HasActions.remove() is the one end that does not come
        through here: it abandons the action, and a dying agent must not
        wake.
        """
        if self.agent._current_action is self:
            self.agent._current_action = None
            self.agent._schedule_wake(self)

    def _resolve_duration(self) -> float:
        """Resolve the duration spec against the agent and validate it."""
        duration = (
            self._duration_spec(self.agent)
            if callable(self._duration_spec)
            else self._duration_spec
        )
        if not (math.isfinite(duration) and duration >= 0):
            raise ValueError(
                f"Action duration must be finite and >= 0, got {duration}."
            )
        return duration

    def _resolve_priority(self) -> None:
        """Resolve the priority spec against the agent, at most once.

        Called at the first start, and from HasActions.interrupt_for before
        it consults should_interrupt, since the incoming action has not
        started yet and its priority would otherwise still read 0.0. A
        callable spec is called at most once either way, unless it returned
        an invalid value.
        """
        if self._priority_resolved:
            return
        priority = (
            self._priority_spec(self.agent)
            if callable(self._priority_spec)
            else self._priority_spec
        )
        if math.isnan(priority):
            raise ValueError(
                "Action priority must not be NaN, since NaN compares False "
                "with every other priority. Use math.inf or -math.inf for "
                "'always' or 'never' preempt."
            )
        self.priority = priority
        self._priority_resolved = True

    def _requirements_met(self, requirements: list[Callable[[Agent], bool]]) -> bool:
        """Whether every requirement in the given list holds right now."""
        return all(requirement(self.agent) for requirement in requirements)

    def _fail(self) -> None:
        """Move to FAILED, release the agent, and fire on_fail()."""
        self._state = ActionState.FAILED
        self._release_agent()
        self.on_fail()

    def _do_complete(self) -> None:
        """Complete the action. Called by the scheduled completion event."""
        if self._state is not ActionState.ACTIVE:
            return
        self._event = None  # it is executing, so there is nothing to cancel
        self._complete()

    def _complete(self) -> None:
        """Finish an ACTIVE action that has reached its deadline.

        Also called before the completion event runs, when an interrupt or
        cancel finds the action due; the queued event is then canceled.
        """
        # Progress reaches 1.0 even if a requirement broke: the full duration
        # elapsed, only the effect is withheld.
        self._progress = 1.0
        self._cancel_event()

        if not self._requirements_met(self.completion_requirements):
            self._fail()
            return

        self._state = ActionState.COMPLETED
        self._release_agent()
        self.on_complete()

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"{self.name}(state={self._state.name}, "
            f"progress={self.progress:.0%}, "
            f"duration={self.duration})"
        )


class HasActions:
    """Mixin that lets an agent perform Actions.

    An agent performs at most one action at a time, held in the read-only
    ``current_action``. Actions are started with start_action(), preempted
    through interrupt_for() under the should_interrupt() policy, stopped
    temporarily with interrupt_action() and ended with cancel_action().
    Whenever an action ends and nothing replaces it, the on_idle() wake
    fires.

    Action instances require their agent to inherit from this mixin.
    """

    if TYPE_CHECKING:
        # Supplied by the Agent class this mixin is combined with.
        model: Model
        unique_id: int

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the action slot."""
        self._current_action: Action | None = None
        self._wake_event: Event | None = None
        self._wake_previous: Action | None = None
        self._last_wake_time: float = float("-inf")
        super().__init__(*args, **kwargs)

    @property
    def current_action(self) -> Action | None:
        """The ACTIVE action this agent performs, or None. Read-only."""
        return self._current_action

    @property
    def is_busy(self) -> bool:
        """Whether the agent is currently performing an action."""
        return self._current_action is not None

    def remove(self) -> None:
        """Remove the agent, abandoning its action and pending wake.

        If the agent is currently performing an action, the action's
        scheduled completion event is cancelled silently, and no hook
        fires. A pending on_idle wake is cancelled as well: a removed
        agent never wakes.

        If your action holds external resources (e.g., a Resource slot,
        a reservation, a lock), override remove() and call
        self.cancel_action() before super().remove() so that on_cancel()
        fires and cleanup logic runs:

            def remove(self):
                self.cancel_action()  # Fires on_cancel for cleanup
                super().remove()
        """
        if self._current_action is not None:
            self._current_action._cancel_event()  # Silent cleanup, no callback
            self._current_action = None

        if self._wake_event is not None:
            self._wake_event.cancel()
            self._wake_event = None
            self._wake_previous = None

        super().remove()

    def start_action(self, action: Action) -> Action:
        """Start a PENDING action.

        If one of the action's start requirements does not hold, the action
        moves to FAILED instead of starting, on_fail() fires and the agent
        stays idle. Check action.has_failed rather than assuming the action
        is running.

        Args:
            action: The Action to perform. Must have been created with
                this agent as its agent.

        Returns:
            The action.

        Raises:
            ValueError: If the action doesn't belong to this agent, is not
                PENDING (continue an INTERRUPTED action with
                action.resume(); a final one can't restart), if the agent is
                busy, or if the resolved duration or priority is invalid.
                The agent's slot is left as it was.
        """
        self._check_owner(action)
        if action.state is not ActionState.PENDING:
            hint = (
                "Use action.resume() to continue an INTERRUPTED action."
                if action.state is ActionState.INTERRUPTED
                else "Create a new action instead."
            )
            raise ValueError(
                f"Cannot start an action in {action.state.name} state; only "
                f"PENDING actions can be started. {hint}"
            )
        self._check_idle()
        action._enter()
        return action

    def interrupt_action(self, *, force: bool = False) -> bool:
        """Interrupt the current action; see Action.interrupt().

        Args:
            force: Interrupt even if the action is not interruptible.

        Returns:
            True if the current action was interrupted. False if the agent
            is idle or the action refused.
        """
        if self._current_action is None:
            return False
        return self._current_action.interrupt(force=force)

    def cancel_action(self) -> bool:
        """Cancel the current action for good; see Action.cancel().

        To stop it temporarily instead, use interrupt_action(force=True).

        Returns:
            True if the current action was canceled, False if idle.
        """
        if self._current_action is None:
            return False
        return self._current_action.cancel()

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
            interrupt_action(force=True) to force. This hook decides
            policy, the flag stays a hard property of the action.
        """
        return current.interruptible and incoming.priority >= current.priority

    def interrupt_for(self, new_action: Action) -> bool:
        """Put new_action in the slot, preempting the current one if allowed.

        A PENDING new_action is started and an INTERRUPTED one resumed. If
        the agent is busy, should_interrupt(current, new_action) decides
        whether to preempt; the current action is then interrupted.

        Args:
            new_action: The Action to perform instead.

        A current action whose deadline has been reached completes first
        (completion wins), and so doesn't block new_action.

        Returns:
            True if new_action is the current action after the call, or
            entered ACTIVE during it (an instant action may have ended
            again). False if the preemption was refused, if a hook of the
            action that ended filled the slot or ended new_action, if
            new_action failed a start requirement, or if new_action was the
            current action and was due, so it completed.

        Raises:
            ValueError: If new_action belongs to another agent or is in a
                final state; these are checked before anything changes.
                An invalid duration raises after the current action was
                interrupted, since it may depend on what on_interrupt()
                changed.

        Notes:
            Passing the current action changes nothing and returns True,
            unless it is due. The False cases differ in what they leave
            behind. A refusal
            changes nothing. A failed requirement does not roll the
            interruption back: the old action is already INTERRUPTED and
            the agent is left idle, since whether to resume it is the
            model's decision.
        """
        self._check_owner(new_action)
        if new_action is self._current_action:
            if new_action._is_due():
                new_action._complete()
                return False
            return True
        if new_action.state not in (ActionState.PENDING, ActionState.INTERRUPTED):
            raise ValueError(
                f"Cannot start an action in {new_action.state.name} state; "
                f"interrupt_for() takes a PENDING or INTERRUPTED action."
            )
        attempts = new_action._attempts

        def slot_outcome() -> bool | None:
            # After an action ended, its hooks may have started new_action,
            # filled the slot with something else, or ended new_action.
            if new_action._attempts != attempts:
                return True
            if self._current_action is not None or new_action.state not in (
                ActionState.PENDING,
                ActionState.INTERRUPTED,
            ):
                return False
            return None

        current = self._current_action
        if current is not None and current._is_due():
            current._complete()
            if (outcome := slot_outcome()) is not None:
                return outcome

        current = self._current_action
        if current is not None:
            new_action._resolve_priority()
            if not self.should_interrupt(current, new_action):
                return False
            # The question is whether the slot is free afterwards, not what
            # interrupt() returned.
            current.interrupt()
            if (outcome := slot_outcome()) is not None:
                return outcome

        return new_action._enter()

    def _check_owner(self, action: Action) -> None:
        """Raise ValueError unless this agent owns the action."""
        if action.agent is not self:
            raise ValueError(
                f"Action's agent (id={action.agent.unique_id}) does not match "
                f"this agent (id={self.unique_id})."
            )

    def _check_idle(self) -> None:
        """Raise ValueError if the agent is performing an action."""
        if self._current_action is not None:
            raise ValueError(
                f"Agent {self.unique_id} is already performing an action "
                f"({self._current_action!r}). Use interrupt_for(), "
                f"interrupt_action() or cancel_action() first."
            )

    def on_idle(self, previous: Action | None) -> None:
        """Called when the agent's action has ended and nothing replaced it.

        Fires for every way an action can end -- completion, interruption,
        cancellation, failure -- as a zero-delay ``Priority.LOW`` event
        rather than synchronously, so every agent finishing at time t
        decides after all completions at t have run.

        Override it to choose what to do next, typically by starting or
        resuming an action. The default does nothing, and nothing is
        resumed automatically: an interrupted action stays interrupted
        unless this hook (or other model code) resumes it.

        Args:
            previous: The action that just ended, in the state it ended in
                (COMPLETED, INTERRUPTED, FAILED or CANCELED). None is
                reserved for wakes not caused by an action ending; no
                built-in trigger sends it today.

        Notes:
            At most one wake fires per agent per model time, so a
            zero-duration action started here cannot re-trigger the hook
            in the same instant. If the guard leaves the agent idle, a
            ``RuntimeWarning`` names the agent, the action that ended and
            the time, since the agent won't act again unless other model
            code starts something. The wake is skipped silently when the
            agent is busy again by the time the event runs -- interrupt_for()
            refills the slot synchronously, so no idle gap ever existed.
            If several actions end before the wake runs, they coalesce
            into one call and ``previous`` is the latest of them.
        """

    def _schedule_wake(self, previous: Action | None) -> None:
        """Queue the deferred on_idle event, coalescing repeats.

        Called by Action._release_agent whenever this agent's slot is
        cleared. If a wake is already pending, no second event is queued --
        only ``previous`` is brought up to date. The once-per-time guard is
        applied when the event runs rather than here, so it only warns about
        an agent that is still idle by then.
        """
        self._wake_previous = previous
        if self._wake_event is not None:
            return
        self._wake_event = self.model.schedule_event(
            self._fire_wake, after=0.0, priority=Priority.LOW
        )

    def _fire_wake(self) -> None:
        """Run the pending wake: call on_idle if the agent is still free."""
        previous = self._wake_previous
        self._wake_event = None
        self._wake_previous = None
        if self._current_action is not None:
            return
        if self._last_wake_time == self.model.time:
            warnings.warn(
                f"Agent {self.unique_id} stays idle: {previous!r} ended at time "
                f"{self.model.time}, after on_idle already ran at that time, "
                f"and on_idle runs at most once per agent per model time. To "
                f"chain actions that take no time, start the next one directly "
                f"instead of from on_idle.",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        # Recorded only when on_idle actually runs, so a busy skip does not
        # suppress a later release at the same model time.
        self._last_wake_time = self.model.time
        self.on_idle(previous)
