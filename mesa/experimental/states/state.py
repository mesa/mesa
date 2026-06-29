"""Continuous physics layer for Mesa 4.0.0a0.

Provides piecewise-linear and piecewise-quadratic trajectory tracking, centralized batch scheduling,
and declarative threshold monitors natively integrated with mesa_signals.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any
from collections.abc import Callable

import mesa.experimental.mesa_signals.core as signals_core
from mesa.experimental.mesa_signals.core import (
    BaseObservable,
    ComputedState,
    _hashable_signal,
)
from mesa.experimental.mesa_signals.core import Observable
from mesa.experimental.mesa_signals.signal_types import ObservableSignals
from mesa.experimental.mesa_signals import ModelSignals

if TYPE_CHECKING:
    from mesa.experimental.mesa_signals.core import HasEmitters
    from mesa.agent import Agent
    from mesa.model import Model
    from mesa.time.events import Event


class ContinuousState(BaseObservable):
    """A descriptor that tracks and extrapolates continuous states over time.

    Args:
        fallback_value (float, optional): Initial value for the state. Defaults to 0.0.
        rate (float | Callable[[Any], float], optional): Either a constant float rate-of-change,
            or a callable evaluated reactively. Defaults to 0.0.
    """

    signal_types = ObservableSignals

    def __init__(
        self,
        fallback_value: float = 0.0,
        rate: float | Callable[[Any], float] = 0.0,
    ) -> None:
        """Initialize the ContinuousState descriptor."""
        super().__init__(fallback_value=fallback_value)
        self.fallback_value = fallback_value
        self._rate_input = rate

    def __set_name__(self, owner: type[HasEmitters], name: str) -> None:
        """Bind the descriptor to the class and create shadow observables.

        Args:
            owner (type[HasEmitters]): The class owning the descriptor.
            name (str): The name of the attribute assigned to this descriptor.

        Raises:
            AttributeError: If the dynamically generated shadow variable already exists.
        """
        super().__set_name__(owner, name)
        self.private_state_name = f"_{name}_continuous_state"
        self.rate_name = f"{name}_rate"

        if self.rate_name in owner.__dict__:
            raise AttributeError(
                f"Namespace collision: '{self.rate_name}' is reserved."
            )

        rate_descriptor = Observable()
        rate_descriptor.__set_name__(owner, self.rate_name)
        setattr(owner, self.rate_name, rate_descriptor)

    def _get_state(self, instance: HasEmitters) -> dict[str, Any]:
        """Retrieve or initialize the internal state tracking dictionary.

        Args:
            instance (HasEmitters): The instance owning the state.

        Returns:
            dict[str, Any]: The internal tracking dictionary.
        """
        if not hasattr(instance, self.private_state_name):
            time = getattr(instance.model, "time", 0.0) if hasattr(instance, "model") else 0.0

            state: dict[str, Any] = {
                "base_value": self.fallback_value,
                "last_time": time,
                "rate_computed": None,
                "current_rate": 0.0,
                "second_order_rate": 0.0,
                "_rate_handler": None,
            }
            setattr(instance, self.private_state_name, state)

            if callable(self._rate_input):
                def compute_rate(agent: Agent) -> float:
                    return self._rate_input(agent)

                rate_computed = ComputedState(instance, self.rate_name, compute_rate)
                state["rate_computed"] = rate_computed

                def on_rate_change(message: Any) -> None:
                    self._snapshot_trajectory(instance)

                state["_rate_handler"] = on_rate_change
                instance.observe(
                    self.rate_name, ObservableSignals.CHANGED, state["_rate_handler"]
                )
            else:
                state["current_rate"] = float(self._rate_input)

        return getattr(instance, self.private_state_name)

    def _find_chained_rate(
        self, comp_state: ComputedState | None, instance: HasEmitters
    ) -> float:
        """Find the rate-of-the-rate if this state depends on another ContinuousState.

        Args:
            comp_state (ComputedState | None): The computed state of the first-order rate.
            instance (HasEmitters): The instance being evaluated.

        Returns:
            float: The active second-order rate, or 0.0 if not chained.
        """
        if comp_state is None:
            return 0.0

        for parent, attrs in comp_state.parents.items():
            for attr_name in attrs:
                descriptor = getattr(type(parent), attr_name, None)
                if isinstance(descriptor, ContinuousState):
                    return descriptor.get_rate(parent)

        return 0.0

    def _extrapolate(self, state: dict[str, Any], dt: float) -> float:
        """Project the value forward using linear and quadratic terms.

        Args:
            state (dict[str, Any]): The state tracking dictionary.
            dt (float): Delta time since the last snapshot.

        Returns:
            float: The mathematically extrapolated continuous value.
        """
        return (
            state["base_value"]
            + state["current_rate"] * dt
            + 0.5 * state["second_order_rate"] * dt * dt
        )

    def _snapshot_trajectory(self, instance: HasEmitters) -> None:
        """Commit the current extrapolated value as the new baseline.

        Args:
            instance (HasEmitters): The instance whose trajectory is shifting.
        """
        state = self._get_state(instance)
        current_time = getattr(instance.model, "time", 0.0)

        dt = current_time - state["last_time"]
        current_value = self._extrapolate(state, dt)

        state["base_value"] = current_value
        state["last_time"] = current_time

        if state["rate_computed"] is not None:
            self._refresh_rate_if_dirty(state, instance)

        signal = signals_core.Message(
            name=self.public_name,
            owner=instance,
            signal_type=ObservableSignals.CHANGED,
            additional_kwargs={"old": current_value, "new": current_value},
        )
        if not instance._suppress:
            instance._mesa_notify(signal)

    def _is_uninitialized_observable_race(self, instance: HasEmitters, exc: AttributeError) -> bool:
        """Check if an AttributeError matches the known init-ordering race condition.

        Args:
            instance (HasEmitters): The instance being evaluated.
            exc (AttributeError): The caught exception.

        Returns:
            bool: True if it is a safe initialization race, False if it is a genuine bug.
        """
        missing_name = getattr(exc, "name", None)
        if missing_name is None or not missing_name.startswith("_"):
            return False

        public_name = missing_name[1:]
        observables = getattr(type(instance), "observables", None)
        if not observables or public_name not in observables:
            return False

        descriptor = getattr(type(instance), public_name, None)
        return isinstance(descriptor, BaseObservable) and descriptor.private_name == missing_name

    def _evaluate_rate(self, instance: HasEmitters) -> float:
        """Safely evaluate the rate utilizing Mesa's native Diffing Engine.

        Args:
            instance (HasEmitters): The instance being evaluated.

        Returns:
            float: The computed rate.

        Raises:
            AttributeError: If the rate callable references a non-existent variable.
        """
        state = self._get_state(instance)
        comp_state: ComputedState | None = state.get("rate_computed")

        if comp_state is None:
            return float(self._rate_input)

        if not comp_state.is_dirty:
            return float(comp_state.value)

        try:
            comp_state.evaluate()
        except AttributeError as exc:
            if not self._is_uninitialized_observable_race(instance, exc):
                raise
            return 0.0

        return float(comp_state.value)

    def _refresh_rate_if_dirty(self, state: dict[str, Any], instance: HasEmitters) -> None:
        """Re-evaluate the first and second-order rates if dependencies have shifted.

        Args:
            state (dict[str, Any]): The tracking state.
            instance (HasEmitters): The instance owning the state.
        """
        comp_state = state.get("rate_computed")
        if comp_state is not None and comp_state.is_dirty:
            state["current_rate"] = self._evaluate_rate(instance)
            state["second_order_rate"] = self._find_chained_rate(comp_state, instance)

    def get_rate(self, instance: HasEmitters) -> float:
        """Return the current rate for the instance, evaluating lazily if needed.

        Args:
            instance (HasEmitters): The instance being evaluated.

        Returns:
            float: The currently active rate.
        """
        state = self._get_state(instance)
        self._refresh_rate_if_dirty(state, instance)
        return state["current_rate"]

    def __get__(self, instance: HasEmitters, owner: type | None = None) -> Any:
        """Extrapolate and return the continuous state.

        Args:
            instance (HasEmitters): The instance being accessed.
            owner (type, optional): The class owning the descriptor. Defaults to None.

        Returns:
            Any: The current mathematical value of the state.
        """
        if instance is None:
            return self

        state = self._get_state(instance)
        self._refresh_rate_if_dirty(state, instance)
        current_time = getattr(instance.model, "time", 0.0)
        dt = current_time - state["last_time"]
        current_value = self._extrapolate(state, dt)

        if signals_core.CURRENT_COMPUTED is not None:
            signals_core.CURRENT_COMPUTED._record_access(
                instance, self.public_name, current_value
            )
            signals_core.PROCESSING_SIGNALS.add(_hashable_signal(instance, self.public_name))

        return current_value

    def __set__(self, instance: HasEmitters, value: float) -> None:
        """Assign a new mathematical baseline to the state.

        Args:
            instance (HasEmitters): The instance being modified.
            value (float): The new baseline value.

        Raises:
            ValueError: If a cyclical dependency is detected.
        """
        if (
            signals_core.CURRENT_COMPUTED is not None
            and _hashable_signal(instance, self.public_name) in signals_core.PROCESSING_SIGNALS
        ):
            raise ValueError("Cyclical dependency detected in ContinuousState.")

        state = self._get_state(instance)
        current_time = getattr(instance.model, "time", 0.0)

        has_subscribers = instance._has_subscribers(self.public_name, ObservableSignals.CHANGED)
        old_value = self.__get__(instance) if has_subscribers else None

        state["base_value"] = float(value)
        state["last_time"] = current_time

        if has_subscribers:
            instance.notify(
                self.public_name,
                ObservableSignals.CHANGED,
                old=old_value,
                new=float(value),
            )
            signals_core.PROCESSING_SIGNALS.discard(_hashable_signal(instance, self.public_name))


class ContinuousScheduler:
    """Centralized master clock for batching continuous-state threshold crossings.

    Args:
        model (Model): The Mesa model instance this scheduler belongs to.
    """

    def __init__(self, model: Model) -> None:
        """Initialize the ContinuousScheduler."""
        self.model = model
        self._active_thresholds: set[tuple[Agent, Threshold]] = set()
        self._master_event: Event | None = None
        self._execute_batch_ref = self._execute_batch

        self.model.observe(
            "agents", ModelSignals.AGENT_ADDED, self._bind_agent_thresholds
        )

    def _bind_agent_thresholds(self, message: Any) -> None:
        """Wire up Threshold descriptors for a newly registered agent.

        Args:
            message (Any): The signal payload containing the agent.
        """
        agent = None

        if hasattr(message, "additional_kwargs") and "args" in message.additional_kwargs:
            args = message.additional_kwargs["args"]
            if args:
                candidate = args[0]
                if (
                    hasattr(candidate, "unique_id")
                    and hasattr(candidate, "model")
                    and candidate is not self.model
                ):
                    agent = candidate

        if agent is None:
            return

        cls = agent.__class__
        for klass in cls.__mro__:
            if "_continuous_thresholds" in klass.__dict__:
                for t_name in klass.__dict__["_continuous_thresholds"]:
                    threshold = getattr(cls, t_name, None)
                    if isinstance(threshold, Threshold):
                        threshold.bind(agent)

    def track(self, instance: Agent, threshold: Threshold) -> None:
        """Register a threshold and update the master event.

        Args:
            instance (Agent): The agent possessing the threshold.
            threshold (Threshold): The threshold descriptor to track.
        """
        self._active_thresholds.add((instance, threshold))
        self._update_master_clock()

    def untrack(self, instance: Agent, threshold: Threshold) -> None:
        """Deregister a threshold from active tracking.

        Args:
            instance (Agent): The agent possessing the threshold.
            threshold (Threshold): The threshold descriptor to untrack.
        """
        self._active_thresholds.discard((instance, threshold))

    def _update_master_clock(self) -> None:
        """Reschedule the master event to the earliest active crossing time."""
        if not self._active_thresholds:
            return

        next_time = min(
            getattr(inst, thresh.time_attr)
            for inst, thresh in self._active_thresholds
        )

        if math.isinf(next_time):
            return

        if self._master_event is None or next_time < self._master_event.time:
            if self._master_event is not None:
                try:
                    self._master_event.cancel()
                except Exception:
                    pass
            self._master_event = self.model.schedule_event(
                self._execute_batch_ref, at=next_time
            )

    def _execute_batch(self) -> None:
        """Fire all thresholds whose crossing time has been reached."""
        self._master_event = None
        current_time = self.model.time

        triggered = [
            (inst, thresh)
            for inst, thresh in self._active_thresholds
            if getattr(inst, thresh.time_attr) <= current_time
        ]

        self.model.random.shuffle(triggered)

        for inst, thresh in triggered:
            thresh.execute(inst)

        self._update_master_clock()


class Threshold:
    """Class-level descriptor that fires a callback when a ContinuousState crosses a limit.

    Args:
        state (ContinuousState): The descriptor to monitor.
        limit (float): The threshold value at which the callback fires.
        callback (str): Name of the agent method to call.
        direction (str, optional): The boundary direction constraint ("rising", "falling", 
            or "crossing"). Defaults to "crossing".
    """

    def __init__(
        self,
        state: ContinuousState,
        limit: float,
        callback: str,
        direction: str = "crossing",
    ) -> None:
        """Initialize the Threshold descriptor."""
        self.state = state
        self.limit = limit
        self.callback = callback
        self.direction = direction

    def __set_name__(self, owner: type, name: str) -> None:
        self.public_name = name
        self.time_attr = f"_{name}_projected_time"
        self.limit_attr = f"_{name}_limit_override"
        self.fired_attr = f"_{name}_fired"

        if "_continuous_thresholds" not in owner.__dict__:
            owner._continuous_thresholds = []
        if name not in owner._continuous_thresholds:
            owner._continuous_thresholds.append(name)

    def _get_limit(self, instance: Agent) -> float:
        """Retrieve the limit for this instance.

        Args:
            instance (Agent): The instance being evaluated.

        Returns:
            float: The active limit (override or default).
        """
        return getattr(instance, self.limit_attr, self.limit)

    def rearm(self, instance: Agent) -> None:
        """Clear the fired flag and recompute the projected crossing time.

        Args:
            instance (Agent): The instance to rearm.
        """
        setattr(instance, self.fired_attr, False)
        self.recalculate(instance)

    def set_limit(self, instance: Agent, value: float) -> None:
        """Set a runtime limit override and immediately rearm.

        Args:
            instance (Agent): The instance to modify.
            value (float): The new numerical limit.
        """
        setattr(instance, self.limit_attr, value)
        self.rearm(instance)

    def bind(self, instance: Agent) -> None:
        """Wire up this threshold for the instance via the signal bus.

        Args:
            instance (Agent): The instance to bind to.
        """
        if hasattr(instance, f"_{self.public_name}_recalc"):
            return

        setattr(instance, self.time_attr, math.inf)
        setattr(instance, self.fired_attr, False)

        if not hasattr(instance.model, "continuous_scheduler"):
            instance.model.continuous_scheduler = ContinuousScheduler(instance.model)

        def _recalc(message: Any = None, _inst: Agent = instance) -> None:
            self.recalculate(_inst)

        setattr(instance, f"_{self.public_name}_recalc", _recalc)

        instance.observe(
            self.state.public_name,
            ObservableSignals.CHANGED,
            getattr(instance, f"_{self.public_name}_recalc"),
        )

        self.recalculate(instance)

    def __get__(self, instance: Agent, owner: type | None = None) -> Any:
        """Retrieve the projected crossing time.

        Args:
            instance (Agent): The instance being evaluated.
            owner (type, optional): The class owning the descriptor. Defaults to None.

        Returns:
            Any: The projected crossing time or math.inf if untracked.
        """
        if instance is None:
            return self
        return getattr(instance, self.time_attr, math.inf)

    def recalculate(self, instance: Agent) -> None:
        """Compute and cache the exact crossing time using quadratic extrapolation.

        Args:
            instance (Agent): The instance to evaluate.
        """
        if not hasattr(instance.model, "continuous_scheduler"):
            return

        if getattr(instance, self.fired_attr, False):
            return

        scheduler = instance.model.continuous_scheduler

        state_dict = self.state._get_state(instance)
        self.state._refresh_rate_if_dirty(state_dict, instance)
        v = state_dict["current_rate"]
        a = state_dict["second_order_rate"]

        current_value = self.state.__get__(instance)
        limit = self._get_limit(instance)

        valid_times = []

        if a == 0.0:
            if v != 0.0:
                t = (limit - current_value) / v
                if t >= 0.0:
                    valid_times.append((t, v))
        else:
            A = 0.5 * a
            B = v
            C = current_value - limit
            discriminant = B**2 - 4 * A * C

            if discriminant >= -1e-12:
                D = max(0.0, discriminant)
                sqrt_D = math.sqrt(D)

                t1 = (-B - sqrt_D) / (2 * A)
                t2 = (-B + sqrt_D) / (2 * A)

                for t in (t1, t2):
                    if t >= -1e-12:
                        t_clean = max(0.0, t)
                        v_cross = B + a * t_clean
                        valid_times.append((t_clean, v_cross))

        future_crossings = []
        for t, v_cross in valid_times:
            if self.direction == "rising" and v_cross <= 0:
                continue
            if self.direction == "falling" and v_cross >= 0:
                continue
            if self.direction == "crossing" and v_cross == 0:
                continue
            
            future_crossings.append(t)

        if not future_crossings:
            self._never_crosses(instance, scheduler)
        else:
            time_to_cross = min(future_crossings)
            projected_time = instance.model.time + time_to_cross
            setattr(instance, self.time_attr, projected_time)
            scheduler.track(instance, self)

    def _never_crosses(self, instance: Agent, scheduler: ContinuousScheduler) -> None:
        """Mark this threshold as having no future crossing and stop tracking it.

        Args:
            instance (Agent): The instance being evaluated.
            scheduler (ContinuousScheduler): The model's master scheduler.
        """
        setattr(instance, self.time_attr, math.inf)
        scheduler.untrack(instance, self)

    def execute(self, instance: Agent) -> None:
        """Fire the callback and remove this threshold from active tracking.

        Args:
            instance (Agent): The instance whose threshold was triggered.
        """
        setattr(instance, self.time_attr, math.inf)
        setattr(instance, self.fired_attr, True)
        instance.model.continuous_scheduler.untrack(instance, self)

        callback_method = getattr(instance, self.callback)
        callback_method()
