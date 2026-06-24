from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from mesa.time import Priority

if TYPE_CHECKING:
    from mesa import Agent, Model

DIR_BOTH = 0
DIR_RISING = 1
DIR_FALLING = -1


class StateTensor:
    """Columnar storage for continuously-evolving agent attributes.

    Provides vectorized NumPy arrays to track values, rates, limits, and
    projections for `ContinuousState` attributes across all agents. Eager
    thresholds receive dedicated rows; lazy thresholds are evaluated on-read.
    """

    def __init__(self, capacity: int = 10_000) -> None:
        """Initialize backing arrays.

        Args:
            capacity: Maximum number of allocated attribute rows.
        """
        self.capacity = capacity

        self.values = np.zeros(capacity, dtype=np.float64)
        self.rates = np.zeros(capacity, dtype=np.float64)
        self.last_update_times = np.zeros(capacity, dtype=np.float64)
        self.limits = np.full(capacity, np.inf, dtype=np.float64)
        self.projected_times = np.full(capacity, np.inf, dtype=np.float64)
        self.directions = np.full(capacity, DIR_BOTH, dtype=np.int8)

        self.is_active = np.zeros(capacity, dtype=bool)
        self.callbacks: np.ndarray[Any, np.dtype[np.object_]] = np.empty(
            capacity, dtype=object
        )
        self.agent_refs: np.ndarray[Any, np.dtype[np.object_]] = np.empty(
            capacity, dtype=object
        )

        # O(1) allocation stack
        self.free_indices = list(range(capacity - 1, -1, -1))
        self.max_active_idx = 0

        self.deferred_callbacks: list[tuple[Agent, str, str]] = []
        self._flush_scheduled = False

    def allocate(
        self,
        agent: Agent,
        initial_value: float,
        rate: float,
        limit: float = np.inf,
        callback: str | None = None,
        direction: str = "both",
    ) -> int:
        """Claim and initialize a free tensor row.

        Args:
            agent: The owning agent.
            initial_value: Starting value.
            rate: Starting rate of change.
            limit: Target threshold limit.
            callback: Name of callback method.
            direction: Crossing constraint ("both", "rising", "falling").

        Returns:
            The allocated row index.

        Raises:
            MemoryError: If capacity is exceeded.
        """
        if not self.free_indices:
            raise MemoryError(f"StateTensor capacity ({self.capacity}) exceeded.")

        idx = self.free_indices.pop()

        self.is_active[idx] = True
        self.values[idx] = initial_value
        self.rates[idx] = rate
        self.last_update_times[idx] = getattr(agent.model, "time", 0.0)
        self.limits[idx] = limit
        self.callbacks[idx] = callback
        self.agent_refs[idx] = agent

        if direction == "rising":
            self.directions[idx] = DIR_RISING
        elif direction == "falling":
            self.directions[idx] = DIR_FALLING
        else:
            self.directions[idx] = DIR_BOTH

        if idx >= self.max_active_idx:
            self.max_active_idx = idx + 1

        return idx

    def remove(self, idx: int) -> None:
        """Deactivate a row and return it to the free pool."""
        if not self.is_active[idx]:
            return

        self.is_active[idx] = False
        self.projected_times[idx] = np.inf
        self.callbacks[idx] = None
        self.agent_refs[idx] = None
        self.free_indices.append(idx)

    def calculate_projections(self) -> None:
        """Recompute projected times for all active rows using vectorized operations."""
        if self.max_active_idx == 0:
            return

        v = self.values[: self.max_active_idx]
        r = self.rates[: self.max_active_idx]
        l = self.limits[: self.max_active_idx]
        lut = self.last_update_times[: self.max_active_idx]
        d = self.directions[: self.max_active_idx]
        active = self.is_active[: self.max_active_idx]

        with np.errstate(divide="ignore", invalid="ignore"):
            delta_t = (l - v) / r

        # Flags agents already mathematically past their threshold bounds
        already_past = (d == DIR_FALLING) & (v <= l)
        already_past |= (d == DIR_RISING) & (v >= l)
        already_past |= (d == DIR_BOTH) & ((v <= l) | (v >= l))

        delta_t[already_past] = 0.0

        delta_t = np.maximum(0.0, delta_t - 1e-9)
        projected = lut + delta_t

        # Mask inactive, infinite, or invalid trajectory projections
        invalid_mask = np.isnan(delta_t) | (~active)
        invalid_mask |= (d == DIR_RISING) & (r <= 0)
        invalid_mask |= (d == DIR_FALLING) & (r >= 0)

        # Guard the fallback net from being suppressed
        invalid_mask &= ~already_past

        projected[invalid_mask] = np.inf
        self.projected_times[: self.max_active_idx] = projected

    def update_single_projection(self, idx: int, current_time: float) -> None:
        """Recompute projected time for a single row."""
        r = self.rates[idx]
        d = self.directions[idx]
        limit = self.limits[idx]
        val = self.values[idx]

        # The Single-Row Fallback Net
        if (
            (d == DIR_FALLING and val <= limit)
            or (d == DIR_RISING and val >= limit)
            or (d == DIR_BOTH and (val <= limit or val >= limit))
        ):
            self.projected_times[idx] = current_time
            return

        if r == 0:
            self.projected_times[idx] = np.inf
            return

        dt = (limit - val) / r

        if dt < 0 or (d == DIR_RISING and r <= 0) or (d == DIR_FALLING and r >= 0):
            self.projected_times[idx] = np.inf
            return

        self.projected_times[idx] = current_time + max(0.0, dt - 1e-9)

    def register(self, agent: Agent) -> None:
        """Allocate tensor rows for all `ContinuousState` attributes on an agent."""
        if hasattr(agent, "_continuous_indices"):
            return  # Idempotency check

        agent._continuous_indices = {}

        for attr_name in dir(type(agent)):
            attr = getattr(type(agent), attr_name)
            if isinstance(attr, ContinuousState):
                indices = []
                initial_rate = attr._get_rate(agent)

                if not attr.eager_thresholds:
                    idx = self.allocate(
                        agent=agent,
                        initial_value=attr.default,
                        rate=initial_rate,
                        limit=float("inf"),
                        callback=None,
                        direction="both",
                    )
                    indices.append(idx)
                    # Kickstart
                    self.update_single_projection(
                        idx, getattr(agent.model, "time", 0.0)
                    )
                else:
                    for threshold in attr.eager_thresholds:
                        idx = self.allocate(
                            agent=agent,
                            initial_value=attr.default,
                            rate=initial_rate,
                            limit=threshold.limit,
                            callback=threshold.callback,
                            direction=threshold.direction,
                        )
                        indices.append(idx)
                        # Kickstart and queue
                        self.update_single_projection(
                            idx, getattr(agent.model, "time", 0.0)
                        )
                        if hasattr(agent.model, "continuous_scheduler"):
                            agent.model.continuous_scheduler.check_and_reschedule(
                                self.projected_times[idx]
                            )

                agent._continuous_indices[attr.name] = indices
                setattr(agent, attr._cache_attr, indices)

                if attr.lazy_thresholds:
                    setattr(agent, attr._lazy_cache_attr, attr.lazy_thresholds)

    def process_deferred_callbacks(self) -> None:
        """Flush queued lazy-mode callbacks in a batched pass."""
        self._flush_scheduled = False
        if not self.deferred_callbacks:
            return

        # Snapshot allowing re-entrant queuing during callbacks
        batch = self.deferred_callbacks
        self.deferred_callbacks = []

        for agent, callback_name, flag in batch:
            setattr(agent, flag, False)
            if agent in agent.model.agents:
                cb = getattr(agent, callback_name, None)
                if cb:
                    cb()


class ContinuousScheduler:
    """Manages event scheduling for the earliest continuous state crossing."""

    def __init__(self, model: Model, tensor: StateTensor) -> None:
        """Initialize the scheduler.

        Args:
            model: The owning model.
            tensor: The StateTensor driving projections.
        """
        self.model = model
        self.tensor = tensor
        self._master_event = None
        self._target_time = 0.0

    def schedule_next_event(self) -> None:
        """Target the master event to the earliest projected crossing."""
        if self.tensor.max_active_idx == 0:
            return

        active_times = self.tensor.projected_times[: self.tensor.max_active_idx]
        next_time = np.min(active_times)

        if np.isinf(next_time):
            return

        safe_time = max(self.model.time, next_time)
        self._target_time = safe_time

        if self._master_event is not None and not self._master_event.CANCELED:
            self._master_event.cancel()

        self._master_event = self.model.schedule_event(
            self._execute_batch, at=safe_time, priority=Priority.HIGH
        )

    def check_and_reschedule(self, new_projection: float) -> None:
        """Retarget the event if a write produces an earlier projection."""
        if np.isinf(new_projection):
            return
        if self._master_event is None or new_projection < self._target_time:
            self.schedule_next_event()

    def _execute_batch(self) -> None:
        """Fire callbacks for rows whose projection time has been reached."""
        current_time = max(self.model.time, self._target_time)

        projected = self.tensor.projected_times[: self.tensor.max_active_idx]
        triggered_indices = np.where(projected <= current_time + 1e-9)[0]

        if len(triggered_indices) == 0:
            self.schedule_next_event()
            return

        batch_targets = [
            (self.tensor.agent_refs[idx], self.tensor.callbacks[idx])
            for idx in triggered_indices
            if self.tensor.is_active[idx]
        ]

        self.tensor.projected_times[triggered_indices] = np.inf
        self.model.random.shuffle(batch_targets)

        for agent, callback_name in batch_targets:
            if agent not in self.model.agents:
                continue
            callback_method = getattr(agent, callback_name, None)
            if callback_method is not None:
                callback_method()

        self.schedule_next_event()


class Threshold:
    """A trigger condition attached to a `ContinuousState` attribute.

    Warning:
        Eager callbacks fire differently based on the trigger source:
        1. Time-Decay (Passive): Projected future crossings are safely deferred
           to the event scheduler queue (`heapq`) and execute between discrete steps.
        2. Explicit Writes (Active): Thresholds crossed due to direct assignment
           (e.g., `agent.energy = 0`) fire synchronously inline, bypassing the queue.

        Due to this inline execution, do not use callbacks that directly remove the
        owning agent (e.g., `remove()`) without explicitly handling reentrancy.
        If a mid-step interaction triggers the threshold, the code executing
        after the update will crash when running against a removed agent.
        Prefer setting flags (e.g., `is_dead = True`) to defer teardown safely.
    """

    def __init__(
        self,
        state: ContinuousState,
        limit: float,
        callback: str,
        direction: str = "both",
        mode: str = "eager",
    ) -> None:
        """Create and register a threshold.

        Args:
            state: Owning ContinuousState.
            limit: Trigger limit.
            callback: Name of the agent callback method.
            direction: "both", "rising", or "falling".
            mode: "eager" (checked on write) or "lazy" (checked on read).

        Notes:
            The `mode` parameter dictates when the tensor evaluates the crossing math and
            interacts with the scheduler. It is a performance optimization tradeoff between
            Read-heavy and Write-heavy attributes.

                * 'eager' (Scheduled): Evaluated strictly on WRITE. The exact future time of the
                crossing is calculated immediately and pushed to the model's event queue.
                Use this for attributes that change trajectory rarely (e.g., disease infection).

                * 'lazy' (Polled): Evaluated strictly on READ. Writes bypass the scheduler entirely.
                Crossings are only discovered retroactively when the attribute is accessed or
                bulk-polled. Use this for attributes that undergo constant continuous writes
                (e.g., metabolic energy drain every tick) to prevent scheduler bottlenecking.
        """
        self.limit = limit
        self.callback = callback
        self.direction = direction
        self.mode = mode
        self._queued_flag = f"_lazy_queued_{callback}_{id(self)}"
        state._register_threshold(self)


class ContinuousState:
    """Descriptor for an agent attribute that evolves continuously.

    Extrapolates state based on elapsed model time and rate. Eager thresholds
    are projected and managed by the scheduler. Lazy thresholds are evaluated
    on read.
    """

    def __init__(
        self,
        default: float = 0.0,
        rate: float | str | Callable = 0.0,
    ) -> None:
        """Configure continuous state defaults.

        Args:
            default: Starting value.
            rate: Float or callable determining the rate of change.
        """
        self.default = default
        self.rate_config = rate
        self.name: str = ""

        self.eager_thresholds: list[Threshold] = []
        self.lazy_thresholds: list[Threshold] = []

        self._is_static_rate = not callable(rate) and not hasattr(rate, "__get__")
        self._static_rate_value = float(rate) if self._is_static_rate else 0.0
        self._cache_attr = ""
        self._lazy_cache_attr = ""
        self._has_thresholds = False

    def __set_name__(self, owner: type, name: str) -> None:
        """Set attribute name and cache keys."""
        self.name = name
        self._cache_attr = f"_idx_{name}"
        self._lazy_cache_attr = f"_lazythr_{name}"

        # Metaprogramming Fallback Registration
        # Handles agents relying solely on defaults without explicit initial assignments
        if not getattr(owner, "_continuous_patched", False):
            original_init = owner.__init__

            import functools

            @functools.wraps(original_init)
            def patched_init(self_agent, *args, **kwargs):
                original_init(self_agent, *args, **kwargs)
                if not hasattr(self_agent, "_continuous_indices"):
                    self_agent.model.state_tensor.register(self_agent)

            owner.__init__ = patched_init
            owner._continuous_patched = True

    def _get_rate(self, instance: Agent) -> float:
        """Resolve the active rate of change."""
        if self._is_static_rate:
            return self._static_rate_value
        if callable(self.rate_config):
            return float(self.rate_config(instance))
        if hasattr(self.rate_config, "__get__"):
            return float(self.rate_config.__get__(instance, type(instance)))
        return float(self.rate_config)

    def _check_triggered(self, t: Threshold, old_val: float, new_val: float) -> bool:
        """Detect crossing between two values relative to a threshold limit."""
        if t.direction == "falling" and old_val >= t.limit > new_val:
            return True
        if t.direction == "rising" and old_val <= t.limit < new_val:
            return True
        if t.direction == "both":
            return (old_val >= t.limit > new_val) or (old_val <= t.limit < new_val)
        return False

    def _register_threshold(self, threshold: Threshold) -> None:
        """Assign threshold to eager or lazy collections."""
        if threshold.mode == "eager":
            self.eager_thresholds.append(threshold)
            self._has_thresholds = True
        else:
            self.lazy_thresholds.append(threshold)

    def __get__(self, instance: Agent | None, owner: type) -> Any:
        """Extrapolate current value and process lazy thresholds."""
        if instance is None:
            return self

        indices = getattr(instance, self._cache_attr, None)
        if indices is None:
            # Just-In-Time Registration on access
            if hasattr(instance, "model") and hasattr(instance.model, "state_tensor"):
                instance.model.state_tensor.register(instance)
                indices = getattr(instance, self._cache_attr, None)
            if not indices:
                return 0.0

        # Reads only need the primary row
        idx0 = indices[0]
        tensor = instance.model.state_tensor
        if not tensor.is_active[idx0]:
            return 0.0

        # Strictly utilize the tensor's rate. Dynamic rates must be explicitly written
        # to the tensor to prevent corrupting the projection integral.
        current_rate = tensor.rates[idx0]
        base_val = tensor.values[idx0]
        dt = instance.model.time - tensor.last_update_times[idx0]
        current_val = base_val + (current_rate * dt)

        if self.lazy_thresholds:
            # Prevents trigger spam by requiring an actual crossing between reads
            last_read_attr = self._lazy_cache_attr + "_last_val"
            old_val = getattr(instance, last_read_attr, base_val)

            lazy_list = getattr(instance, self._lazy_cache_attr, self.lazy_thresholds)
            for t in lazy_list:
                if self._check_triggered(t, old_val, current_val):
                    if not getattr(instance, t._queued_flag, False):
                        setattr(instance, t._queued_flag, True)
                        tensor.deferred_callbacks.append(
                            (instance, t.callback, t._queued_flag)
                        )
                        if not tensor._flush_scheduled:
                            tensor._flush_scheduled = True
                            instance.model.schedule_event(
                                tensor.process_deferred_callbacks, after=1e-9
                            )

            # Update cache for the next read
            setattr(instance, last_read_attr, current_val)

        return current_val

    def __set__(self, instance: Agent, value: float) -> None:
        """Store new baseline, adjust trajectory, and trigger eager thresholds."""
        indices = getattr(instance, self._cache_attr, None)
        if indices is None:
            # Just-In-Time Registration on assignment
            if hasattr(instance, "model") and hasattr(instance.model, "state_tensor"):
                instance.model.state_tensor.register(instance)
                indices = getattr(instance, self._cache_attr, None)
            if not indices:
                return

        idx0 = indices[0]
        tensor = instance.model.state_tensor
        if not tensor.is_active[idx0]:
            return

        # Extrapolate previous trajectory for crossing detection
        old_dt = instance.model.time - tensor.last_update_times[idx0]
        old_val = tensor.values[idx0] + (tensor.rates[idx0] * old_dt)

        new_rate = self._get_rate(instance)

        # Sync all associated tensor rows
        for idx in indices:
            tensor.values[idx] = value
            tensor.last_update_times[idx] = instance.model.time
            tensor.rates[idx] = new_rate

            if self._has_thresholds:
                tensor.update_single_projection(idx, instance.model.time)
                instance.model.continuous_scheduler.check_and_reschedule(
                    tensor.projected_times[idx]
                )

        if self.eager_thresholds:
            for t in self.eager_thresholds:
                if self._check_triggered(t, old_val, value):
                    cb = getattr(instance, t.callback, None)
                    if cb:
                        cb()

        # Reset lazy baseline to prevent false missed crossings
        if self.lazy_thresholds:
            setattr(instance, self._lazy_cache_attr + "_last_val", value)
