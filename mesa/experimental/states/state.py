from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from mesa.time.events import Priority

if TYPE_CHECKING:
    from mesa import Agent, Model

DIR_BOTH = 0
DIR_RISING = 1
DIR_FALLING = -1


class StateTensor:
    """Columnar storage for continuously-evolving agent attributes."""

    def __init__(self, capacity: int = 10_000) -> None:
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
        if not self.is_active[idx]:
            return

        self.is_active[idx] = False
        self.projected_times[idx] = np.inf
        self.callbacks[idx] = None
        self.agent_refs[idx] = None
        self.free_indices.append(idx)

    def calculate_projections(self) -> None:
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

        delta_t = np.maximum(0.0, delta_t - 1e-9)
        projected = lut + delta_t

        invalid_mask = np.isnan(delta_t) | (~active)
        invalid_mask |= (d == DIR_RISING) & (r <= 0)
        invalid_mask |= (d == DIR_FALLING) & (r >= 0)

        projected[invalid_mask] = np.inf
        self.projected_times[: self.max_active_idx] = projected

    def update_single_projection(self, idx: int, current_time: float) -> None:
        r = self.rates[idx]
        d = self.directions[idx]

        if r == 0:
            self.projected_times[idx] = np.inf
            return

        dt = (self.limits[idx] - self.values[idx]) / r

        if dt < 0 or (d == DIR_RISING and r <= 0) or (d == DIR_FALLING and r >= 0):
            self.projected_times[idx] = np.inf
            return

        self.projected_times[idx] = current_time + max(0.0, dt - 1e-9)

    def register(self, agent: Agent) -> None:
        if not hasattr(agent, "_continuous_indices"):
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

                agent._continuous_indices[attr.name] = indices

                # Store the FULL LIST of indices so multi-threshold attributes don't orphan rows
                setattr(agent, attr._cache_attr, indices)

                if attr.lazy_thresholds:
                    setattr(agent, attr._lazy_cache_attr, attr.lazy_thresholds)

    def process_deferred_callbacks(self) -> None:
        self._flush_scheduled = False
        if not self.deferred_callbacks:
            return

        batch = self.deferred_callbacks
        self.deferred_callbacks = []

        for agent, callback_name, flag in batch:
            setattr(agent, flag, False)
            if agent in agent.model.agents:
                cb = getattr(agent, callback_name, None)
                if cb:
                    cb()


class ContinuousScheduler:
    def __init__(self, model: Model, tensor: StateTensor) -> None:
        self.model = model
        self.tensor = tensor
        self._master_event = None
        self._target_time = 0.0

    def schedule_next_event(self) -> None:
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
        if np.isinf(new_projection):
            return
        if self._master_event is None or new_projection < self._target_time:
            self.schedule_next_event()

    def _execute_batch(self) -> None:
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
    def __init__(
        self,
        state: ContinuousState,
        limit: float,
        callback: str,
        direction: str = "both",
        mode: str = "eager",
    ) -> None:
        self.limit = limit
        self.callback = callback
        self.direction = direction
        self.mode = mode
        self._queued_flag = f"_lazy_queued_{callback}_{id(self)}"
        state._register_threshold(self)


class ContinuousState:
    def __init__(
        self,
        default: float = 0.0,
        rate: float | str | Callable = 0.0,
        min_value: float = -float("inf"),
        max_value: float = float("inf"),
    ) -> None:
        self.default = default
        self.rate_config = rate
        self.min_value = min_value
        self.max_value = max_value
        self.name: str = ""

        self.eager_thresholds: list[Threshold] = []
        self.lazy_thresholds: list[Threshold] = []

        self._is_static_rate = not callable(rate) and not hasattr(rate, "__get__")
        self._static_rate_value = float(rate) if self._is_static_rate else 0.0
        self._needs_clamp = (min_value != -float("inf")) or (max_value != float("inf"))
        self._cache_attr = ""
        self._lazy_cache_attr = ""
        self._has_thresholds = False

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name
        self._cache_attr = f"_idx_{name}"
        self._lazy_cache_attr = f"_lazythr_{name}"

    def _get_rate(self, instance: Agent) -> float:
        if self._is_static_rate:
            return self._static_rate_value
        if callable(self.rate_config):
            return float(self.rate_config(instance))
        if hasattr(self.rate_config, "__get__"):
            return float(self.rate_config.__get__(instance, type(instance)))
        return float(self.rate_config)

    def _check_triggered(self, t: Threshold, old_val: float, new_val: float) -> bool:
        """Accurate zero-crossing detection requiring the limit to be passed between ticks."""
        if t.direction == "falling" and old_val > t.limit >= new_val:
            return True
        if t.direction == "rising" and old_val < t.limit <= new_val:
            return True
        if t.direction == "both":
            return (old_val < t.limit <= new_val) or (old_val > t.limit >= new_val)
        return False

    def _register_threshold(self, threshold: Threshold) -> None:
        if threshold.mode == "eager":
            self.eager_thresholds.append(threshold)
            self._has_thresholds = True
        else:
            self.lazy_thresholds.append(threshold)

    def __get__(self, instance: Agent | None, owner: type) -> Any:
        if instance is None:
            return self

        indices = getattr(instance, self._cache_attr, None)
        if indices is None:
            indices = getattr(instance, "_continuous_indices", {}).get(self.name)
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

        if self._needs_clamp:
            current_val = max(self.min_value, min(self.max_value, current_val))

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
        indices = getattr(instance, self._cache_attr, None)
        if indices is None:
            indices = getattr(instance, "_continuous_indices", {}).get(self.name)
            if not indices:
                return

        idx0 = indices[0]
        tensor = instance.model.state_tensor
        if not tensor.is_active[idx0]:
            return

        # 1. Lock in the exact value mathematically *before* this new write
        old_dt = instance.model.time - tensor.last_update_times[idx0]
        old_val = tensor.values[idx0] + (tensor.rates[idx0] * old_dt)

        # 2. Process clamps and determine new valid trajectory
        clamped_val = (
            max(self.min_value, min(self.max_value, value))
            if self._needs_clamp
            else value
        )
        new_rate = self._get_rate(instance)

        # 3. Safely update ALL tensor rows assigned to this attribute
        for idx in indices:
            tensor.values[idx] = clamped_val
            tensor.last_update_times[idx] = instance.model.time
            tensor.rates[idx] = new_rate

            if self._has_thresholds:
                tensor.update_single_projection(idx, instance.model.time)
                instance.model.continuous_scheduler.check_and_reschedule(
                    tensor.projected_times[idx]
                )

        # 4. Determine zero-crossings for instantaneous synchronous triggers
        if self.eager_thresholds:
            for t in self.eager_thresholds:
                if self._check_triggered(t, old_val, clamped_val):
                    cb = getattr(instance, t.callback, None)
                    if cb:
                        cb()
