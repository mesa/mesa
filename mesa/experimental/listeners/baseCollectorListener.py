"""BaseCollectorListener for the Custom Listeners.

Subclasses must implement:
    - _store_dataset_snapshot: Store a single dataset snapshot
    - _get_dataset_dataframe: Retrieve stored data as DataFrame
    - clear: Clear stored data
    - summary: Return collection status summary

The base class handles:
    - Observable subscription and signal handling
    - Collection interval management
    - Dataset configuration
    - Enable/disable logic


Key Design Principles:
    1. ABC-based design for custom storage backends
    2. Separation of concerns: BaseCollectorListener(owns config) vs CustomListener(owns storage)
    3. Observable-based collection
"""

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd

from mesa import Model
from mesa.experimental.mesa_signals import SignalType


@dataclass
class DatasetConfig:
    """Configuration for dataset collection behavior.

    Attributes:
        interval: Collection frequency in time units
        start_time: First time to begin collection
        end_time: Last time to collect (None = collect forever)
        window_size: Max snapshots to keep (None = keep all, N = sliding window)
        enabled: Whether collection is active
        _next_collection: Next scheduled collection time (internal)

    Examples:
        # Collect every step from 0 to infinity
        DatasetConfig(interval=1, start_time=0)

        # Collect every 5 steps from 100 to 500
        DatasetConfig(interval=5, start_time=100, end_time=500)

        # Collect every step, keep only last 1000 snapshots
        DatasetConfig(interval=1, start_time=0, window_size=1000)

        # Warmup phase: skip first 100 steps, collect 100-200
        DatasetConfig(interval=1, start_time=100, end_time=200)
    """

    interval: int | float = 1
    start_time: int | float = 0
    end_time: int | float | None = None
    window_size: int | None = None
    enabled: bool = True
    _next_collection: int | float = 0

    def __post_init__(self):
        """Validate configuration."""
        if self.interval <= 0:
            raise ValueError(f"interval must be > 0, got {self.interval}")
        if self.start_time < 0:
            raise ValueError(f"start_time must be >= 0, got {self.start_time}")
        if self.end_time is not None and self.end_time <= self.start_time:
            raise ValueError(
                f"end_time ({self.end_time}) must be > start_time ({self.start_time})"
            )
        if self.window_size is not None and self.window_size <= 0:
            raise ValueError(f"window_size must be > 0 or None, got {self.window_size}")

        self._next_collection = self.start_time

    def should_collect(self, current_time: float) -> bool:
        """Check if we should collect at this time.

        Args:
            current_time: The current simulation time

        Returns:
            True if collection should happen, False otherwise
        """
        if not self.enabled:
            return False

        # Before start time
        if current_time < self.start_time:
            return False

        # After end time
        if self.end_time is not None and current_time > self.end_time:
            return False

        # Not at scheduled time
        return not current_time < self._next_collection

    def update_next_collection(self, current_time: float) -> None:
        """Update the next collection time.

        Args:
            current_time: The current simulation time
        """
        self._next_collection = current_time + self.interval

        # Auto-disable if we've passed end_time
        if self.end_time is not None and self._next_collection > self.end_time:
            self.enabled = False


class BaseCollectorListener(ABC):
    """Base class for data collection listeners."""

    def __init__(
        self,
        model: Model,
        config: dict[str, DatasetConfig | dict[str, Any]] | None = None,
    ):
        """Initialize the listener.

        Args:
            model: The model to observe.
            config: Config mapping dataset names to configuration.
                    Values can be dicts or DatasetConfig objects.
        """
        self.model = model
        self.registry = getattr(model, "data_registry", None)

        if self.registry is None:
            raise AttributeError("Model must have a DataRegistry (model.data_registry)")

        # Parse configuration
        self.configs: dict[str, DatasetConfig] = {}

        # Load defaults from registry
        for name in self.registry.datasets:
            self.configs[name] = DatasetConfig()

        # Override with user config
        if config:
            for name, user_cfg in config.items():
                if isinstance(user_cfg, DatasetConfig):
                    # Copy to ensure unique state
                    self.configs[name] = copy.copy(user_cfg)
                    # Note: __post_init__ was already called when user created the object
                else:
                    # Update default dataclass fields from dict
                    current = self.configs[name]
                    for key, value in user_cfg.items():
                        if hasattr(current, key):
                            setattr(current, key, value)
                    # Re-validate
                    current.__post_init__()

        # Initialize storage for each dataset
        for name, dataset in self.registry.datasets.items():
            self._initialize_dataset_storage(name, dataset)

        self._subscribe_to_model()

    def _subscribe_to_model(self) -> None:
        """Subscribe to model.time for automatic collection."""
        # Subscribe to time units observable
        self.model.observe("time", SignalType.CHANGE, self._on_time_change)

        # Check for immediate collection (Initial State / Time 0)
        current_time = self.model.time

        for name, config in self.configs.items():
            # Check if we should collect NOW (e.g. start_time=0, current_time=0)
            if config.should_collect(current_time):
                dataset = self.registry.datasets[name]
                data_snapshot = dataset.data

                self._store_dataset_snapshot(name, current_time, data_snapshot)

                # Important: Advance the next schedule so we don't double-collect
                config.update_next_collection(current_time)

    def _on_time_change(self, signal) -> None:
        """Handle time change signal."""
        current_time = signal.new

        for name, config in self.configs.items():
            if not config.enabled or current_time < config._next_collection:
                continue

            dataset = self.registry.datasets[name]
            data_snapshot = dataset.data

            self._store_dataset_snapshot(name, current_time, data_snapshot)

            # Update next collection time (may auto-disable)
            config.update_next_collection(current_time)

    @abstractmethod
    def _initialize_dataset_storage(self, dataset_name: str, dataset: Any) -> None:
        """Initialize storage for a specific dataset."""

    @abstractmethod
    def _store_dataset_snapshot(
        self, dataset_name: str, time: int | float, data: Any
    ) -> None:
        """Store a single snapshot of data."""

    def collect(self) -> None:
        """Manually trigger collection (for manual mode or testing).

        Normally you don't need to call this - it's handled via observables.
        But you can call it manually if needed.
        """
        current_time = self.model.time

        for name, config in self.configs.items():
            if not config.enabled:
                continue

            dataset = self.registry.datasets[name]
            data_snapshot = dataset.data
            self._store_dataset_snapshot(name, current_time, data_snapshot)
            # Note: We do NOT update _next_collection here. Manual collection
            # is treated as an "extra" snapshot that shouldn't disrupt the regular interval

    @abstractmethod
    def clear(self, dataset_name: str | None = None) -> None:
        """Clear stored data."""

    def enable_dataset(self, dataset_name: str) -> None:
        """Enable collection for a dataset."""
        if dataset_name not in self.configs:
            raise KeyError(f"Dataset '{dataset_name}' not found")
        self.configs[dataset_name].enabled = True

    def disable_dataset(self, dataset_name: str) -> None:
        """Disable collection for a dataset."""
        if dataset_name not in self.configs:
            raise KeyError(f"Dataset '{dataset_name}' not found")
        self.configs[dataset_name].enabled = False

    @abstractmethod
    def get_table_dataframe(self, name: str) -> pd.DataFrame:
        """Convert stored data to pandas DataFrame."""

    def get_all_dataframes(self) -> dict[str, pd.DataFrame]:
        """Get DataFrames for all datasets."""
        return {name: self.get_table_dataframe(name) for name in self.configs}

    @abstractmethod
    def summary(self) -> dict[str, Any]:
        """Get collection status summary."""

    # def __del__(self):
    #     """Cleanup when listener is destroyed."""
    #     if isinstance(self.model, HasObservables):
    #         with contextlib.suppress(ValueError, KeyError):
    #             self.model.unobserve("time", SignalType.CHANGE, self._on_time_change)
