"""BaseDataRecorder for the Custom Data Recorders.

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
    2. Separation of concerns: BaseDataRecorder(owns config) vs CustomRecorder(owns storage)
    3. Observable-based collection
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

from mesa.experimental.mesa_signals import ModelSignals, ObservableSignals

if TYPE_CHECKING:
    from mesa import Model

    from .dataset import DataRegistry, DataSet


@dataclass
class DatasetConfig:
    """Configuration for dataset collection behavior.

    Attributes:
        interval: Collection frequency in time units
        start_time: First time to begin collection
        end_time: Last time to collect (None = collect forever)
        window_size: Max snapshots to keep (None = keep all, N = sliding window).
            For a TableDataSet each snapshot holds the rows added since the
            previous collection, so the window keeps the rows of the last N
            collections rather than the last N rows.
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
    _last_collection: int | float = -1

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


class BaseDataRecorder(ABC):
    """Base class for data recorders."""

    def __init__(
        self,
        model: Model,
        config: dict[str, DatasetConfig | dict[str, Any]] | None = None,
    ):
        """Initialize the recorder.

        Args:
            model: The model to observe.
            config: Config mapping dataset names to configuration.
                    Values can be dicts or DatasetConfig objects.
        """
        self.model = model
        self.registry: DataRegistry = getattr(model, "data_registry", None)

        if self.registry is None:
            raise AttributeError("Model must have a DataRegistry (model.data_registry)")

        if config is None:
            config = {}

        # Parse configuration
        self.configs: dict[str, DatasetConfig] = {}

        # # Load defaults from registry
        # for name in self.registry.datasets:
        #     self.configs[name] = DatasetConfig()

        # Override with user config
        if config:
            for name, user_cfg in config.items():
                if isinstance(user_cfg, DatasetConfig):
                    # Copy to ensure unique state
                    self.configs[name] = copy.copy(user_cfg)
                    # Note: __post_init__ was already called when user created the object
                else:
                    # Update default dataclass fields from dict
                    configuration = DatasetConfig(**user_cfg)
                    self.configs[name] = configuration
                    # for key, value in user_cfg.items():
                    #     if hasattr(current, key):
                    #         setattr(current, key, value)
                    # Re-validate
                    # current.__post_init__()

        # Initialize storage for each dataset
        for name in config:
            dataset = self.registry[name]
            self._initialize_dataset_storage(name, dataset)

        # Append-only datasets (TableDataSet): rows recorded so far per dataset
        self._rows_recorded: dict[str, int] = {}

        self._subscribe_to_model()

    def _subscribe_to_model(self) -> None:
        """Subscribe to model.time for automatic collection."""
        # Subscribe to observables
        self.model.observe("time", ObservableSignals.CHANGED, self._on_time_change)
        self.model.observe("model", ModelSignals.RUN_ENDED, self._on_run_ended)

    def _on_time_change(self, signal) -> None:
        """Handle time change signal."""
        current_time = signal.additional_kwargs.get("old")

        for name, config in self.configs.items():
            is_overwrite = current_time == config._last_collection

            if (
                not config.enabled
                or current_time < config._last_collection
                or (current_time < config._next_collection and not is_overwrite)
            ):
                continue

            self._record_dataset(name, current_time, is_overwrite)
            config._last_collection = current_time

            # Update next collection time (may auto-disable)
            if current_time >= config._next_collection:
                config.update_next_collection(current_time)

    def _on_run_ended(self, signal) -> None:
        """Capture the final state when the run finishes."""
        current_time = self.model.time

        for name, config in self.configs.items():
            if (
                not config.enabled
                or current_time < config.start_time
                or current_time < config._last_collection
            ):
                continue

            is_overwrite = current_time == config._last_collection

            self._record_dataset(name, current_time, is_overwrite)
            config._last_collection = current_time

    def _record_dataset(
        self, name: str, time: int | float, is_overwrite: bool = False
    ) -> None:
        """Record the current data of a dataset at the given time.

        This is the single place where all collection paths (time changes,
        the end of a run, and manual ``collect()``) record a dataset.

        Args:
            name: name of the dataset
            time: the simulation time to record the data at
            is_overwrite: whether this replaces what is already recorded at
                this time. Ignored for append-only datasets: their recorded
                rows never change, so there is nothing to replace, and an
                overwrite would delete rows the snapshot does not include again.
        """
        dataset = self.registry.datasets[name]
        if self._is_append_only(dataset):
            is_overwrite = False
        data_snapshot = self._take_snapshot(name, dataset)
        self._store_dataset_snapshot(name, time, data_snapshot, is_overwrite)

    @staticmethod
    def _is_append_only(dataset: Any) -> bool:
        """Return whether rows are only ever added to the dataset.

        Such a dataset (e.g., a TableDataSet) defines ``get_rows_since``. The
        method is looked up on the type, so only dataset classes that define
        it qualify (not, e.g., objects with dynamic attributes such as mocks).
        """
        return getattr(type(dataset), "get_rows_since", None) is not None

    def _take_snapshot(self, name: str, dataset: Any) -> Any:
        """Return the data to store for a dataset at this collection.

        For an append-only dataset only the rows added since the previous
        collection are returned, because the dataset itself is cumulative.
        For all other datasets this is simply ``dataset.data``.
        """
        if not self._is_append_only(dataset):
            return dataset.data

        rows = dataset.get_rows_since(self._rows_recorded.get(name, 0))
        self._rows_recorded[name] = self._rows_recorded.get(name, 0) + len(rows)
        return rows

    @abstractmethod
    def _initialize_dataset_storage(self, dataset_name: str, dataset: Any) -> None:
        """Initialize storage for a specific dataset."""

    @abstractmethod
    def _store_dataset_snapshot(
        self,
        dataset_name: str,
        time: int | float,
        data: Any,
        is_overwrite: bool = False,
    ) -> None:
        """Store a single snapshot of data.

        Args:
            dataset_name: Name of the dataset being stored.
            time: Simulation time of this snapshot.
            data: The data to store.
            is_overwrite: If True, replace the most recently stored
                snapshot for this dataset instead of appending a new one.
                Every concrete recorder gets this from both collection
                paths (the observable-driven one and manual ``collect()``)
                whenever two snapshots land on the same ``time``.
        """

    def collect(self) -> None:
        """Manually trigger collection (for manual mode or testing).

        Normally you don't need to call this - it's handled via observables.
        But you can call it manually if needed.
        """
        current_time = self.model.time

        for name, config in self.configs.items():
            if not config.enabled:
                continue

            # A second collect() at the same time (e.g. two manual calls
            # between real steps) replaces the prior snapshot for that time
            # instead of appending a duplicate, matching _on_time_change.
            # _record_dataset() ignores this for append-only datasets, since
            # their rows never change and an overwrite would otherwise drop
            # rows the new snapshot doesn't re-include.
            is_overwrite = current_time == config._last_collection

            self._record_dataset(name, current_time, is_overwrite)
            config._last_collection = current_time

            if (current_time - config.start_time) % config.interval == 0:
                config.update_next_collection(current_time)

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

    def add_dataset(
        self, dataset: DataSet, configuration: DatasetConfig | None = None
    ) -> None:
        """Add a dataset to the collection.

        Args:
            dataset (DataSet): Dataset to record
            configuration (DatasetConfig|None): Configuration for recording the dataset

        """
        if configuration is None:
            configuration = DatasetConfig()

        name = dataset.name
        self.configs[name] = copy.copy(configuration)
        self._initialize_dataset_storage(name, dataset)

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
    #     """Cleanup when recorder is destroyed."""
    #     with contextlib.suppress(ValueError, KeyError):
    #         self.model.unobserve("time", ObservableSignals.CHANGE, self._on_time_change)
