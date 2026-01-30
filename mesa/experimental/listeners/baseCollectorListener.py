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

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd

from mesa import Model
from mesa.experimental.mesa_signals import HasObservables, SignalType
from mesa.experimental.statistics import TableDataSet


@dataclass
class DatasetConfig:
    """Configuration for dataset collection behavior.

    Attributes:
        interval: Collection frequency in time units
        start: First time to begin collection
        enabled: Whether collection is active
        _next_collection: next scheduled collection time
    """

    interval: int | float = 1
    start: int | float = 0
    enabled: bool = True
    _next_collection: int | float = 0

    def __post_init__(self):
        """Validate configuration."""
        if self.interval <= 0:
            raise ValueError(f"interval must be > 0, got {self.interval}")
        if self.start < 0:
            raise ValueError(f"start must be >= 0, got {self.start}")
        self._next_collection = self.start


class BaseCollectorListener(ABC):
    """Abstract base class for data collection listeners.

    This class defines the interface that all collector listeners must implement,
    enabling custom storage backends (SQL, Parquet, HDF5, etc.) while reusing
    the signal handling and collection orchestration logic.
    """

    def __init__(
        self,
        model: Model,
        config: dict[str, dict[str, Any]] | None = None,
    ):
        """Initialize the listener and subscribe to model observables.

        Args:
            model: Mesa model instance with data_registry attribute
            config: Per-dataset configuration {name: {interval, start}}

        Raises:
            AttributeError: If model.data_registry not found
            TypeError: If data_registry doesn't have datasets attribute
        """
        self.model = model

        # Validate DataRegistry exists
        if not hasattr(model, "data_registry"):
            raise AttributeError(
                "CollectorListener requires 'model.data_registry'. "
                "Create DataRegistry() before initializing listener."
            )

        if not hasattr(model.data_registry, "datasets"):
            raise TypeError(
                "model.data_registry must have 'datasets' attribute. "
                "Ensure you're using a compatible DataRegistry."
            )

        self.registry = model.data_registry

        # Dataset configurations managed by base class
        self.configs: dict[str, DatasetConfig] = {}

        # Initialize storage and configs for all datasets in registry
        self._initialize_datasets(config or {})

        # Subscribe to model time observable
        self._subscribe_to_time()

    def _initialize_datasets(self, user_config: dict[str, dict[str, Any]]) -> None:
        """Initialize dataset configurations and call subclass initialization.

        This method is called during __init__ to set up configurations and
        delegate storage initialization to subclasses.
        """
        for dataset_name, dataset in self.registry.datasets.items():
            # SKIP TableDataSet: They manage their own history
            if isinstance(dataset, TableDataSet):
                continue

            # Merge user config with defaults
            dataset_config = user_config.get(dataset_name, {})
            config = DatasetConfig(**dataset_config)
            self.configs[dataset_name] = config

            # Let subclass initialize its storage backend
            self._initialize_dataset_storage(dataset_name, dataset)

    @abstractmethod
    def _initialize_dataset_storage(self, dataset_name: str, dataset: Any) -> None:
        """Initialize storage for a specific dataset.

        Called once per dataset during initialization. Subclasses should
        set up their storage backend here (e.g., create tables, open files).

        Args:
            dataset_name: Name of the dataset
            dataset: The dataset object from the registry
        """

    def _subscribe_to_time(self) -> None:
        """Subscribe to model.time observable for automatic collection.

        This subscribes to the 'time' observable on the model. When time
        changes, the handler is called and we check
        if any datasets are due for collection.

        Falls back gracefully if model doesn't have observables.
        """
        if isinstance(self.model, HasObservables):
            # Subscribe to time observable
            self.model.observe("time", SignalType.CHANGE, self._on_time_change)
        else:
            warnings.warn(
                "Model does not inherit from HasObservables. "
                "Auto-collection disabled. "
                "Call listener.collect() manually or make model inherit HasObservables from mesa_signals. ",
                UserWarning,
                stacklevel=2,
            )

    def _on_time_change(self, signal) -> None:
        """Handle time change signal.

        Called automatically when model.time changes.
        Checks intervals and collects data for due datasets.

        Args:
            signal: The signal object from Observable
        """
        current_time = signal.new

        for name, config in self.configs.items():
            # Exit early if not due
            if not config.enabled:
                continue
            if current_time < config._next_collection:
                continue

            # Extract data from registry and store
            try:
                dataset = self.registry.datasets[name]
                data_snapshot = dataset.data

                # Delegate storage to subclass
                self._store_dataset_snapshot(name, current_time, data_snapshot)

                # Update next collection time
                config._next_collection = current_time + config.interval

            except Exception as e:
                warnings.warn(
                    f"Collection failed: dataset='{name}', time={current_time}: {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    @abstractmethod
    def _store_dataset_snapshot(
        self, dataset_name: str, time: int | float, data: Any
    ) -> None:
        """Store a single dataset snapshot.

        This is the core storage method that subclasses must implement.
        Called automatically during collection when a dataset is due.

        Args:
            dataset_name: Name of the dataset being stored
            time: Current simulation time
            data: The data snapshot from the dataset
                  - np.ndarray for NumpyAgentDataSet
                  - list[dict] for AgentDataSet
                  - dict for ModelDataSet
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
            if current_time < config._next_collection:
                continue

            try:
                dataset = self.registry.datasets[name]
                data_snapshot = dataset.data
                self._store_dataset_snapshot(name, current_time, data_snapshot)
                config._next_collection = current_time + config.interval
            except Exception as e:
                warnings.warn(
                    f"Collection failed: dataset='{name}', time={current_time}: {e}",
                    RuntimeError,
                    stacklevel=2,
                )

    @abstractmethod
    def clear(self, dataset_name: str | None = None) -> None:
        """Clear stored data.

        Args:
            dataset_name: Specific dataset to clear, or None for all
        """

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
        """Convert stored data to pandas DataFrame.

        Args:
            name: Dataset name

        Returns:
            pandas DataFrame with all collected data

        Raises:
            KeyError: If dataset doesn't exist
        """

    def get_all_dataframes(self) -> dict[str, pd.DataFrame]:
        """Get DataFrames for all datasets.

        Returns:
            Dictionary mapping dataset names to DataFrames
        """
        return {name: self.get_table_dataframe(name) for name in self.configs}

    @abstractmethod
    def summary(self) -> dict[str, Any]:
        """Get collection status summary.

        Returns:
            Dictionary with summary statistics
        """

    def __del__(self):
        """Cleanup when listener is destroyed."""
        # Unsubscribe from observable
        if isinstance(self.model, HasObservables):
            self.model.unobserve("time", SignalType.CHANGE, self._on_time_change)
