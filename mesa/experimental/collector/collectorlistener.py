"""High-performance Listener for the DataRegistry Architecture.

This module orchestrates data collection from Mesa's DataRegistry, managing
storage and conversion to analysis-ready formats.

Architecture:
    DataRegistry (statistics.py) → CollectorListener (this file) → Analysis

    - DataRegistry: Pure extraction (what to collect)
    - CollectorListener: Storage orchestration (efficient accumulation)
    - Observable-based auto-collection - subscribes to model.steps observable

Key Design Principles:
    1. Zero-copy where possible (NumpyAgentDataSet already is a view)
    2. Block storage for efficient memory usage
    3. Lazy DataFrame conversion (only when requested)
    4. Observable-based collection - clean signal subscription
    5. Minimal overhead - O(1) interval checking

"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from mesa.experimental.mesa_signals import HasObservables, SignalType
from mesa.experimental.statistics import TableDataSet

if TYPE_CHECKING:
    from mesa.model import Model

# Optional Polars for high-performance output
try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


@dataclass
class DatasetConfig:
    """Configuration for dataset collection behavior.

    Attributes:
        interval: Collection frequency in steps
        start: First step to begin collection
        enabled: Whether collection is active
        next_collection: Internal - next scheduled collection step
    """

    interval: int = 1
    start: int = 0
    enabled: bool = True
    next_collection: int = 0

    def __post_init__(self):
        """Validate configuration."""
        if self.interval < 1:
            raise ValueError(f"interval must be >= 1, got {self.interval}")
        if self.start < 0:
            raise ValueError(f"start must be >= 0, got {self.start}")
        self.next_collection = self.start


@dataclass
class DatasetStorage:
    """Storage container for collected dataset snapshots.

    Uses block storage for memory efficiency:
    - Numpy arrays: stored directly (already efficient)
    - Dicts/lists: accumulated in blocks, converted to DataFrame on demand

    Attributes:
        blocks: List of (step, data) tuples
        metadata: Dataset information (type, columns, dtype)
        config: Collection configuration
        total_rows: Total data points collected
        estimated_size_bytes: Approximate memory usage
    """

    blocks: list[tuple[int, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    config: DatasetConfig = field(default_factory=DatasetConfig)
    total_rows: int = 0
    estimated_size_bytes: int = 0


class CollectorListener:
    """Orchestrates data collection from DataRegistry using Observable signals.

    Automatically collects data after each model step by subscribing to the
    model's 'steps' observable. Clean, efficient, no scheduler overhead.

    Responsibilities:
        - Subscribe to model.steps observable
        - Collect data snapshots at configured intervals
        - Store snapshots efficiently in memory
        - Convert to analysis-ready formats (pandas/polars)

    Usage:
        # Basic - auto-collect via observable
        listener = CollectorListener(model)

        # Custom intervals per dataset
        listener = CollectorListener(
            model,
            config={
                "wealth": {"interval": 1},      # Every step
                "positions": {"interval": 10},   # Every 10 steps
                "summary": {"interval": 1, "start": 100}
            }
        )

        # Access data after simulation
        wealth_df = listener.get_table_dataframe("wealth")
        all_data = listener.get_all_dataframes()

    How It Works:
        1. Model completes user's step()
        2. Model._wrapped_step increments self.steps
        3. steps Observable emits CHANGE signal
        4. Listener's _on_step_change() is called
        5. Listener checks intervals and collects if due
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
        self.storage: dict[str, DatasetStorage] = {}

        # Initialize storage for all datasets in registry
        self._initialize_storage(config or {})

        # Subscribe to model steps observable
        self._subscribe_to_steps()

    def _initialize_storage(self, user_config: dict[str, dict[str, Any]]) -> None:
        """Create storage containers for all datasets."""
        for dataset_name, dataset in self.registry.datasets.items():
            # SKIP TableDataSet: They manage their own history
            if isinstance(dataset, TableDataSet):
                continue

            # Merge user config with defaults
            dataset_config = user_config.get(dataset_name, {})
            config = DatasetConfig(**dataset_config)

            self.storage[dataset_name] = DatasetStorage(
                config=config, metadata={"initialized": True}
            )

    def _subscribe_to_steps(self) -> None:
        """Subscribe to model.steps observable for automatic collection.

        This subscribes to the 'steps' observable on the model. When steps
        changes (after each step completes), our handler is called and we
        check if any datasets are due for collection.

        Falls back gracefully if model doesn't have observables.
        """
        if isinstance(self.model, HasObservables):
            # Subscribe to steps observable
            self.model.observe("steps", SignalType.CHANGE, self._on_step_change)
        else:
            warnings.warn(
                "Model does not inherit from HasObservables. "
                "Auto-collection disabled. "
                "Call listener.collect() manually or make model inherit HasObservables.",
                UserWarning,
                stacklevel=2,
            )

    def _on_step_change(self, signal) -> None:
        """Handle step change signal.

        Called automatically when model.steps changes.
        Checks intervals and collects data for due datasets.

        Args:
            signal: The signal object from Observable
        """
        # Signal.new contains the new step value
        current_step = signal.new

        for name, storage in self.storage.items():
            # Exit early if not due
            if not storage.config.enabled:
                continue
            if current_step < storage.config.next_collection:
                continue

            # Extract data from registry
            try:
                dataset = self.registry.datasets[name]
                data_snapshot = dataset.data

                # Store based on data type
                self._store_data(name, current_step, data_snapshot, storage)

                # Update next collection time
                storage.config.next_collection = current_step + storage.config.interval

            except Exception as e:
                warnings.warn(
                    f"Collection failed: dataset='{name}', step={current_step}: {e}",
                    RuntimeError,
                    stacklevel=2,
                )

    def collect(self) -> None:
        """Manually trigger collection (for manual mode or testing).

        Normally you don't need to call this - it's automatic via observables.
        But you can call it manually if needed.
        """
        current_step = self.model.steps

        for name, storage in self.storage.items():
            if not storage.config.enabled:
                continue
            if current_step < storage.config.next_collection:
                continue

            try:
                dataset = self.registry.datasets[name]
                data_snapshot = dataset.data
                self._store_data(name, current_step, data_snapshot, storage)
                storage.config.next_collection = current_step + storage.config.interval
            except Exception as e:
                warnings.warn(
                    f"Collection failed: dataset='{name}', step={current_step}: {e}",
                    RuntimeError,
                    stacklevel=2,
                )

    def _store_data(
        self, name: str, step: int, data: Any, storage: DatasetStorage
    ) -> None:
        """Store data snapshot efficiently based on type."""
        added_bytes = 0

        # Numpy array (NumpyAgentDataSet) - MUST COPY
        if isinstance(data, np.ndarray):
            if data.size > 0:
                # CRITICAL: Copy to prevent mutation
                data_copy = data.copy()
                storage.blocks.append((step, data_copy))
                storage.total_rows += len(data_copy)
                added_bytes = data_copy.nbytes

                # Store metadata on first collection
                if "type" not in storage.metadata:
                    storage.metadata["type"] = "numpyagentdataset"
                    storage.metadata["dtype"] = data.dtype
                    # Try to get column names from dataset
                    try:
                        dataset = self.registry.datasets[name]
                        storage.metadata["columns"] = list(dataset._attributes)
                    except (AttributeError, KeyError):
                        # Fallback to generic names
                        n_cols = data.shape[1] if data.ndim > 1 else 1
                        storage.metadata["columns"] = [
                            f"col_{i}" for i in range(n_cols)
                        ]

        # List of dicts (AgentDataSet)
        elif isinstance(data, list):
            if data:
                storage.blocks.append((step, data))
                storage.total_rows += len(data)
                added_bytes = len(data) * 100  # Estimate

                if "type" not in storage.metadata:
                    storage.metadata["type"] = "agentdataset"
                    storage.metadata["columns"] = list(data[0].keys())

        # Single dict (ModelDataSet)
        elif isinstance(data, dict):
            # Add step directly to dict
            row = {**data, "step": step}
            storage.blocks.append(row)
            storage.total_rows += 1
            added_bytes = 100  # Estimate

            if "type" not in storage.metadata:
                storage.metadata["type"] = "modeldataset"
                storage.metadata["columns"] = [*list(data.keys()), "step"]

        # Fallback
        else:
            storage.blocks.append((step, data))
            storage.total_rows += 1
            added_bytes = 100

            if "type" not in storage.metadata:
                storage.metadata["type"] = "custom"

        storage.estimated_size_bytes += added_bytes

    def clear(self, dataset_name: str | None = None) -> None:
        """Clear stored data.

        Args:
            dataset_name: Specific dataset to clear, or None for all
        """
        if dataset_name is None:
            for storage in self.storage.values():
                storage.blocks.clear()
                storage.total_rows = 0
                storage.estimated_size_bytes = 0
        else:
            if dataset_name not in self.storage:
                raise KeyError(f"Dataset '{dataset_name}' not found")

            storage = self.storage[dataset_name]
            storage.blocks.clear()
            storage.total_rows = 0
            storage.estimated_size_bytes = 0

    def enable_dataset(self, dataset_name: str) -> None:
        """Enable collection for a dataset."""
        if dataset_name not in self.storage:
            raise KeyError(f"Dataset '{dataset_name}' not found")
        self.storage[dataset_name].config.enabled = True

    def disable_dataset(self, dataset_name: str) -> None:
        """Disable collection for a dataset."""
        if dataset_name not in self.storage:
            raise KeyError(f"Dataset '{dataset_name}' not found")
        self.storage[dataset_name].config.enabled = False

    def get_table_dataframe(self, name: str) -> pd.DataFrame:
        """Convert stored data to pandas DataFrame.

        Args:
            name: Dataset name

        Returns:
            pandas DataFrame with all collected data

        Raises:
            KeyError: If dataset doesn't exist
        """
        if name not in self.storage:
            raise KeyError(f"Dataset '{name}' not found")

        storage = self.storage[name]

        if not storage.blocks:
            # Empty DataFrame with correct columns
            columns = storage.metadata.get("columns", [])
            return pd.DataFrame(columns=columns)

        data_type = storage.metadata.get("type", "unknown")

        # Dispatch to appropriate converter
        if data_type == "numpyagentdataset":
            return self._convert_numpyAgentDataSet(storage)
        elif data_type == "agentdataset":
            return self._convert_agentDataSet(storage)
        elif data_type == "modeldataset":
            return self._convert_modelDataSet(storage)
        else:
            # Fallback
            warnings.warn(
                f"Unknown data type '{data_type}' for '{name}'",
                RuntimeWarning,
                stacklevel=2,
            )
            return pd.DataFrame(storage.blocks)

    def _convert_numpyAgentDataSet(self, storage: DatasetStorage) -> pd.DataFrame:
        """Convert numpy array blocks to DataFrame."""
        columns = storage.metadata.get("columns", [])

        # Build list of DataFrames
        dfs = []
        for step, array in storage.blocks:
            df = pd.DataFrame(array, columns=columns)
            df["step"] = step
            dfs.append(df)

        if not dfs:
            return pd.DataFrame(columns=[*columns, "step"])

        # Concatenate efficiently
        return pd.concat(dfs, ignore_index=True)

    def _convert_agentDataSet(self, storage: DatasetStorage) -> pd.DataFrame:
        """Convert list-of-dicts blocks to DataFrame."""
        # Use Polars if available for better performance
        if HAS_POLARS:
            rows = []
            for step, block in storage.blocks:
                for row in block:
                    rows.append({**row, "step": step})

            if not rows:
                return pd.DataFrame(
                    columns=[*storage.metadata.get("columns", []), "step"]
                )

            return pl.DataFrame(rows).to_pandas()

        # Fallback to pandas
        rows = []
        for step, block in storage.blocks:
            for row in block:
                rows.append({**row, "step": step})

        if not rows:
            return pd.DataFrame(columns=[*storage.metadata.get("columns", []), "step"])

        return pd.DataFrame(rows)

    def _convert_modelDataSet(self, storage: DatasetStorage) -> pd.DataFrame:
        """Convert model dict blocks to DataFrame."""
        if not storage.blocks:
            return pd.DataFrame(columns=storage.metadata.get("columns", []))

        # Model dicts already have 'step' added
        if HAS_POLARS:
            return pl.DataFrame(storage.blocks).to_pandas()

        return pd.DataFrame(storage.blocks)

    def get_all_dataframes(self) -> dict[str, pd.DataFrame]:
        """Get DataFrames for all datasets.

        Returns:
            Dictionary mapping dataset names to DataFrames
        """
        return {name: self.get_table_dataframe(name) for name in self.storage}

    def estimate_memory_usage(self) -> float:
        """Estimate current memory usage in MB."""
        total_bytes = sum(s.estimated_size_bytes for s in self.storage.values())
        return total_bytes / (1024 * 1024)

    def summary(self) -> dict[str, Any]:
        """Get collection status summary.

        Returns:
            Dictionary with summary statistics
        """
        return {
            "datasets": len(self.storage),
            "total_rows": sum(s.total_rows for s in self.storage.values()),
            "memory_mb": self.estimate_memory_usage(),
            "datasets_detail": {
                name: {
                    "enabled": storage.config.enabled,
                    "interval": storage.config.interval,
                    "blocks": len(storage.blocks),
                    "rows": storage.total_rows,
                    "next_collection": storage.config.next_collection,
                    "type": storage.metadata.get("type", "unknown"),
                }
                for name, storage in self.storage.items()
            },
        }

    def __repr__(self) -> str:
        """String representation."""
        memory = f"{self.estimate_memory_usage():.1f}MB" if self.storage else "0MB"
        return (
            f"CollectorListener("
            f"datasets={len(self.storage)}, "
            f"rows={sum(s.total_rows for s in self.storage.values())}, "
            f"memory={memory})"
        )

    def __del__(self):
        """Cleanup when listener is destroyed."""
        # Unsubscribe from observable
        if isinstance(self.model, HasObservables):
            self.model.unobserve("steps", SignalType.CHANGE, self._on_step_change)
