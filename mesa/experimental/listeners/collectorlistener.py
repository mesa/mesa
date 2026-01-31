"""CollectorListener for the DataRegistry Architecture.

This module orchestrates data collection from Mesa's DataRegistry, managing
storage and conversion to analysis-ready formats.

Architecture:
    DataRegistry (statistics.py) → CollectorListener → Analysis

    - DataRegistry: Pure extraction (what to collect)
    - CollectorListener: Storage orchestration (efficient accumulation) derived from BaseCollectorListener
    - Observable-based auto-collection - subscribes to model.time observable

"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .baseCollectorListener import BaseCollectorListener, DatasetConfig

if TYPE_CHECKING:
    from mesa.model import Model

# Optional Polars for high-performance output
try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


@dataclass
class DatasetStorage:
    """Storage container for collected dataset snapshots."""

    blocks: list[tuple[int, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    total_rows: int = 0
    estimated_size_bytes: int = 0


class CollectorListener(BaseCollectorListener):
    """In-memory collector listener (default implementation)."""

    def __init__(
        self,
        model: Model,
        config: dict[str, DatasetConfig | dict[str, Any]] | None = None,
    ):
        """Initialize the listener and subscribe to model observables."""
        self.storage: dict[str, DatasetStorage] = {}
        super().__init__(model, config)

    def _initialize_dataset_storage(self, dataset_name: str, dataset: Any) -> None:
        """Initialize in-memory storage for a dataset."""
        self.storage[dataset_name] = DatasetStorage(metadata={"initialized": True})

    def _store_dataset_snapshot(
        self, dataset_name: str, time: int | float, data: Any
    ) -> None:
        """Store data snapshot in memory based on type."""
        storage = self.storage[dataset_name]
        added_bytes = 0

        match data:
            # Numpy array (NumpyAgentDataSet)
            case np.ndarray() if data.size > 0:
                # CRITICAL: Copy to prevent mutation
                data_copy = data.copy()
                storage.blocks.append((time, data_copy))
                storage.total_rows += len(data_copy)
                added_bytes = data_copy.nbytes

                # Store metadata on first collection
                if "type" not in storage.metadata:
                    storage.metadata["type"] = "numpyagentdataset"
                    storage.metadata["dtype"] = data.dtype
                    # Try to get column names from dataset
                    try:
                        dataset = self.registry.datasets[dataset_name]
                        storage.metadata["columns"] = list(dataset._attributes)
                    except (AttributeError, KeyError):
                        # Fallback to generic names
                        n_cols = data.shape[1] if data.ndim > 1 else 1
                        storage.metadata["columns"] = [
                            f"col_{i}" for i in range(n_cols)
                        ]

            # List of dicts (AgentDataSet)
            case list() if data:
                storage.blocks.append((time, data))
                storage.total_rows += len(data)
                added_bytes = len(data) * 100  # Estimate

                if "type" not in storage.metadata:
                    storage.metadata["type"] = "agentdataset"
                    storage.metadata["columns"] = list(data[0].keys())

            # Single dict (ModelDataSet)
            case dict():
                row = {**data, "time": time}
                storage.blocks.append(row)
                storage.total_rows += 1
                added_bytes = 100  # Estimate

                if "type" not in storage.metadata:
                    storage.metadata["type"] = "modeldataset"
                    storage.metadata["columns"] = [*list(data.keys()), "time"]

            # Handle empty containers explicitly to prevent falling through to fallback
            case np.ndarray() | list():
                pass

            # Fallback for custom types
            case _:
                storage.blocks.append((time, data))
                storage.total_rows += 1
                added_bytes = 100

                if "type" not in storage.metadata:
                    storage.metadata["type"] = "custom"

        storage.estimated_size_bytes += added_bytes

    def clear(self, dataset_name: str | None = None) -> None:
        """Clear stored data."""
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

    def get_table_dataframe(self, name: str) -> pd.DataFrame:
        """Convert stored data to pandas DataFrame."""
        if name not in self.storage:
            raise KeyError(f"Dataset '{name}' not found")

        storage = self.storage[name]

        if not storage.blocks:
            # Empty DataFrame with correct columns
            columns = storage.metadata.get("columns", [])
            return pd.DataFrame(columns=columns)

        data_type = storage.metadata.get("type", "unknown")

        # Dispatch to appropriate converter
        match data_type:
            case "numpyagentdataset":
                return self._convert_numpyAgentDataSet(storage)
            case "agentdataset":
                return self._convert_agentDataSet(storage)
            case "modeldataset":
                return self._convert_modelDataSet(storage)
            case _:
                # Fallback
                warnings.warn(
                    f"Unknown data type '{data_type}' for '{name}'",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return pd.DataFrame(storage.blocks)

    def _convert_numpyAgentDataSet(self, storage: DatasetStorage) -> pd.DataFrame:
        """Convert numpy array blocks to DataFrame using np.vstack."""
        columns = storage.metadata.get("columns", [])
        if not storage.blocks:
            return pd.DataFrame(columns=[*columns, "time"])

        arrays = []
        times = []
        for time, array in storage.blocks:
            arrays.append(array)
            times.extend([time] * len(array))

        combined_array = np.vstack(arrays)
        df = pd.DataFrame(combined_array, columns=columns)
        df["time"] = times
        return df

    def _convert_agentDataSet(self, storage: DatasetStorage) -> pd.DataFrame:
        """Convert list-of-dicts blocks to DataFrame."""
        # Use Polars if available for better performance
        if HAS_POLARS:
            rows = []
            for time, block in storage.blocks:
                for row in block:
                    rows.append({**row, "time": time})

            if not rows:
                return pd.DataFrame(
                    columns=[*storage.metadata.get("columns", []), "time"]
                )

            return pl.DataFrame(rows).to_pandas()

        # Fallback to pandas
        rows = []
        for time, block in storage.blocks:
            for row in block:
                rows.append({**row, "time": time})

        if not rows:
            return pd.DataFrame(columns=[*storage.metadata.get("columns", []), "time"])

        return pd.DataFrame(rows)

    def _convert_modelDataSet(self, storage: DatasetStorage) -> pd.DataFrame:
        """Convert model dict blocks to DataFrame."""
        if not storage.blocks:
            return pd.DataFrame(columns=storage.metadata.get("columns", []))

        # Model dicts already have 'time' added
        if HAS_POLARS:
            return pl.DataFrame(storage.blocks).to_pandas()

        return pd.DataFrame(storage.blocks)

    def estimate_memory_usage(self) -> float:
        """Estimate current memory usage in MB."""
        total_bytes = sum(s.estimated_size_bytes for s in self.storage.values())
        return total_bytes / (1024 * 1024)

    def summary(self) -> dict[str, Any]:
        """Get collection status summary."""
        return {
            "datasets": len(self.storage),
            "total_rows": sum(s.total_rows for s in self.storage.values()),
            "memory_mb": self.estimate_memory_usage(),
            "datasets_detail": {
                name: {
                    "enabled": self.configs[name].enabled,
                    "interval": self.configs[name].interval,
                    "blocks": len(storage.blocks),
                    "rows": storage.total_rows,
                    "next_collection": self.configs[name]._next_collection,
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
