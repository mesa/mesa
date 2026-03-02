"""DataRecorder for the DataRegistry Architecture.

This module orchestrates data collection from Mesa's DataRegistry, managing
storage and conversion to analysis-ready formats.

Architecture:
    DataRegistry → DataRecorder → Analysis

    - DataRegistry: Pure extraction (what to collect)
    - DataRecorders: Storage orchestration (efficient accumulation) derived from BaseDataRecorder
    - Observable-based auto-collection - subscribes to model.time observable

"""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import sqlite3
import warnings
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .basedatarecorder import BaseDataRecorder, DatasetConfig

if TYPE_CHECKING:
    from mesa.model import Model


@dataclass
class DatasetStorage:
    """Storage container with sliding window support."""

    blocks: deque = field(default_factory=deque)
    metadata: dict[str, Any] = field(default_factory=dict)
    total_rows: int = 0
    estimated_size_bytes: int = 0


class DataRecorder(BaseDataRecorder):
    """In-memory data recorder (default implementation)."""

    def __init__(
        self,
        model: Model,
        config: dict[str, DatasetConfig | dict[str, Any]] | None = None,
    ):
        """Initialize the recorder and subscribe to model observables."""
        self.storage: dict[str, DatasetStorage] = {}
        super().__init__(model, config)

    def _initialize_dataset_storage(self, dataset_name: str, dataset: Any) -> None:
        """Initialize storage with deque for efficient windowing."""
        config = self.configs[dataset_name]
        maxlen = config.window_size if config.window_size else None

        self.storage[dataset_name] = DatasetStorage(
            blocks=deque(maxlen=maxlen), metadata={"initialized": True}
        )

    def _store_dataset_snapshot(
        self, dataset_name: str, time: int | float, data: Any
    ) -> None:
        """Store data snapshot with automatic window management."""
        storage = self.storage[dataset_name]
        config = self.configs[dataset_name]

        # Track old data if we're about to evict
        old_data = None
        if config.window_size and len(storage.blocks) >= config.window_size:
            _old_time, old_data = storage.blocks[0]

        # Store new data
        added_bytes = 0

        match data:
            case np.ndarray():
                if data.size > 0:
                    ids = None
                    dataset = self.registry.datasets[dataset_name]
                    ids = getattr(dataset, "agent_ids", None)
                    ids_col = ids.reshape(-1, 1)
                    data_to_store = np.hstack([ids_col, data])

                    data_copy = data_to_store.copy()
                    storage.blocks.append((time, data_copy))
                    storage.total_rows += len(data_copy)
                    added_bytes = data_copy.nbytes

                    if "type" not in storage.metadata:
                        storage.metadata["type"] = "numpyagentdataset"
                        storage.metadata["dtype"] = data.dtype
                        dataset = self.registry.datasets[dataset_name]
                        storage.metadata["columns"] = list(dataset._attributes)

            case list():
                if data:
                    storage.blocks.append((time, data))
                    storage.total_rows += len(data)
                    added_bytes = len(data) * 100

                    if "type" not in storage.metadata:
                        storage.metadata["type"] = "agentdataset"
                        storage.metadata["columns"] = list(data[0].keys())

            case dict():
                row = {**data, "time": time}
                storage.blocks.append(row)
                storage.total_rows += 1
                added_bytes = 100

                if "type" not in storage.metadata:
                    storage.metadata["type"] = "modeldataset"
                    storage.metadata["columns"] = [*list(data.keys()), "time"]

            case _:
                storage.blocks.append((time, data))
                storage.total_rows += 1
                added_bytes = 100

                if "type" not in storage.metadata:
                    storage.metadata["type"] = "custom"

        # Update bookkeeping for evicted data
        if old_data is not None:
            match old_data:
                case np.ndarray():
                    storage.total_rows -= len(old_data)
                    storage.estimated_size_bytes -= old_data.nbytes
                case list():
                    storage.total_rows -= len(old_data)
                    storage.estimated_size_bytes -= len(old_data) * 100
                case dict():
                    storage.total_rows -= 1
                    storage.estimated_size_bytes -= 100
                case _:
                    storage.total_rows -= 1
                    storage.estimated_size_bytes -= 100

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

        data_type = storage.metadata.get("type", "unknown")

        if not storage.blocks and data_type == "unknown":
            return pd.DataFrame(columns=["time"])

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
        """Convert numpy array blocks to DataFrame."""
        columns = storage.metadata.get("columns", [])
        if not storage.blocks:
            final_cols = ["agent_id", *columns, "time"]
            return pd.DataFrame(columns=final_cols)

        arrays = []
        times = []
        for time, array in storage.blocks:
            arrays.append(array)
            times.extend([time] * len(array))

        combined_array = np.vstack(arrays)
        df_cols = ["agent_id", *columns]
        df = pd.DataFrame(combined_array, columns=df_cols)
        df["time"] = times
        return df

    def _convert_agentDataSet(self, storage: DatasetStorage) -> pd.DataFrame:
        """Convert list-of-dicts blocks to DataFrame."""
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
            f"DataRecorder("
            f"datasets={len(self.storage)}, "
            f"rows={sum(s.total_rows for s in self.storage.values())}, "
            f"memory={memory})"
        )


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON Encoder that handles Numpy types."""

    def default(self, obj):
        """Convert Numpy types to native Python types."""
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)


class JSONDataRecorder(BaseDataRecorder):
    """Store data as JSON files."""

    def __init__(
        self,
        model,
        config: dict[str, DatasetConfig | dict[str, Any]] | None = None,
        output_dir=".",
    ):
        """Initialize JSON Recorder."""
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data: dict[str, list] = {}
        super().__init__(model, config)

    def _initialize_dataset_storage(self, dataset_name: str, dataset: Any) -> None:
        """Initialize empty list for dataset."""
        self.data[dataset_name] = []

    def _store_dataset_snapshot(
        self, dataset_name: str, time: int | float, data: Any
    ) -> None:
        """Store snapshot as dict."""
        match data:
            case dict():
                self.data[dataset_name].append({"time": time, "data": data})
            case list():
                self.data[dataset_name].append({"time": time, "data": data})
            case np.ndarray():
                self.data[dataset_name].append({"time": time, "data": data.tolist()})
            case _:
                self.data[dataset_name].append({"time": time, "data": data})

    def get_table_dataframe(self, name: str) -> pd.DataFrame:
        """Convert stored JSON-like data to DataFrame."""
        if name not in self.data:
            raise KeyError(f"Dataset '{name}' not found")

        records = []
        for snapshot in self.data[name]:
            time = snapshot["time"]
            data = snapshot["data"]
            if isinstance(data, list) and data and isinstance(data[0], dict):
                for row in data:
                    records.append({**row, "time": time})
            elif isinstance(data, dict):
                records.append({**data, "time": time})
            else:
                # Handle scalar or simple list
                records.append({"time": time, "value": data})

        return pd.DataFrame(records) if records else pd.DataFrame()

    def clear(self, dataset_name: str | None = None) -> None:
        """Clear data."""
        if dataset_name is None:
            self.data.clear()
        else:
            if dataset_name in self.data:
                self.data[dataset_name].clear()

    def summary(self) -> dict[str, Any]:
        """Get summary."""
        return {
            "datasets": len(self.data),
            "output_dir": str(self.output_dir),
            "details": {
                name: {
                    "snapshots": len(snapshots),
                    "enabled": self.configs[name].enabled,
                }
                for name, snapshots in self.data.items()
            },
        }

    def save_to_json(self):
        """Save all data to JSON files."""
        for name, snapshots in self.data.items():
            filepath = self.output_dir / f"{name}.json"
            with open(filepath, "w") as f:
                json.dump(snapshots, f, indent=2, cls=NumpyJSONEncoder)


class ParquetDataRecorder(BaseDataRecorder):
    """Store collected data in Parquet files."""

    def __init__(
        self,
        model,
        config: dict[str, DatasetConfig | dict[str, Any]] | None = None,
        output_dir: str = ".",
    ):
        """Initialize Parquet storage backend."""
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Buffers for batching writes
        self.buffers: dict[str, list] = {}
        self.buffer_size = 1000  # Write every N rows

        super().__init__(model, config)

    def _initialize_dataset_storage(self, dataset_name: str, dataset: Any) -> None:
        """Initialize buffer for dataset."""
        self.buffers[dataset_name] = []

    def _store_dataset_snapshot(
        self, dataset_name: str, time: int | float, data: Any
    ) -> None:
        """Buffer data and write to Parquet when buffer is full."""
        buffer = self.buffers[dataset_name]

        match data:
            case np.ndarray() if data.size > 0:
                dataset = self.registry.datasets[dataset_name]
                columns = list(dataset._attributes)
                ids = dataset.agent_ids

                data_to_store = data
                ids_col = ids.reshape(-1, 1)
                data_to_store = np.hstack([ids_col, data])
                columns = ["agent_id", *columns]

                df = pd.DataFrame(data_to_store, columns=columns)
                df["time"] = time
                buffer.extend(df.to_dict("records"))

            case list() if data:
                buffer.extend([{**row, "time": time} for row in data])

            case dict():
                buffer.append({**data, "time": time})

        # Flush to disk if buffer is full
        if len(buffer) >= self.buffer_size:
            self._flush_buffer(dataset_name)

    def _flush_buffer(self, dataset_name: str):
        """Write buffer to Parquet file."""
        buffer = self.buffers[dataset_name]
        if not buffer:
            return

        df = pd.DataFrame(buffer)
        filepath = self.output_dir / f"{dataset_name}.parquet"

        # Append to existing file or create new
        if filepath.exists():
            existing = pd.read_parquet(filepath)
            df = pd.concat([existing, df], ignore_index=True)

        df.to_parquet(filepath, index=False, compression="snappy")
        buffer.clear()

    def get_table_dataframe(self, name: str) -> pd.DataFrame:
        """Read data from Parquet file."""
        if name not in self.buffers:
            raise KeyError(f"Dataset '{name}' not found")

        # Flush any remaining buffered data first
        self._flush_buffer(name)

        filepath = self.output_dir / f"{name}.parquet"
        if not filepath.exists():
            return pd.DataFrame()

        return pd.read_parquet(filepath)

    def clear(self, dataset_name: str | None = None) -> None:
        """Delete Parquet files."""
        if dataset_name is None:
            for name in self.buffers:
                filepath = self.output_dir / f"{name}.parquet"
                if filepath.exists():  # pragma: no cover
                    filepath.unlink()
                self.buffers[name].clear()
        else:
            if dataset_name not in self.buffers:
                raise KeyError(f"Dataset '{dataset_name}' not found")

            filepath = self.output_dir / f"{dataset_name}.parquet"
            if filepath.exists():  # pragma: no cover
                filepath.unlink()
            self.buffers[dataset_name].clear()

    def summary(self) -> dict[str, Any]:
        """Get collection status summary."""
        summary_data = {
            "datasets": len(self.buffers),
            "output_dir": str(self.output_dir),
        }

        for name in self.buffers:
            filepath = self.output_dir / f"{name}.parquet"
            if filepath.exists():
                # Get file size
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                # Get row count
                df = pd.read_parquet(filepath)
                row_count = len(df)
            else:
                size_mb = 0
                row_count = 0

            # Add buffered rows
            row_count += len(self.buffers[name])

            summary_data[name] = {
                "enabled": self.configs[name].enabled,
                "interval": self.configs[name].interval,
                "rows": row_count,
                "size_mb": size_mb,
                "next_collection": self.configs[name]._next_collection,
            }

        return summary_data

    def __del__(self):  # pragma : no cover
        """Flush all buffers on cleanup."""
        with contextlib.suppress(
            RuntimeError, ImportError, NameError, OSError, FileNotFoundError
        ):
            for name in self.buffers:
                self._flush_buffer(name)
        # super().__del__()


class SQLDataRecorder(BaseDataRecorder):
    """Store collected data in SQLite database."""

    def __init__(
        self,
        model,
        config: dict[str, DatasetConfig | dict[str, Any]] | None = None,
        db_path: str = ":memory:",
    ):
        """Initialize SQL storage backend."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.metadata: dict[str, dict] = {}
        super().__init__(model, config)

    def _initialize_dataset_storage(self, dataset_name: str, dataset: Any) -> None:
        """Initialize SQL table metadata."""
        self.metadata[dataset_name] = {"table_created": False, "columns": []}

    def _store_dataset_snapshot(
        self, dataset_name: str, time: int | float, data: Any
    ) -> None:
        """Store data snapshot in SQL."""
        match data:
            case np.ndarray() if data.size > 0:
                self._store_numpy_data(dataset_name, time, data)
            case list() if data:
                self._store_list_data(dataset_name, time, data)
            case dict():
                self._store_dict_data(dataset_name, time, data)
            case _:
                pass

    def _store_numpy_data(self, dataset_name: str, time: int | float, data: np.ndarray):
        """Store numpy array as SQL records."""
        dataset = self.registry.datasets[dataset_name]
        columns = [col for col in dataset._attributes if col != "time"]

        # Check for IDs
        if dataset is not None:
            ids = dataset.agent_ids
            ids_col = ids.reshape(-1, 1)
            data = np.hstack([ids_col, data])
            columns = ["agent_id", *columns]

        if not self.metadata[dataset_name]["table_created"]:
            col_defs = ", ".join([f'"{col}" REAL' for col in columns])
            self.conn.execute(
                f'CREATE TABLE IF NOT EXISTS "{dataset_name}" (time REAL, {col_defs})'
            )
            self.metadata[dataset_name]["table_created"] = True
            self.metadata[dataset_name]["columns"] = columns

        df = pd.DataFrame(data, columns=columns)
        df["time"] = time
        df.to_sql(dataset_name, self.conn, if_exists="append", index=False)

    def _store_list_data(self, dataset_name: str, time: int | float, data: list[dict]):
        """Store list of dicts as SQL records."""
        if not self.metadata[dataset_name]["table_created"]:
            columns = [k for k in data[0] if k != "time"]
            col_defs = ", ".join([f'"{col}" REAL' for col in columns])
            self.conn.execute(
                f'CREATE TABLE IF NOT EXISTS "{dataset_name}" (time REAL, {col_defs})'
            )
            self.metadata[dataset_name]["table_created"] = True
            self.metadata[dataset_name]["columns"] = columns

        rows = [{**row, "time": time} for row in data]
        df = pd.DataFrame(rows)
        df.to_sql(dataset_name, self.conn, if_exists="append", index=False)

    def _store_dict_data(self, dataset_name: str, time: int | float, data: dict):
        """Store single dict as SQL record."""
        if not self.metadata[dataset_name]["table_created"]:
            columns = [k for k in data if k != "time"]
            col_defs = ", ".join([f'"{col}" REAL' for col in columns])
            self.conn.execute(
                f'CREATE TABLE IF NOT EXISTS "{dataset_name}" (time REAL, {col_defs})'
            )
            self.metadata[dataset_name]["table_created"] = True
            self.metadata[dataset_name]["columns"] = columns

        row = {**data, "time": time}
        df = pd.DataFrame([row])
        df.to_sql(dataset_name, self.conn, if_exists="append", index=False)

    def get_table_dataframe(self, name: str) -> pd.DataFrame:
        """Convert stored data to pandas DataFrame."""
        if name not in self.metadata:
            raise KeyError(f"Dataset '{name}' not found")

        if not self.metadata[name]["table_created"]:
            return pd.DataFrame()

        return pd.read_sql(f'SELECT * FROM "{name}"', self.conn)  # noqa: S608

    def query(self, sql: str) -> pd.DataFrame:
        """Execute custom SQL query."""
        return pd.read_sql(sql, self.conn)

    def clear(self, dataset_name: str | None = None) -> None:
        """Clear stored data by dropping tables."""
        if dataset_name is None:
            for name in self.metadata:
                self.conn.execute(f'DROP TABLE IF EXISTS "{name}"')
                self.metadata[name]["table_created"] = False
        else:
            if dataset_name not in self.metadata:
                raise KeyError(f"Dataset '{dataset_name}' not found")

            self.conn.execute(f'DROP TABLE IF EXISTS "{dataset_name}"')
            self.metadata[dataset_name]["table_created"] = False

        self.conn.commit()

    def summary(self) -> dict[str, Any]:
        """Get collection status summary."""
        summary_data = {"datasets": len(self.metadata), "database": self.db_path}

        for name, meta in self.metadata.items():
            if meta["table_created"]:
                cursor = self.conn.execute(f'SELECT COUNT(*) FROM "{name}"')  # noqa: S608
                row_count = cursor.fetchone()[0]
            else:
                row_count = 0

            summary_data[name] = {
                "enabled": self.configs[name].enabled,
                "interval": self.configs[name].interval,
                "rows": row_count,
                "next_collection": self.configs[name]._next_collection,
            }

        return summary_data

    def __del__(self):  # pragma : no cover
        """Close database connection on cleanup."""
        # super().__del__()
        if hasattr(self, "conn"):
            self.conn.close()
