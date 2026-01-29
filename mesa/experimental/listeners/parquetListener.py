"""Example: parquet-based storage backend for CollectorListener.

This demonstrates howto create a custom storage backend
by subclassing BaseCollectorListener.
"""

import os
import pathlib
from typing import Any

import numpy as np
import pandas as pd

from .collectorlistener import BaseCollectorListener


class ParquetListener(BaseCollectorListener):
    """Store collected data in Parquet files.

    Usage:
        listener = ParquetListener(model, output_dir="results/")
        model.run_model()

        Each dataset is saved to a separate parquet file
    """

    def __init__(
        self,
        model,
        config: dict[str, dict[str, Any]] | None = None,
        output_dir: str = ".",
    ):
        """Initialize Parquet storage backend.

        Args:
            model: Mesa model instance
            config: Per-dataset configuration
            output_dir: Directory to store parquet files

        Usage:
            model = WealthModel(n_agents=500)

            # Create listener with Parquet storage
            listener = ParquetListener(model, output_dir="results/")

            # Run simulation
            for _ in range(500):
                model.step()

            # Get results
            wealth_df = listener.get_table_dataframe("wealth")

            print(listener.summary())
        """
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Buffers for batching writes
        self.buffers: dict[str, list] = {}
        self.buffer_size = 1000  # Write every N rows

        super().__init__(model, config)

    def _initialize_dataset_storage(self, dataset_name: str, dataset: Any) -> None:
        """Initialize buffer for dataset."""
        self.buffers[dataset_name] = []

    def _store_dataset_snapshot(self, dataset_name: str, step: int, data: Any) -> None:
        """Buffer data and write to Parquet when buffer is full."""
        buffer = self.buffers[dataset_name]

        match data:
            case np.ndarray() if data.size > 0:
                # Convert numpy to records
                try:
                    dataset = self.registry.datasets[dataset_name]
                    columns = list(dataset._attributes)
                except (AttributeError, KeyError):
                    n_cols = data.shape[1] if data.ndim > 1 else 1
                    columns = [f"col_{i}" for i in range(n_cols)]

                df = pd.DataFrame(data, columns=columns)
                df["step"] = step
                buffer.extend(df.to_dict("records"))

            case list() if data:
                buffer.extend([{**row, "step": step} for row in data])

            case dict():
                buffer.append({**data, "step": step})

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
                if filepath.exists():
                    filepath.unlink()
                self.buffers[name].clear()
        else:
            if dataset_name not in self.buffers:
                raise KeyError(f"Dataset '{dataset_name}' not found")

            filepath = self.output_dir / f"{dataset_name}.parquet"
            if filepath.exists():
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

    def __del__(self):
        """Flush all buffers on cleanup."""
        for name in self.buffers:
            self._flush_buffer(name)
        super().__del__()
