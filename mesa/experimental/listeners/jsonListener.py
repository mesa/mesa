"""Example: JSON-based storage backend for CollectorListener."""

import json
import pathlib
from typing import Any

import numpy as np
import pandas as pd

from .baseCollectorListener import BaseCollectorListener, DatasetConfig


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
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)


class JSONListener(BaseCollectorListener):
    """Store data as JSON files."""

    def __init__(
        self,
        model,
        config: dict[str, DatasetConfig | dict[str, Any]] | None = None,
        output_dir=".",
    ):
        """Initialize JSON Listener."""
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
