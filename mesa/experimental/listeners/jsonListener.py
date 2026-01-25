"""Example: JSON-based storage backend for CollectorListener.

This demonstrates how to create a custom storage backend
by subclassing BaseCollectorListener.
"""

import json
import pathlib
from typing import Any

import numpy as np
import pandas as pd

from .baseCollectorListener import BaseCollectorListener


class JSONListener(BaseCollectorListener):
    """Minimal example: Store data as JSON files.

    Usage:
        model = WealthModel(n_agents=20)
        listener = JSONListener(model, output_dir="json_results/")

        # Run simulation
        for _ in range(50):
            model.step()

        # Save to JSON files
        listener.save_to_json()

        print(listener.summary())

        # Get as DataFrame
        wealth_df = listener.get_table_dataframe("wealth")
    """

    def __init__(
        self, model, config: dict[str, dict[str, Any]] | None = None, output_dir="."
    ):
        """Initialize JSON Listener."""
        self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data: dict[str, list] = {}
        super().__init__(model, config)

    def _initialize_dataset_storage(self, dataset_name: str, dataset: Any) -> None:
        """Initialize empty list for dataset."""
        self.data[dataset_name] = []

    def _store_dataset_snapshot(self, dataset_name: str, step: int, data: Any) -> None:
        """Store snapshot as dict."""
        match data:
            case np.ndarray():
                self.data[dataset_name].append({"step": step, "data": data.tolist()})
            case list() | dict():
                self.data[dataset_name].append({"step": step, "data": data})

    def get_table_dataframe(self, name: str) -> pd.DataFrame:
        """Convert JSON data to DataFrame."""
        if name not in self.data:
            raise KeyError(f"Dataset '{name}' not found")

        # Simple conversion
        # FIXME Make it smarter
        records = []
        for snapshot in self.data[name]:
            step = snapshot["step"]
            data = snapshot["data"]
            if isinstance(data, list) and data and isinstance(data[0], dict):
                for row in data:
                    records.append({**row, "step": step})
            elif isinstance(data, dict):
                records.append({**data, "step": step})

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
                json.dump(snapshots, f, indent=2)
