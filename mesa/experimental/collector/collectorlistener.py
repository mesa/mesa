"""High-performance Listener for the DataRegistry Architecture.

This module acts as the 'Conductor', triggering the DataRegistry to extract
data at the right time and storing it efficiently using Lazy Block Storage.
"""

from __future__ import annotations

import contextlib
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from mesa.experimental.devs import Priority

if TYPE_CHECKING:
    from mesa.model import Model

# Optional Polars import for high-performance output generation
with contextlib.suppress(ImportError):
    import polars as pl


class CollectorListener:
    """Orchestrates data collection using the Model's scheduler.

    It binds to the model and schedules a recurring collection event that
    runs after the model step is complete.
    """

    def __init__(self, model: Model):
        """Initialize the listener and schedule the first collection."""
        self.model = model

        # Ensure the Registry exists
        if not hasattr(model, "data_registry"):
            raise AttributeError(
                "CollectorListener requires 'model.data' to be a DataRegistry."
            )
        self.registry = model.data_registry

        # Storage: Dictionary of Lists
        # Structure: {"wealth": [(step, block), ...], "gini": [{row}, ...]}
        self.tables: dict[str, list[Any]] = defaultdict(list)

        # Scheduling: Start the recursive loop
        # Schedule at time=1 because model.step() usually increments time to 1.0 first.
        # Priority.LOW ensures we run AFTER the agents have moved.
        self.model.schedule_at(self._collect, time=1, priority=Priority.LOW)

    def _collect(self):
        """The main collection callback."""
        current_step = self.model.steps

        # Iterate over all configured datasets in the registry
        for name, dataset in self.registry.datasets.items():
            # Trigger the compiled logic in statistics.py
            collected_data = dataset.data

            # Store tuple (Step, Block) for O(1) performance.
            if isinstance(collected_data, list):
                if collected_data:
                    self.tables[name].append((current_step, collected_data))

            # Numpy Agent Data (Numpy Array)
            # Store tuple (Step, Array Copy). We MUST copy to prevent mutation.
            elif isinstance(collected_data, np.ndarray):
                if collected_data.size > 0:
                    self.tables[name].append((current_step, collected_data.copy()))

            # Model Data (Single Dict)
            # Eagerly inject step because it's cheap (1 row)
            elif isinstance(collected_data, dict):
                row = collected_data.copy()
                row["step"] = current_step
                self.tables[name].append(row)

            # Fallback
            else:
                self.tables[name].append((current_step, collected_data))

        # RECURSION: Schedule the next collection
        with contextlib.suppress(ValueError):
            self.model.schedule_after(self._collect, delay=1, priority=Priority.LOW)

    def clear(self):
        """Clear all stored data. Call this on model reset."""
        self.tables.clear()
        self.model.schedule_at(self._collect, time=1, priority=Priority.LOW)

    def get_table_dataframe(self, name: str) -> pd.DataFrame:
        """Convert the stored data into a DataFrame."""
        if name not in self.tables:
            raise KeyError(f"Table '{name}' not found.")

        raw_data = self.tables[name]
        if not raw_data:
            return pd.DataFrame()

        first_item = raw_data[0]

        # Tuple (Step, Block) -> Agents
        if isinstance(first_item, tuple):
            _, block = first_item

            # Numpy Array
            if isinstance(block, np.ndarray):
                return self._flatten_numpy_data(name, raw_data)

            # List of Dicts
            return self._flatten_standard_agent_data(raw_data)

        # Dict -> Model Data
        if "pl" in globals() and pl is not None:
            return pl.DataFrame(raw_data).to_pandas()
        return pd.DataFrame(raw_data)

    def _flatten_standard_agent_data(self, raw_data: list[tuple[int, list[dict]]]):
        """Explode List-of-Dicts blocks."""
        if "pl" in globals() and pl is not None:
            rows = []
            for step, block in raw_data:
                for row in block:
                    row["step"] = step
                    rows.append(row)
            return pl.DataFrame(rows).to_pandas()

        rows = []
        for step, block in raw_data:
            for row in block:
                row_copy = row.copy()
                row_copy["step"] = step
                rows.append(row_copy)
        return pd.DataFrame(rows)

    def _flatten_numpy_data(self, name: str, raw_data: list[tuple[int, np.ndarray]]):
        """Explode Numpy blocks."""
        dataset = self.registry.datasets[name]
        columns = list(dataset._args)

        all_dfs = []
        for step, array in raw_data:
            df = pd.DataFrame(array, columns=columns)
            df["step"] = step
            all_dfs.append(df)

        return pd.concat(all_dfs, ignore_index=True)
