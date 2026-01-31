"""Example: SQL-based storage backend for CollectorListener."""

import sqlite3
from typing import Any

import numpy as np
import pandas as pd

from .baseCollectorListener import BaseCollectorListener, DatasetConfig


class SQLListener(BaseCollectorListener):
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
        try:
            dataset = self.registry.datasets[dataset_name]
            columns = [col for col in dataset._attributes if col != "time"]
        except (AttributeError, KeyError):
            n_cols = data.shape[1] if data.ndim > 1 else 1
            columns = [f"col_{i}" for i in range(n_cols)]

        if not self.metadata[dataset_name]["table_created"]:
            col_defs = ", ".join([f'"{col}" REAL' for col in columns])
            self.conn.execute(
                f'CREATE TABLE IF NOT EXISTS "{dataset_name}" (time REAL, {col_defs})'
            )
            self.metadata[dataset_name]["table_created"] = True
            self.metadata[dataset_name]["columns"] = columns

        if data.shape[1] > len(columns):
            all_cols = list(self.registry.datasets[dataset_name]._attributes)
            df = pd.DataFrame(data, columns=all_cols)
            if "time" in df.columns:
                df = df.drop(columns=["time"])
        else:
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

    def __del__(self):
        """Close database connection on cleanup."""
        super().__del__()
        if hasattr(self, "conn"):
            self.conn.close()
