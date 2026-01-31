"""Example: SQL-based storage backend for CollectorListener.

This demonstrates how to create a custom storage backend
by subclassing BaseCollectorListener.
"""

import sqlite3
from typing import Any

import numpy as np
import pandas as pd

from .collectorlistener import BaseCollectorListener


class SQLListener(BaseCollectorListener):
    """Store collected data in SQLite database.

    Usage:
        model = WealthModel(n_agents=100)

        # Create listener with SQL storage
        listener = SQLListener(model, db_path="simulation.db")

        # Run simulation
        for _ in range(200):
            model.step()

        # Query with SQL
        avg_wealth = listener.query(
            "SELECT step, AVG(wealth) as avg_wealth FROM wealth GROUP BY step ORDER BY step"
        )
        print(avg_wealth.head())

        # Filter data efficiently
        recent_wealthy = listener.query(
            "SELECT * FROM wealth WHERE step > 150 AND wealth > 5"
        )

        # Or get full DataFrame
        wealth_df = listener.get_table_dataframe("wealth")

    """

    def __init__(
        self,
        model,
        config: dict[str, dict[str, Any]] | None = None,
        db_path: str = ":memory:",
    ):
        """Initialize SQL storage backend.

        Args:
            model: Mesa model instance
            config: Per-dataset configuration
            db_path: Path to SQLite database (":memory:" for in-memory)
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.metadata: dict[str, dict] = {}

        super().__init__(model, config)

    def _initialize_dataset_storage(self, dataset_name: str, dataset: Any) -> None:
        """Create a table for each dataset."""
        # We'll create the table on first insert when we know the schema
        self.metadata[dataset_name] = {"table_created": False, "columns": None}

    def _store_dataset_snapshot(self, dataset_name: str, step: int, data: Any) -> None:
        """Store snapshot in SQL table."""
        match data:
            case np.ndarray() if data.size > 0:
                self._store_numpy_data(dataset_name, step, data)

            case list() if data:
                self._store_list_data(dataset_name, step, data)

            case dict():
                self._store_dict_data(dataset_name, step, data)

    def _store_numpy_data(self, dataset_name: str, step: int, data: np.ndarray):
        """Store numpy array as SQL records."""
        # Get column names from dataset
        try:
            dataset = self.registry.datasets[dataset_name]
            columns = list(dataset._attributes)
        except (AttributeError, KeyError):
            n_cols = data.shape[1] if data.ndim > 1 else 1
            columns = [f"col_{i}" for i in range(n_cols)]

        # Create table on first insert
        if not self.metadata[dataset_name]["table_created"]:
            col_defs = ", ".join([f"{col} REAL" for col in columns])
            self.conn.execute(
                f'CREATE TABLE "{dataset_name}" (step INTEGER, {col_defs})'
            )
            self.metadata[dataset_name]["table_created"] = True
            self.metadata[dataset_name]["columns"] = columns

        # Convert to DataFrame and insert
        df = pd.DataFrame(data, columns=columns)
        df["step"] = step
        df.to_sql(dataset_name, self.conn, if_exists="append", index=False)

    def _store_list_data(self, dataset_name: str, step: int, data: list[dict]):
        """Store list of dicts as SQL records."""
        if not self.metadata[dataset_name]["table_created"]:
            # Infer schema from first record
            columns = list(data[0].keys())
            col_defs = ", ".join([f"{col} REAL" for col in columns])
            self.conn.execute(
                f'CREATE TABLE "{dataset_name}" (step INTEGER, {col_defs})'
            )
            self.metadata[dataset_name]["table_created"] = True
            self.metadata[dataset_name]["columns"] = columns

        # Add step to each record
        rows = [{**row, "step": step} for row in data]
        df = pd.DataFrame(rows)
        df.to_sql(dataset_name, self.conn, if_exists="append", index=False)

    def _store_dict_data(self, dataset_name: str, step: int, data: dict):
        """Store single dict as SQL record."""
        if not self.metadata[dataset_name]["table_created"]:
            columns = list(data.keys())
            col_defs = ", ".join([f"{col} REAL" for col in columns])
            self.conn.execute(
                f'CREATE TABLE "{dataset_name}" (step INTEGER, {col_defs})'
            )
            self.metadata[dataset_name]["table_created"] = True
            self.metadata[dataset_name]["columns"] = columns

        row = {**data, "step": step}
        df = pd.DataFrame([row])
        df.to_sql(dataset_name, self.conn, if_exists="append", index=False)

    def get_table_dataframe(self, name: str) -> pd.DataFrame:
        """Retrieve data from SQL table as DataFrame."""
        if name not in self.metadata:
            raise KeyError(f"Dataset '{name}' not found")

        if not self.metadata[name]["table_created"]:
            # Table not created yet - no data collected
            return pd.DataFrame()

        return pd.read_sql(f'SELECT * FROM "{name}"', self.conn)  # noqa: S608

    def query(self, sql: str) -> pd.DataFrame:
        """Execute arbitrary SQL query.

        Args:
            sql: SQL query string

        Returns:
            Query results as DataFrame

        Example:
            df = listener.query("SELECT AVG(wealth) as avg_wealth FROM wealth GROUP BY step")
        """
        return pd.read_sql(sql, self.conn)

    def clear(self, dataset_name: str | None = None) -> None:
        """Clear stored data by dropping tables."""
        if dataset_name is None:
            # Clear all tables
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
