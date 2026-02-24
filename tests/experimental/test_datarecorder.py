"""Tests for DataRecorders."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from mesa.agent import Agent
from mesa.experimental.data_collection import (
    DataRecorder,
    DataRegistry,
    DataSet,
    DatasetConfig,
    JSONDataRecorder,
    ParquetDataRecorder,
    SQLDataRecorder,
)
from mesa.experimental.data_collection.datarecorders import NumpyJSONEncoder
from mesa.experimental.mesa_signals import ModelSignals
from mesa.model import Model


class MockAgent(Agent):
    """A simple agent for testing."""

    def __init__(self, model, value):
        """Initialize the agent."""
        super().__init__(model)
        self.value = value
        self.other_value = value * 2


class MockModel(Model):
    """A simple model for testing."""

    def __init__(self, n=10):
        """Initialize the model."""
        super().__init__()
        self.n = n
        self.model_val = 0

        self.data_registry.track_model(self, "model_data", fields=["model_val"])

        self.data_registry.track_agents_numpy(
            MockAgent, "numpy_data", fields=["value", "other_value"], n=n, dtype=float
        )
        agents = MockAgent.create_agents(self, n, list(range(n)))

        self.data_registry.track_agents(agents, "agent_data", fields=["value"])


class CustomDataType:
    """Custom data type for testing fallback case."""

    def __init__(self, data):
        """Init."""
        self.data = data


def test_dataset_config_validation():
    """Test window_size validation."""
    # Valid window size
    config = DatasetConfig(window_size=100)
    assert config.window_size == 100

    # None is valid
    config = DatasetConfig(window_size=None)
    assert config.window_size is None

    # Invalid window sizes
    with pytest.raises(ValueError):
        DatasetConfig(window_size=0)

    with pytest.raises(ValueError):
        DatasetConfig(window_size=-10)


def test_dataset_config_interval_validation():
    """Test interval validation with zero and negative values."""
    with pytest.raises(ValueError):
        DatasetConfig(interval=0)

    with pytest.raises(ValueError):
        DatasetConfig(interval=-5)


def test_dataset_config_time_validation():
    """Test start_time and end_time validation."""
    with pytest.raises(ValueError):
        DatasetConfig(start_time=-1)

    with pytest.raises(ValueError):
        DatasetConfig(start_time=4, end_time=2)


def test_dataset_config_should_collect_disabled():
    """Test should_collect returns False when disabled."""
    config = DatasetConfig(enabled=False)
    assert not config.should_collect(0)
    assert not config.should_collect(100)


def test_dataset_config_should_collect_after_end_time():
    """Test should_collect returns False after end_time."""
    config = DatasetConfig(start_time=0, end_time=10)
    assert config.should_collect(5)  # Within range
    assert not config.should_collect(11)  # After end_time


def test_dataset_config_update_next_collection_auto_disable():
    """Test that updating next_collection auto-disables at end_time."""
    config = DatasetConfig(interval=5, start_time=0, end_time=20)
    assert config.enabled

    # Update to time that would schedule next collection beyond end_time
    config.update_next_collection(18)
    assert config._next_collection == 23
    assert not config.enabled  # Should auto-disable


def test_base_recorder_no_registry():
    """Test that BaseDataRecorder raises error without DataRegistry."""
    model = Model()
    delattr(model, "data_registry")

    with pytest.raises(AttributeError):
        DataRecorder(model)


def test_base_recorder_config_dict_vs_dataclass():
    """Test that config can be passed as dict or DatasetConfig."""
    model = MockModel(n=5)

    # Test with dict
    config1 = {"model_data": {"interval": 2, "start_time": 5}}
    recorder1 = DataRecorder(model, config=config1)
    assert recorder1.configs["model_data"].interval == 2
    assert recorder1.configs["model_data"].start_time == 5

    # Test with DatasetConfig object
    config2 = {"model_data": DatasetConfig(interval=3, start_time=10)}
    recorder2 = DataRecorder(model, config=config2)
    assert recorder2.configs["model_data"].interval == 3
    assert recorder2.configs["model_data"].start_time == 10


def test_base_recorder_enable_disable_dataset():
    """Test enabling and disabling datasets."""
    model = MockModel(n=5)
    recorder = DataRecorder(model, {"model_data": DatasetConfig()})

    # Disable a dataset
    recorder.disable_dataset("model_data")
    assert not recorder.configs["model_data"].enabled

    # Enable it again
    recorder.enable_dataset("model_data")
    assert recorder.configs["model_data"].enabled

    # Test with nonexistent dataset
    with pytest.raises(KeyError):
        recorder.enable_dataset("nonexistent")

    with pytest.raises(KeyError):
        recorder.disable_dataset("nonexistent")


def test_base_recorder_manual_collect():
    """Test manual collection via collect() method."""
    model = MockModel(n=5)
    recorder = DataRecorder(
        model, {"model_data": DatasetConfig(), "agent_data": DatasetConfig()}
    )
    recorder.clear()

    # Manually trigger collection
    recorder.collect()
    recorder.finalise()

    # Should have collected data
    assert len(recorder.storage["model_data"].blocks) > 0
    assert len(recorder.storage["agent_data"].blocks) > 0


def test_base_recorder_get_all_dataframes():
    """Test get_all_dataframes method."""
    model = MockModel(n=5)
    recorder = DataRecorder(
        model,
        {
            "model_data": DatasetConfig(),
            "agent_data": DatasetConfig(),
            "numpy_data": DatasetConfig(),
        },
    )

    model.step()

    dfs = recorder.get_all_dataframes()

    assert "model_data" in dfs
    assert "agent_data" in dfs
    assert "numpy_data" in dfs
    assert isinstance(dfs["model_data"], pd.DataFrame)
    assert isinstance(dfs["agent_data"], pd.DataFrame)
    assert isinstance(dfs["numpy_data"], pd.DataFrame)


def test_data_recorder_custom_data_type():
    """Test DataRecorder with custom/unknown data type (fallback case)."""
    model = Model()
    model.data_registry = DataRegistry()

    # Create a custom dataset that returns an unknown type
    custom_dataset = Mock(spec=DataSet)
    custom_dataset.name = "custom_data"
    custom_dataset.data = CustomDataType("test_data")
    model.data_registry.datasets["custom_data"] = custom_dataset

    recorder = DataRecorder(model, {"custom_data": DatasetConfig()})
    recorder.clear()

    # Manually trigger collection
    recorder.collect()

    # Should store as custom type
    storage = recorder.storage["custom_data"]
    assert storage.metadata["type"] == "custom"
    assert len(storage.blocks) > 0


def test_data_recorder_empty_numpy_array():
    """Test storing empty numpy array."""
    model = MockModel(n=0)  # No agents
    recorder = DataRecorder(model, {"numpy_data": DatasetConfig()})
    recorder.clear()

    # Try to collect with empty array
    model.step()

    # Should handle gracefully
    df = recorder.get_table_dataframe("numpy_data")
    assert len(df) == 0


def test_data_recorder_empty_list():
    """Test storing empty list (no agents)."""
    model = Model()
    model.data_registry = DataRegistry()
    recorder = DataRecorder(model)
    recorder.clear()

    # Track agents but with no agents
    model.data_registry.track_agents(
        model.agents, "empty_agents", fields=["value"]
    ).record(recorder)

    model.step()

    df = recorder.get_table_dataframe("empty_agents")
    assert len(df) == 0


def test_data_recorder_window_eviction_numpy():
    """Test window eviction bookkeeping for numpy arrays."""
    model = MockModel(n=5)
    recorder = DataRecorder(model, config={"numpy_data": DatasetConfig(window_size=2)})
    recorder.clear()

    # Fill window
    model.step()  # 1
    model.step()  # 2
    model.step()  # 3 - should evict first

    storage = recorder.storage["numpy_data"]
    assert len(storage.blocks) == 2


def test_data_recorder_window_eviction_list():
    """Test window eviction bookkeeping for list data."""
    model = MockModel(n=5)
    recorder = DataRecorder(model, config={"agent_data": DatasetConfig(window_size=2)})
    recorder.clear()

    # Fill window
    model.step()  # 1
    model.step()  # 2
    model.step()  # 3 - should evict first

    storage = recorder.storage["agent_data"]
    assert len(storage.blocks) == 2


def test_data_recorder_window_eviction_dict():
    """Test window eviction bookkeeping for dict data."""
    model = MockModel(n=5)
    recorder = DataRecorder(model, config={"model_data": DatasetConfig(window_size=2)})
    recorder.clear()

    # Fill window
    model.step()  # 1
    model.step()  # 2
    model.step()  # 3 - should evict first

    storage = recorder.storage["model_data"]
    assert len(storage.blocks) == 2


def test_data_recorder_window_eviction_custom():
    """Test window eviction bookkeeping for custom data type."""
    model = Model()
    model.data_registry = DataRegistry()

    custom_dataset = Mock()
    custom_dataset.name = "custom_data"
    custom_dataset.data = CustomDataType("test")
    model.data_registry.datasets["custom_data"] = custom_dataset

    recorder = DataRecorder(model, config={"custom_data": DatasetConfig(window_size=2)})
    recorder.clear()

    # Fill window
    recorder.collect()
    recorder.collect()
    recorder.collect()  # Should evict first

    storage = recorder.storage["custom_data"]
    assert len(storage.blocks) == 2


def test_data_recorder_clear_nonexistent_dataset():
    """Test clearing nonexistent dataset raises KeyError."""
    model = MockModel(n=5)
    recorder = DataRecorder(model)

    with pytest.raises(KeyError):
        recorder.clear("nonexistent")


def test_data_recorder_clear_single_dataset():
    """Test clearing specific dataset."""
    model = MockModel(n=5)
    recorder = DataRecorder(
        model, {"model_data": DatasetConfig(), "agent_data": DatasetConfig()}
    )

    model.step()

    # Clear only model_data
    recorder.clear("model_data")

    assert len(recorder.storage["model_data"].blocks) == 0
    assert recorder.storage["model_data"].total_rows == 0
    assert recorder.storage["model_data"].estimated_size_bytes == 0

    # Other datasets should still have data
    assert len(recorder.storage["agent_data"].blocks) > 0


def test_data_recorder_clear_all_datasets():
    """Test clearing all datasets."""
    model = MockModel(n=5)
    recorder = DataRecorder(
        model,
        {
            "model_data": DatasetConfig(),
            "agent_data": DatasetConfig(),
            "numpy_data": DatasetConfig(),
        },
    )

    model.step()

    # Clear all
    recorder.clear()

    for name in recorder.storage:
        assert len(recorder.storage[name].blocks) == 0
        assert recorder.storage[name].total_rows == 0
        assert recorder.storage[name].estimated_size_bytes == 0


def test_data_recorder_get_table_dataframe_nonexistent():
    """Test get_table_dataframe with nonexistent dataset."""
    model = MockModel(n=5)
    recorder = DataRecorder(model)

    with pytest.raises(KeyError):
        recorder.get_table_dataframe("nonexistent")


def test_data_recorder_get_table_dataframe_empty():
    """Test get_table_dataframe returns empty DataFrame with correct columns."""
    model = MockModel(n=5)
    recorder = DataRecorder(
        model,
        {
            "model_data": DatasetConfig(),
        },
    )
    recorder.collect()
    recorder.clear()

    df = recorder.get_table_dataframe("model_data")

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert "model_val" in df.columns
    assert "time" in df.columns


def test_data_recorder_get_table_dataframe_unknown_type_warning():
    """Test that unknown data types trigger warning."""
    model = Model()
    model.data_registry = DataRegistry()

    custom_dataset = Mock(spec=DataSet)
    custom_dataset.name = "custom_data"
    custom_dataset.data = CustomDataType("test")
    model.data_registry.datasets["custom_data"] = custom_dataset

    recorder = DataRecorder(model, {"custom_data": DatasetConfig()})
    recorder.clear()
    recorder.collect()

    # Manually corrupt the metadata to trigger warning
    recorder.storage["custom_data"].metadata["type"] = "unknown_type"

    with pytest.warns(RuntimeWarning):
        _ = recorder.get_table_dataframe("custom_data")


def test_json_recorder_numpy_types():
    """Test JSONDataRecorder handles numpy types in custom encoder."""
    with tempfile.TemporaryDirectory() as temp_dir:
        model = MockModel(n=2)
        recorder = JSONDataRecorder(
            model,
            {
                "model_data": DatasetConfig(),
                "agent_data": DatasetConfig(),
                "numpy_data": DatasetConfig(),
            },
            output_dir=temp_dir,
        )

        model.step()
        df = recorder.get_table_dataframe("numpy_data")
        assert not df.empty

        recorder.save_to_json()

        # Verify JSON files exist and are valid
        with open(Path(temp_dir) / "numpy_data.json") as f:
            data = json.load(f)
            assert isinstance(data, list)


def test_json_recorder_numpy_encoder_types():
    """Test NumpyJSONEncoder handles various numpy types."""
    encoder = NumpyJSONEncoder()

    # Test int types
    assert encoder.default(np.int32(5)) == 5
    assert encoder.default(np.int64(10)) == 10

    # Test float types
    assert encoder.default(np.float64(2.71)) == pytest.approx(2.71, rel=1e-6)

    # Test bool type
    assert encoder.default(np.bool_(True)) is True
    assert encoder.default(np.bool_(False)) is False

    # Test array type
    arr = np.array([1, 2, 3])
    assert encoder.default(arr) == [1, 2, 3]


def test_json_recorder_clear():
    """Test JSONDataRecorder clear functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        model = MockModel(n=2)
        recorder = JSONDataRecorder(
            model,
            {
                "model_data": DatasetConfig(),
                "agent_data": DatasetConfig(),
                "numpy_data": DatasetConfig(),
            },
            output_dir=temp_dir,
        )

        model.step()
        recorder.save_to_json()

        # Clear specific dataset
        recorder.clear("model_data")
        df = recorder.get_table_dataframe("model_data")
        assert df.empty

        # Clear All
        recorder.clear()
        with pytest.raises(KeyError):
            recorder.get_table_dataframe("agent_data")


def test_json_recorder_summary():
    """Test JSONDataRecorder summary."""
    with tempfile.TemporaryDirectory() as temp_dir:
        model = MockModel(n=2)
        recorder = JSONDataRecorder(model, output_dir=temp_dir)

        model.step()

        summary = recorder.summary()
        assert "datasets" in summary
        assert "output_dir" in summary


def test_parquet_recorder_buffer_and_flush():
    """Test ParquetDataRecorder buffer and flush mechanisms."""
    pytest.importorskip("pyarrow")

    with tempfile.TemporaryDirectory() as temp_dir:
        model = MockModel(n=5)
        recorder = ParquetDataRecorder(
            model,
            {
                "model_data": DatasetConfig(),
                "agent_data": DatasetConfig(),
                "numpy_data": DatasetConfig(),
            },
            output_dir=temp_dir,
        )
        recorder.buffer_size = 100
        recorder.clear("model_data")

        # Add data to buffer without flushing
        model.step()

        # Check buffer has data
        assert len(recorder.buffers["model_data"]) > 0

        # Manually flush
        recorder._flush_buffer("model_data")

        # Buffer should be cleared
        assert len(recorder.buffers["model_data"]) == 0

        # File should exist
        filepath = Path(temp_dir) / "model_data.parquet"
        assert filepath.exists()


def test_parquet_recorder_empty_buffer_flush():
    """Test flushing empty buffer does nothing."""
    pytest.importorskip("pyarrow")

    with tempfile.TemporaryDirectory() as temp_dir:
        model = MockModel(n=5)
        recorder = ParquetDataRecorder(
            model,
            {
                "model_data": DatasetConfig(),
                "agent_data": DatasetConfig(),
                "numpy_data": DatasetConfig(),
            },
            output_dir=temp_dir,
        )
        recorder.clear()

        # Flush empty buffer
        recorder._flush_buffer("model_data")

        # Should not create file
        filepath = Path(temp_dir) / "model_data.parquet"
        assert not filepath.exists()


def test_parquet_recorder_append_to_existing():
    """Test appending to existing parquet file."""
    pytest.importorskip("pyarrow")

    with tempfile.TemporaryDirectory() as temp_dir:
        model = MockModel(n=5)
        recorder = ParquetDataRecorder(
            model,
            {
                "model_data": DatasetConfig(),
                "agent_data": DatasetConfig(),
                "numpy_data": DatasetConfig(),
            },
            output_dir=temp_dir,
        )
        recorder.buffer_size = 1
        recorder.clear()

        # First write
        model.step()

        # Second write (should append)
        model.step()

        df = recorder.get_table_dataframe("model_data")
        assert len(df) >= 2


def test_parquet_recorder_get_nonexistent_dataset():
    """Test getting nonexistent dataset from parquet."""
    pytest.importorskip("pyarrow")

    with tempfile.TemporaryDirectory() as temp_dir:
        model = MockModel(n=5)
        recorder = ParquetDataRecorder(model, output_dir=temp_dir)

        with pytest.raises(KeyError):
            recorder.get_table_dataframe("nonexistent")


def test_parquet_recorder_get_nonexistent_file():
    """Test getting dataset when file doesn't exist."""
    pytest.importorskip("pyarrow")

    with tempfile.TemporaryDirectory() as temp_dir:
        model = MockModel(n=5)
        recorder = ParquetDataRecorder(
            model,
            {
                "model_data": DatasetConfig(),
                "agent_data": DatasetConfig(),
                "numpy_data": DatasetConfig(),
            },
            output_dir=temp_dir,
        )
        recorder.clear()

        # Didn't step, so no data written
        df = recorder.get_table_dataframe("model_data")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


def test_parquet_recorder_clear_nonexistent():
    """Test clearing nonexistent dataset."""
    pytest.importorskip("pyarrow")

    with tempfile.TemporaryDirectory() as temp_dir:
        model = MockModel(n=5)
        recorder = ParquetDataRecorder(model, output_dir=temp_dir)

        with pytest.raises(KeyError):
            recorder.clear("nonexistent")


def test_parquet_recorder_summary_with_files():
    """Test summary with existing parquet files."""
    pytest.importorskip("pyarrow")

    with tempfile.TemporaryDirectory() as temp_dir:
        model = MockModel(n=5)
        recorder = ParquetDataRecorder(
            model,
            {
                "model_data": DatasetConfig(),
                "agent_data": DatasetConfig(),
                "numpy_data": DatasetConfig(),
            },
            output_dir=temp_dir,
        )
        recorder.buffer_size = 1
        recorder.clear()

        model.step()

        summary = recorder.summary()

        assert "output_dir" in summary
        assert "model_data" in summary
        assert summary["model_data"]["rows"] > 0
        assert "size_mb" in summary["model_data"]


def test_parquet_recorder_summary_no_files():
    """Test summary when no files exist yet."""
    pytest.importorskip("pyarrow")

    with tempfile.TemporaryDirectory() as temp_dir:
        model = MockModel(n=5)
        recorder = ParquetDataRecorder(
            model,
            {
                "model_data": DatasetConfig(),
                "agent_data": DatasetConfig(),
                "numpy_data": DatasetConfig(),
            },
            output_dir=temp_dir,
        )
        recorder.clear()

        summary = recorder.summary()

        assert summary["model_data"]["size_mb"] == 0
        assert summary["model_data"]["rows"] == 0


def test_parquet_recorder_cleanup_on_delete():
    """Test that __del__ flushes buffers."""
    pytest.importorskip("pyarrow")

    with tempfile.TemporaryDirectory() as temp_dir:
        model = MockModel(n=5)
        recorder = ParquetDataRecorder(
            model,
            {
                "model_data": DatasetConfig(),
                "agent_data": DatasetConfig(),
                "numpy_data": DatasetConfig(),
            },
            output_dir=temp_dir,
        )
        recorder.buffer_size = 100
        recorder.clear()

        model.step()

        del recorder

        # File should exist (buffer was flushed)
        filepath = Path(temp_dir) / "model_data.parquet"
        assert filepath.exists()


def test_parquet_recorder_dict_data_storage():
    """Test storing dict data (model data) in parquet."""
    pytest.importorskip("pyarrow")

    with tempfile.TemporaryDirectory() as temp_dir:
        model = MockModel(n=5)
        recorder = ParquetDataRecorder(
            model,
            {
                "model_data": DatasetConfig(),
                "agent_data": DatasetConfig(),
                "numpy_data": DatasetConfig(),
            },
            output_dir=temp_dir,
        )
        recorder.buffer_size = 1
        recorder.clear()

        model.step()

        # Check dict was stored correctly
        df = recorder.get_table_dataframe("model_data")
        assert "model_val" in df.columns
        assert "time" in df.columns


def test_parquet_recorder_list_data_storage():
    """Test storing list data (agent data) in parquet."""
    pytest.importorskip("pyarrow")

    with tempfile.TemporaryDirectory() as temp_dir:
        model = MockModel(n=5)
        recorder = ParquetDataRecorder(
            model,
            {
                "model_data": DatasetConfig(),
                "agent_data": DatasetConfig(),
                "numpy_data": DatasetConfig(),
            },
            output_dir=temp_dir,
        )
        recorder.clear()

        model.step()
        recorder.collect()  # force collect after step

        # Check list was stored correctly
        df = recorder.get_table_dataframe("agent_data")
        assert "value" in df.columns
        assert "time" in df.columns
        assert len(df) == 10


def test_sql_recorder_store_empty_numpy():
    """Test SQL recorder with empty numpy array."""
    model = MockModel(n=0)  # No agents
    recorder = SQLDataRecorder(
        model,
        {
            "model_data": DatasetConfig(),
            "agent_data": DatasetConfig(),
            "numpy_data": DatasetConfig(),
        },
        db_path=":memory:",
    )

    model.step()

    # Should handle gracefully
    df = recorder.get_table_dataframe("numpy_data")
    assert len(df) == 0


def test_sql_recorder_store_empty_list():
    """Test SQL recorder with empty list."""
    model = Model()
    model.data_registry = DataRegistry()

    recorder = SQLDataRecorder(model, db_path=":memory:")
    model.data_registry.track_agents(
        model.agents, "empty_agents", fields=["value"]
    ).record(recorder)

    model.step()

    # Should handle gracefully (no table created)
    df = recorder.get_table_dataframe("empty_agents")
    assert len(df) == 0


def test_sql_recorder_numpy_without_dataset():
    """Test SQL recorder stores numpy data when dataset not in registry."""
    model = MockModel(n=2)
    recorder = SQLDataRecorder(
        model,
        {
            "model_data": DatasetConfig(),
            "agent_data": DatasetConfig(),
            "numpy_data": DatasetConfig(),
        },
        db_path=":memory:",
    )

    # Manually store data
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    recorder._store_dataset_snapshot("numpy_data", 0, data)

    df = recorder.get_table_dataframe("numpy_data")
    assert len(df) == 2
    assert "value" in df.columns


def test_sql_recorder_numpy_with_time_column():
    """Test SQL recorder with Numpy data."""
    model = MockModel(n=2)
    recorder = SQLDataRecorder(
        model,
        {
            "model_data": DatasetConfig(),
            "agent_data": DatasetConfig(),
            "numpy_data": DatasetConfig(),
        },
        db_path=":memory:",
    )

    model.step()

    df = recorder.get_table_dataframe("numpy_data")
    assert not df.empty
    assert "time" in df.columns
    assert "agent_id" in df.columns


def test_sql_recorder_get_nonexistent_dataset():
    """Test getting nonexistent dataset."""
    model = MockModel(n=5)
    recorder = SQLDataRecorder(model, db_path=":memory:")

    with pytest.raises(KeyError):
        recorder.get_table_dataframe("nonexistent")


def test_sql_recorder_get_empty_dataset():
    """Test getting dataset when table not created."""
    model = MockModel(n=5)
    recorder = SQLDataRecorder(
        model,
        {
            "model_data": DatasetConfig(),
            "agent_data": DatasetConfig(),
            "numpy_data": DatasetConfig(),
        },
        db_path=":memory:",
    )
    recorder.clear("model_data")

    df = recorder.get_table_dataframe("model_data")
    assert len(df) == 0

    model.step()
    recorder.clear()
    df = recorder.get_table_dataframe("model_data")
    assert len(df) == 0


def test_sql_recorder_clear_nonexistent():
    """Test clearing nonexistent dataset."""
    model = MockModel(n=5)
    recorder = SQLDataRecorder(model, db_path=":memory:")

    with pytest.raises(KeyError):
        recorder.clear("nonexistent")


def test_sql_recorder_summary_no_tables():
    """Test summary when tables not created."""
    model = MockModel(n=5)
    recorder = SQLDataRecorder(
        model,
        {
            "model_data": DatasetConfig(),
            "agent_data": DatasetConfig(),
            "numpy_data": DatasetConfig(),
        },
        db_path=":memory:",
    )
    recorder.collect()
    summary = recorder.summary()

    assert summary["model_data"]["rows"] == 1


def test_sql_recorder_cleanup_on_delete():
    """Test connection cleanup on __del__."""
    model = MockModel(n=5)
    recorder = SQLDataRecorder(model, db_path=":memory:")

    conn = recorder.conn

    # Delete recorder
    del recorder

    # Connection should be closed (this will raise an error)
    with pytest.raises(Exception):
        conn.execute("SELECT 1")


def test_sql_recorder_with_file_database():
    """Test SQL recorder with file database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        model = MockModel(n=5)
        recorder = SQLDataRecorder(
            model,
            {
                "model_data": DatasetConfig(),
                "agent_data": DatasetConfig(),
                "numpy_data": DatasetConfig(),
            },
            db_path=db_path,
        )

        model.step()

        df = recorder.get_table_dataframe("model_data")
        assert len(df) > 0

        recorder.conn.close()

        # File should exist
        assert os.path.exists(db_path)
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_recorder_with_disabled_dataset():
    """Test that disabled datasets are not collected."""
    model = MockModel(n=5)

    config = {
        "model_data": DatasetConfig(enabled=False),
        "agent_data": DatasetConfig(enabled=True),
    }

    recorder = DataRecorder(model, config=config)
    recorder.clear()

    model.step()

    # Disabled dataset should have no data
    assert len(recorder.storage["model_data"].blocks) == 0

    # Enabled dataset should have data
    assert len(recorder.storage["agent_data"].blocks) > 0


def test_recorder_end_time_behavior():
    """Test that collection stops at end_time."""
    model = MockModel(n=5)

    config = {"model_data": DatasetConfig(interval=1, start_time=0, end_time=2)}

    recorder = DataRecorder(model, config=config)
    recorder.clear()

    # Step through time
    model.step()  # t=1
    model.step()  # t=2
    model.step()  # t=3 (should not collect)

    # Should only have data from t=1 -> t=2
    df = recorder.get_table_dataframe("model_data")
    times = df["time"].unique()
    assert 1.0 in times
    assert 2.0 in times
    assert 3.0 not in times


def test_recorder_start_time_behavior():
    """Test that collection starts at start_time."""
    model = MockModel(n=5)

    config = {"model_data": DatasetConfig(interval=1, start_time=2)}

    recorder = DataRecorder(model, config=config)

    # Initial collection should not happen (t=0 < start_time=2)
    assert len(recorder.storage["model_data"].blocks) == 0

    # Step to start_time
    model.step()  # t=1 (should not collect)
    model.step()  # t=2 (should not collect)
    model.step()  # the change to t=3 triggers a single collect for t=2

    df = recorder.get_table_dataframe("model_data")
    times = df["time"].unique()
    assert 1.0 not in times
    assert 2.0 in times


# --- RUN_ENDED signal and auto-finalize tests ---


class SteppingAgent(Agent):
    """A separate agent class for RUN_ENDED tests (avoids NumpyAgentDataSet conflicts)."""

    def __init__(self, model, value):
        """Initialize the agent."""
        super().__init__(model)
        self.value = value


class SteppingModel(Model):
    """A model that stops after a configurable number of steps."""

    def __init__(self, max_steps=5, n=3):
        """Initialize the model."""
        super().__init__()
        self.max_steps = max_steps
        self.model_val = 0

        self.data_registry.track_model(self, "model_data", fields=["model_val"])

        agents = SteppingAgent.create_agents(self, n, list(range(n)))
        self.data_registry.track_agents(agents, "agent_data", fields=["value"])

    def step(self):
        """Increment model_val and stop after max_steps."""
        self.model_val += 1
        for agent in self.agents:
            agent.value += 1
        if self.model_val >= self.max_steps:
            self.running = False


def test_run_ended_signal_exists():
    """Test that RUN_ENDED signal type exists in ModelSignals."""
    assert hasattr(ModelSignals, "RUN_ENDED")
    assert ModelSignals.RUN_ENDED == "run_ended"


def test_run_ended_signal_is_observable():
    """Test that 'model' is registered as an observable on Model."""
    model = SteppingModel()
    assert "model" in model.observables


def test_run_model_emits_run_ended():
    """Test that run_model() emits the RUN_ENDED signal."""
    model = SteppingModel(max_steps=3)

    signal_received = []

    def handler(signal):
        signal_received.append(signal)

    model.observe("model", ModelSignals.RUN_ENDED, handler)
    model.run_model()

    assert len(signal_received) == 1
    assert signal_received[0].signal_type == ModelSignals.RUN_ENDED


def test_run_ended_auto_finalise_captures_final_state():
    """Test that RUN_ENDED auto-finalise captures the final simulation state.

    Without RUN_ENDED, the last time point is not recorded because
    _on_time_change only records at the OLD time. This test verifies
    that the final state at the current model time is now captured.
    """
    model = SteppingModel(max_steps=5)
    recorder = DataRecorder(
        model,
        {"model_data": DatasetConfig(), "agent_data": DatasetConfig()},
    )
    recorder.clear()

    model.run_model()

    # The final time should be 5.0 (5 steps from time 0)
    df_model = recorder.get_table_dataframe("model_data")
    times = sorted(df_model["time"].unique())

    # Time 5.0 (the final state) should be captured via RUN_ENDED
    assert 5.0 in times, f"Final time 5.0 not found in recorded times: {times}"

    # The final model_val should be 5
    final_row = df_model[df_model["time"] == 5.0]
    assert final_row["model_val"].values[0] == 5

    # Agent data should also have the final state
    df_agent = recorder.get_table_dataframe("agent_data")
    agent_times = sorted(df_agent["time"].unique())
    assert 5.0 in agent_times


def test_finalise_deduplication():
    """Test that calling finalise() multiple times at the same time is idempotent."""
    model = SteppingModel(max_steps=3)
    recorder = DataRecorder(model, {"model_data": DatasetConfig()})
    recorder.clear()

    model.run_model()

    # run_model already triggered finalise via RUN_ENDED
    df_before = recorder.get_table_dataframe("model_data")
    count_before = len(df_before)

    # Calling finalise again should NOT add duplicate data
    recorder.finalise()
    df_after = recorder.get_table_dataframe("model_data")
    count_after = len(df_after)

    assert count_before == count_after, (
        f"Duplicate data added: {count_before} rows before, {count_after} after"
    )


def test_finalise_deduplication_resets_on_new_time():
    """Test that dedup tracking allows new snapshots at new times."""
    model = SteppingModel(max_steps=10)
    recorder = DataRecorder(model, {"model_data": DatasetConfig()})
    recorder.clear()

    # Run for 3 steps manually
    model.step()  # t=1
    model.step()  # t=2
    model.step()  # t=3

    # Finalise at t=3
    recorder.finalise()
    df1 = recorder.get_table_dataframe("model_data")
    count_at_3 = len(df1[df1["time"] == 3.0])
    assert count_at_3 == 1

    # Advance further
    model.step()  # t=4

    # Finalise at t=4 should work (new time)
    recorder.finalise()
    df2 = recorder.get_table_dataframe("model_data")
    assert 4.0 in df2["time"].values


def test_run_ended_with_manual_stepping():
    """Test that run_model() emits RUN_ENDED even with few steps."""
    model = SteppingModel(max_steps=1)
    recorder = DataRecorder(model, {"model_data": DatasetConfig()})
    recorder.clear()

    model.run_model()

    df = recorder.get_table_dataframe("model_data")
    # Should have captured the final state at t=1.0
    assert 1.0 in df["time"].values


def test_run_ended_disabled_dataset_not_collected():
    """Test that disabled datasets are not collected on RUN_ENDED."""
    model = SteppingModel(max_steps=3)
    recorder = DataRecorder(
        model,
        {
            "model_data": DatasetConfig(enabled=True),
            "agent_data": DatasetConfig(enabled=False),
        },
    )
    recorder.clear()

    model.run_model()

    # model_data should have data (enabled)
    assert len(recorder.storage["model_data"].blocks) > 0

    # agent_data should have no data (disabled)
    assert len(recorder.storage["agent_data"].blocks) == 0


def test_run_ended_with_json_recorder():
    """Test that RUN_ENDED works with JSONDataRecorder."""
    with tempfile.TemporaryDirectory() as temp_dir:
        model = SteppingModel(max_steps=3)
        recorder = JSONDataRecorder(
            model,
            {"model_data": DatasetConfig()},
            output_dir=temp_dir,
        )

        model.run_model()

        df = recorder.get_table_dataframe("model_data")
        times = sorted(df["time"].unique())
        assert 3.0 in times, f"Final time 3.0 not in JSON recorder times: {times}"


def test_run_ended_with_sql_recorder():
    """Test that RUN_ENDED works with SQLDataRecorder."""
    model = SteppingModel(max_steps=3)
    recorder = SQLDataRecorder(
        model,
        {"model_data": DatasetConfig()},
        db_path=":memory:",
    )

    model.run_model()

    df = recorder.get_table_dataframe("model_data")
    times = sorted(df["time"].unique())
    assert 3.0 in times, f"Final time 3.0 not in SQL recorder times: {times}"


def test_run_ended_data_completeness():
    """Test that all time points are captured across a full run.

    Before this fix, using run_model() with periodic time-change collection
    would miss the final time point. Now with RUN_ENDED, the complete
    timeline should be captured.
    """
    max_steps = 5
    model = SteppingModel(max_steps=max_steps)
    recorder = DataRecorder(model, {"model_data": DatasetConfig()})
    recorder.clear()

    model.run_model()

    df = recorder.get_table_dataframe("model_data")
    recorded_times = sorted(df["time"].unique())

    # _on_time_change records at old time (0, 1, 2, 3, 4)
    # RUN_ENDED finalise records at current time (5)
    # So we should have all times from 0 through 5
    for t in range(max_steps + 1):
        assert float(t) in recorded_times, (
            f"Time {t} missing from recorded times: {recorded_times}"
        )


def test_run_ended_model_val_progression():
    """Test that model values progress correctly across recorded snapshots."""
    model = SteppingModel(max_steps=4)
    recorder = DataRecorder(model, {"model_data": DatasetConfig()})
    recorder.clear()

    model.run_model()

    df = recorder.get_table_dataframe("model_data")
    df_sorted = df.sort_values("time")

    # model_val increments by 1 each step
    # t=0: model_val=0 (before any step), t=1: model_val=1, ..., t=4: model_val=4
    for _, row in df_sorted.iterrows():
        t = row["time"]
        expected_val = int(t)
        assert row["model_val"] == expected_val, (
            f"At time {t}, expected model_val={expected_val}, got {row['model_val']}"
        )


def test_step_without_run_model_no_run_ended():
    """Test that manual step() calls do not emit RUN_ENDED."""
    model = SteppingModel(max_steps=10)

    signal_received = []

    def handler(signal):
        signal_received.append(signal)

    model.observe("model", ModelSignals.RUN_ENDED, handler)

    # Manual steps should not emit RUN_ENDED
    model.step()
    model.step()
    model.step()

    assert len(signal_received) == 0


def test_multiple_recorders_both_receive_run_ended():
    """Test that multiple recorders all receive the RUN_ENDED signal."""
    model = SteppingModel(max_steps=3)
    recorder1 = DataRecorder(model, {"model_data": DatasetConfig()})
    recorder2 = DataRecorder(model, {"model_data": DatasetConfig()})
    recorder1.clear()
    recorder2.clear()

    model.run_model()

    df1 = recorder1.get_table_dataframe("model_data")
    df2 = recorder2.get_table_dataframe("model_data")

    assert 3.0 in df1["time"].values
    assert 3.0 in df2["time"].values
