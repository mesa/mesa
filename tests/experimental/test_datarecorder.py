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
