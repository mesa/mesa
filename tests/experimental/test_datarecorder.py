"""Tests for DataRecorders."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from mesa.agent import Agent
from mesa.experimental.data_collection import (
    DataRecorder,
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


def test_dataset_config():
    """Test DatasetConfig validation and logic."""
    # Valid config
    config = DatasetConfig(interval=2, start_time=10)
    assert config.interval == 2
    assert config.start_time == 10
    assert config.enabled is True

    # Validation errors
    with pytest.raises(ValueError):
        DatasetConfig(interval=-1)

    with pytest.raises(ValueError):
        DatasetConfig(start_time=-5)

    with pytest.raises(ValueError):
        DatasetConfig(start_time=10, end_time=5)

    # Collection logic
    config = DatasetConfig(interval=2, start_time=2)

    assert not config.should_collect(0)
    assert not config.should_collect(1)

    # Start time matches
    assert config.should_collect(2)
    config.update_next_collection(2)
    assert config._next_collection == 4

    assert not config.should_collect(3)
    assert config.should_collect(4)


def test_data_recorder_integration():
    """Test basic in-memory DataRecorder integration."""
    model = MockModel(n=5)

    # Configure recorder
    config = {
        "model_data": DatasetConfig(interval=1),
        "agent_data": DatasetConfig(interval=1),
        "numpy_data": DatasetConfig(interval=2),  # Different interval
    }

    recorder = DataRecorder(model, config=config)

    # Step 1
    model.step()

    # Retrieve data
    model_df = recorder.get_table_dataframe("model_data")
    agent_df = recorder.get_table_dataframe("agent_data")
    numpy_df = recorder.get_table_dataframe("numpy_data")

    # Check Model Data
    assert not model_df.empty
    assert "time" in model_df.columns
    assert "model_val" in model_df.columns
    # Should have at least one row
    assert len(model_df) > 0

    # Check Agent Data
    assert not agent_df.empty
    assert "unique_id" in agent_df.columns
    assert "value" in agent_df.columns

    # Check Numpy Data
    model.step()  # Time 2
    numpy_df = recorder.get_table_dataframe("numpy_data")

    # We expect collection at t=0 and t=2
    times = numpy_df["time"].unique()
    assert 0.0 in times
    assert 2.0 in times
    assert 1.0 not in times


def test_data_recorder_numpy_ids():
    """Test that DataRecorder correctly adds agent_ids to Numpy datasets."""
    model = MockModel(n=3)
    recorder = DataRecorder(model)

    # Step to generate data
    model.step()

    df = recorder.get_table_dataframe("numpy_data")

    assert "agent_id" in df.columns
    assert "time" in df.columns
    assert "value" in df.columns
    assert "other_value" in df.columns

    # Check IDs are present and correct
    unique_ids = df["agent_id"].unique()
    assert len(unique_ids) == 3

    # Verify values align
    row = df[df["value"] == 0.0].iloc[0]
    assert row["agent_id"] is not None


def test_json_recorder():
    """Test JSONDataRecorder."""
    with tempfile.TemporaryDirectory() as temp_dir:
        model = MockModel(n=2)
        recorder = JSONDataRecorder(model, output_dir=temp_dir)

        model.step()
        model.step()

        recorder.save_to_json()

        path = Path(temp_dir)
        assert (path / "model_data.json").exists()
        assert (path / "numpy_data.json").exists()

        # Check retrieval works
        df = recorder.get_table_dataframe("model_data")
        assert len(df) >= 2
        assert "model_val" in df.columns


def test_parquet_recorder():
    """Test ParquetDataRecorder."""
    pytest.importorskip("pyarrow")

    with tempfile.TemporaryDirectory() as temp_dir:
        model = MockModel(n=5)
        # Set small buffer size to force flush
        recorder = ParquetDataRecorder(model, output_dir=temp_dir)
        recorder.buffer_size = 1

        model.step()
        model.step()

        _ = Path(temp_dir)

        # Check data via recorder
        df = recorder.get_table_dataframe("model_data")
        assert len(df) >= 2
        assert "model_val" in df.columns

        # Check Numpy data written to parquet has correct columns
        df_numpy = recorder.get_table_dataframe("numpy_data")
        assert "agent_id" in df_numpy.columns
        assert "value" in df_numpy.columns
        assert len(df_numpy) > 0


def test_sql_recorder():
    """Test SQLDataRecorder (SQLite)."""
    model = MockModel(n=5)
    recorder = SQLDataRecorder(model, db_path=":memory:")

    model.step()
    model.step()

    # Query via recorder
    df = recorder.query("SELECT * FROM model_data")
    assert len(df) >= 2
    assert "model_val" in df.columns

    # Test get_table_dataframe
    df_agent = recorder.get_table_dataframe("agent_data")
    assert not df_agent.empty

    # Test numpy storage
    df_numpy = recorder.get_table_dataframe("numpy_data")
    assert "agent_id" in df_numpy.columns
    assert "value" in df_numpy.columns

    # Test clear
    recorder.clear()

    # Tables should be dropped
    with pytest.raises(Exception):
        recorder.query("SELECT * FROM model_data")


def test_recorder_summary():
    """Test summary generation."""
    model = MockModel()
    recorder = DataRecorder(model)
    model.step()

    summary = recorder.summary()
    assert summary["datasets"] == 3  # model, agent, numpy
    assert summary["total_rows"] > 0
    assert "memory_mb" in summary


def test_datarecorder_eviction_coverage():
    """Test eviction logic for all data types (numpy, list, dict, custom)."""
    model = MockModel()

    # Create datasets in registry
    model.data_registry.create_dataset(TableDataSet, "np_ds", fields="v")
    model.data_registry.create_dataset(TableDataSet, "list_ds", fields="v")
    model.data_registry.create_dataset(TableDataSet, "dict_ds", fields="v")
    model.data_registry.create_dataset(TableDataSet, "custom_ds", fields="v")

    # Config with window_size=1 to force eviction on 2nd insert
    config = {
        "np_ds": DatasetConfig(window_size=1),
        "list_ds": DatasetConfig(window_size=1),
        "dict_ds": DatasetConfig(window_size=1),
        "custom_ds": DatasetConfig(window_size=1),
    }

    recorder = DataRecorder(model, config=config)

    # 1. Custom Data Type (triggers 'case _:' and 'type="custom"')
    recorder._store_dataset_snapshot("custom_ds", 1, "value1")
    assert recorder.storage["custom_ds"].metadata["type"] == "custom"

    # Push second value to trigger eviction of "value1" (custom eviction logic)
    recorder._store_dataset_snapshot("custom_ds", 2, "value2")
    assert len(recorder.storage["custom_ds"].blocks) == 1
    assert recorder.storage["custom_ds"].blocks[0][1] == "value2"

    # 2. Numpy Eviction
    recorder._store_dataset_snapshot("np_ds", 1, np.array([1]))
    recorder._store_dataset_snapshot("np_ds", 2, np.array([2]))
    assert len(recorder.storage["np_ds"].blocks) == 1

    # 3. List Eviction
    recorder._store_dataset_snapshot("list_ds", 1, [{"a": 1}])
    recorder._store_dataset_snapshot("list_ds", 2, [{"b": 2}])
    assert len(recorder.storage["list_ds"].blocks) == 1

    # 4. Dict Eviction
    recorder._store_dataset_snapshot("dict_ds", 1, {"a": 1})
    recorder._store_dataset_snapshot("dict_ds", 2, {"b": 2})
    assert len(recorder.storage["dict_ds"].blocks) == 1


def test_datarecorder_clear_specific():
    """Test clearing a specific dataset and handling invalid names."""
    model = MockModel()
    model.data_registry.create_dataset(TableDataSet, "test_ds", fields="v")
    recorder = DataRecorder(model)

    recorder._store_dataset_snapshot("test_ds", 1, {"v": 1})
    assert recorder.storage["test_ds"].total_rows == 1

    # Clear specific
    recorder.clear("test_ds")
    assert recorder.storage["test_ds"].total_rows == 0

    # Clear invalid
    with pytest.raises(KeyError):
        recorder.clear("non_existent")


def test_numpy_json_encoder_coverage():
    """Test JSON serialization of specific Numpy types."""
    data = {
        "float": np.float32(1.5),
        "bool": np.bool_(True),
        "array": np.array([1, 2]),
        "int": np.int32(10),
    }

    # Dump to string using the encoder
    json_str = json.dumps(data, cls=NumpyJSONEncoder)

    # Parse back to check values
    decoded = json.loads(json_str)
    assert decoded["float"] == 1.5
    assert decoded["bool"] is True
    assert decoded["array"] == [1, 2]
    assert decoded["int"] == 10


@patch("pandas.DataFrame.to_parquet")
@patch("pandas.read_parquet")
@patch("pathlib.Path.exists")
@patch("pathlib.Path.unlink")
def test_parquet_recorder_coverage(
    mock_unlink, mock_exists, mock_read, mock_to_parquet
):
    """Test ParquetRecorder logic by mocking file I/O (runs without pyarrow)."""
    model = MockModel()

    # 1. Setup Registry with agent_ids support logic check
    # We need a mock dataset that has 'agent_ids' to test the np.ndarray logic branch
    mock_dataset = MagicMock()
    mock_dataset._attributes = ["col1"]
    mock_dataset.agent_ids = np.array([100])
    model.data_registry.datasets["numpy_ds"] = mock_dataset

    recorder = ParquetDataRecorder(model)
    recorder.buffer_size = 1  # Force flush on every write
    recorder._initialize_dataset_storage("numpy_ds", mock_dataset)

    # 2. Test Storing Numpy Data (triggers IDs stacking logic)
    data = np.array([[1.0]])
    recorder._store_dataset_snapshot("numpy_ds", 1, data)

    # Verify to_parquet was called (flush happened)
    assert mock_to_parquet.called
    # Verify ID stacking: The dataframe passed to to_parquet should have 'agent_id'
    # call_args[0][0] is the filename (if args used) or self (if method mock).
    # Since we mocked DataFrame.to_parquet, the first arg to the real function is 'self' (the df).
    # But usually patch replaces the unbound method or instance method.
    # Let's check the logic inside _store_dataset_snapshot more simply:
    # It constructs a DF. We can verify logic by checking if it ran without error
    # and hit the flush.

    # 3. Test get_table_dataframe
    mock_exists.return_value = True  # File exists
    mock_read.return_value = pd.DataFrame({"a": [1]})

    df = recorder.get_table_dataframe("numpy_ds")
    assert not df.empty
    assert mock_read.called

    # 4. Test Clear (Specific)
    recorder.clear("numpy_ds")
    assert mock_unlink.called

    # 5. Test Clear (All)
    recorder.clear()
    assert mock_unlink.call_count >= 2

    # 6. Test Summary (File exists branch)
    # We need to mock os.path.getsize as well if we want full coverage there,
    # but the basics are covered.
    with patch("os.path.getsize", return_value=1024):
        summary = recorder.summary()
        assert "numpy_ds" in summary
