"""Tests for DataRecorders."""

import tempfile
from pathlib import Path

import pytest

from mesa.agent import Agent
from mesa.experimental.data_collection import (
    DataRecorder,
    DatasetConfig,
    JSONDataRecorder,
    ParquetDataRecorder,
    SQLDataRecorder,
)
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
