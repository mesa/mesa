"""Unit tests for the surrogate modeling experimental module."""

import numpy as np

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.experimental.surrogate import Emulator, batch_to_xy, sample_parameters


class MockModel(Model):
    """Simple model for testing surrogate training and prediction."""

    def __init__(self, x=1, y=2, **kwargs):
        """Initializes the mock model with coordinates."""
        super().__init__(**kwargs)
        self.x = x
        self.y = y
        self.result = x * y
        self.datacollector = DataCollector(model_reporters={"Outcome": "result"})

    def step(self):
        """Collects data and terminates the run."""
        self.datacollector.collect(self)
        self.running = False


def test_batch_to_xy():
    """Verifies that batch results are correctly converted to X and Y arrays."""
    data = [{"x": 1, "y": 2, "Outcome": 2}, {"x": 2, "y": 3, "Outcome": 6}]
    x_data, y_data = batch_to_xy(data, ["x", "y"], "Outcome")
    assert x_data.shape == (2, 2)
    assert y_data.shape == (2,)
    assert y_data[1] == 6


def test_sample_parameters():
    """Tests if Latin Hypercube Sampling generates valid parameter sets."""
    param_space = {"x": (0, 10), "y": (0, 5)}
    samples = sample_parameters(param_space, n_samples=5, seed=42)
    assert len(samples) == 5
    for s in samples:
        assert 0 <= s["x"] <= 10
        assert 0 <= s["y"] <= 5


def test_emulator_integration():
    """Ensures the Emulator can train and predict using a MockModel."""
    param_space = {"x": (1, 5), "y": (1, 5)}
    model_params = ["x", "y"]

    emu = Emulator(MockModel, model_params)
    samples = sample_parameters(param_space, n_samples=10, seed=42)

    emu.train(samples, output_metric="Outcome")
    assert emu.is_trained

    prediction = emu.predict({"x": 3, "y": 3})
    assert isinstance(prediction, (float, np.float64))
