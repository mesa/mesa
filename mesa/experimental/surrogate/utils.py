"""Data transformation utilities for mapping simulation results to ML inputs."""

import pandas as pd


def batch_to_xy(batch_data, input_params, output_metric):
    """Converts Mesa batch_run results into ML-ready NumPy arrays."""
    df = pd.DataFrame(batch_data)

    x_data = df[input_params].values
    y_data = df[output_metric].values

    return x_data, y_data
