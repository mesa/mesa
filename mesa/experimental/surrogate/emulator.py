"""Module for emulating Mesa models using machine learning regressors."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from mesa.batchrunner import batch_run

from .utils import batch_to_xy


class Emulator:
    """A surrogate model that approximates Mesa simulation outcomes."""

    def __init__(self, model_cls, model_params, regressor=None):
        """Initializes the surrogate emulator with a model class and regressor."""
        self.model_cls = model_cls
        self.model_params = model_params
        self.regressor = regressor or RandomForestRegressor(n_estimators=100)
        self.is_trained = False

    def train(self, sampled_configs, max_steps=100, output_metric="Target"):
        """Runs simulations and trains the underlying regressor on results."""
        results = []
        for config in sampled_configs:
            run_result = batch_run(
                self.model_cls,
                parameters={k: [v] for k, v in config.items()},
                max_steps=max_steps,
                rng=[None],
            )
            results.extend(run_result)

        x_data, y_data = batch_to_xy(results, self.model_params, output_metric)
        self.regressor.fit(x_data, y_data)
        self.is_trained = True

    def predict(self, params_dict):
        """Predicts the model outcome for a given set of parameters."""
        if not self.is_trained:
            raise ValueError("Emulator must be trained before prediction.")
        x_input = np.array([[params_dict[p] for p in self.model_params]])
        return self.regressor.predict(x_input)[0]
