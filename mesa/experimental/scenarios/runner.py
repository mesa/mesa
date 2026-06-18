"""Classes for running parameter sweeps over scenarios."""
from typing import Any

import pandas as pd

from mesa import Model
from mesa.experimental.scenarios import Scenario


class RunConfiguration:
    """Defines how a single Scenario is executed and what is extracted from it.

    Can be used as is for simple use cases or subclassed by overriding one or more of the following
    methods

    - ``instantiate_model`` — construct a Model from a Scenario (default:
      ``model_class(*model_args, scenario=scenario, *model_kwargs)``).
    - ``run_model`` — advance the model. Default delegates to ``model.run_until`` based on the ``until`` attribute.
      Override for alternative run control
    - ``extract_output`` — return a dict with outcome names as key and dataframes as values

    Stopping is the model's responsibility. ``RunConfiguration`` only chooses
    which run primitive to call.

    """
    def __init__(self, model_class: type[Model], until:float | int, model_args:None| list[Any]=None,
                 model_kwargs:None|dict[str, Any]=None, outcomes: None|str|list[str]=None):
        """Initialize a RunConfiguration object.

        Args:
            model_class: the model class to instantiate
            until: until which time to run the model
            model_args: any additional model arguments
            model_kwargs: any additional model keyword arguments
            outcomes: the outcomes to extract. If None, extract all outcomes.
        """
        super().__init__()

        if not (isinstance(model_class, type) and issubclass(model_class, Model)):
            raise TypeError("model_class must be a subclass of Model")
        if not isinstance(until, (int, float)):
            raise TypeError("until must be an int or float")
        if until<=0:
            raise ValueError("until must be positive")

        self.model_class = model_class
        self.model_args = [] if model_args is None else model_args
        self.model_kwargs = {} if model_kwargs is None else model_kwargs
        self.until = until

        if isinstance(outcomes, str):
            outcomes = [outcomes]
        self.outcomes = outcomes

    def instantiate_model(self, scenario) -> Model:
        """Instantiate the model."""
        return self.model_class(*self.model_args, scenario=scenario, **self.model_kwargs)

    def run_model(self, model: Model):
        """Run the model."""
        model.run_until(self.until)

    def extract_output(self, model: Model) -> dict[str, pd.DataFrame]:
        """Extract output from model."""
        # fixme:: this code assumed that the recorder is assigned to model.data_recorder
        #   this is probably a a convention that needs to be pinned down explicitly on the model class
        if self.outcomes is None:
            return model.data_recorder.get_all_dataframes()
        else:
            return {k:model.data_recorder.get_table_dataframe(k) for k in self.outcomes}

    def __call__(self, scenario: Scenario) -> dict[str, pd.DataFrame]:
        """Run the scenario and extract output."""
        model = self.instantiate_model(scenario)
        self.run_model(model)
        output = self.extract_output(model)
        return output

