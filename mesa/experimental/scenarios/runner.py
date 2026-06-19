"""Classes for running parameter sweeps over scenarios."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from mesa.experimental.scenarios import Scenario

if TYPE_CHECKING:
    from mesa.model import Model


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
                 model_kwargs:None|dict[str, Any]=None, outcomes: None|str|list[str]=None, data_recorder_attr_name="data_recorder"):
        """Initialize a RunConfiguration object.

        Args:
            model_class: the model class to instantiate
            until: until which time to run the model
            model_args: any additional model arguments
            model_kwargs: any additional model keyword arguments
            outcomes: the outcomes to extract. If None, extract all outcomes.
            data_recorder_attr_name : the name of the data recorder attribute to use on the model
        """
        super().__init__()

        # we need to avoid circular imports
        from mesa.model import Model  # noqa: PLC0415
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

        # fixme:: this code leaves it to the user to set the attribute to which the recorder is assigned
        #   this is probably a a convention that needs to be pinned down explicitly on the model class
        self.data_recorder_attr_name = data_recorder_attr_name

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
        recorder = getattr(model, self.data_recorder_attr_name)

        if self.outcomes is None:
            return recorder.get_all_dataframes()
        else:
            return {k:recorder.get_table_dataframe(k) for k in self.outcomes}

    def __call__(self, scenario: Scenario) -> dict[str, pd.DataFrame]:
        """Run the scenario and extract output."""
        model = self.instantiate_model(scenario)
        self.run_model(model)
        output = self.extract_output(model)
        return output

