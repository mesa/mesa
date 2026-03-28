"""Execution class for running scenarios."""

from __future__ import annotations

from typing import Any

from mesa.experimental.scenarios import Scenario
from mesa.model import Model


class RunSpec:
    """Execution specification for a single Scenario.

    Defines how a model is constructed, executed, and how results are extracted.
    Designed to be subclassed for custom behavior.
    """

    def __init__(self, model_cls: type[Model], steps: int = 100) -> None:
        """Initialising Runspec."""
        self.model_cls = model_cls
        self.steps = steps

    def build_model(self, scenario: Scenario) -> Model:
        """Construct a model instance from a Scenario."""
        return self.model_cls(scenario=scenario)

    def execute(self, model: Model) -> None:
        """Run the model for a fixed number of steps."""
        model.run_for(self.steps)

    def extract(self, model: Model) -> Any:
        """Extract results from the model."""
        return model.data_registry

    def format_output(self, scenario: Scenario, result: Any) -> tuple[Any, Any, Any]:
        """Format the output of a run."""
        return scenario.scenario_id, scenario.replication_id, result

    def __call__(self, scenario: Scenario) -> tuple[Any, Any, Any]:
        """Execute the full run pipeline for a Scenario."""
        model = self.build_model(scenario)
        self.execute(model)
        result = self.extract(model)
        return self.format_output(scenario, result)
