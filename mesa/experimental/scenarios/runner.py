"""Classes for running parameter sweeps over scenarios."""

from __future__ import annotations

import traceback
from collections.abc import Iterable
from concurrent.futures import BrokenExecutor, as_completed
from typing import TYPE_CHECKING, Any

import pandas as pd

from mesa.experimental.scenarios.exceptions import (
    FailureInfo,
    FailureOrigin,
    ModelInstantiationException,
    ModelRunException,
    OutcomeExtractionException,
    RunStageException,
)
from mesa.experimental.scenarios.store import InMemoryStore, RunId

if TYPE_CHECKING:
    from concurrent.futures import Executor

    from mesa.experimental.scenarios import Scenario
    from mesa.experimental.scenarios.store import Reference, Store, Writer
    from mesa.model import Model


class RunConfiguration:
    """Defines how a single Scenario is executed and what is extracted from it.

    Can be used as is for simple use cases or subclassed by overriding one or more of the following
    methods

    - ``instantiate_model`` — construct a Model from a Scenario (default:
      ``model_class(*model_args, scenario=scenario, **model_kwargs)``).
    - ``run_model`` — advance the model. Default delegates to ``model.run_until`` based on the ``until`` attribute.
      Override for alternative run control
    - ``extract_output`` — return a dict with outcome names as key and dataframes as values

    Stopping is the model's responsibility. ``RunConfiguration`` only chooses
    which run primitive to call.

    """

    def __init__(
        self,
        model_class: type[Model],
        until: float | int,
        model_args: None | list[Any] = None,
        model_kwargs: None | dict[str, Any] = None,
        outcomes: None | str | list[str] = None,
        data_recorder_attr_name="data_recorder",
    ):
        """Initialize a RunConfiguration object.

        Args:
            model_class: the model class to instantiate
            until: until which time to run the model
            model_args: any additional model arguments
            model_kwargs: any additional model keyword arguments
            outcomes: the outcomes to extract. If None, extract all outcomes.
            data_recorder_attr_name : the name of the data recorder attribute to use on the model
        """
        # we need to avoid circular imports
        from mesa.model import Model  # noqa: PLC0415

        if not (isinstance(model_class, type) and issubclass(model_class, Model)):
            raise TypeError("model_class must be a subclass of Model")
        if not isinstance(until, (int, float)):
            raise TypeError("until must be an int or float")
        if until <= 0:
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

    def instantiate_model(self, scenario: Scenario) -> Model:
        """Instantiate the model."""
        return self.model_class(
            *self.model_args, scenario=scenario, **self.model_kwargs
        )

    def run_model(self, model: Model) -> None:
        """Run the model."""
        model.run_until(self.until)

    def extract_output(self, model: Model) -> dict[str, pd.DataFrame]:
        """Extract output from model."""
        recorder = getattr(model, self.data_recorder_attr_name)

        if self.outcomes is None:
            return recorder.get_all_dataframes()
        else:
            return {k: recorder.get_table_dataframe(k) for k in self.outcomes}

    def __call__(self, scenario: Scenario) -> dict[str, pd.DataFrame]:
        """Run the scenario and extract output."""
        try:
            model = self.instantiate_model(scenario)
        except Exception as e:
            raise ModelInstantiationException(
                self.model_class, self.model_args, self.model_kwargs, scenario
            ) from e

        try:
            self.run_model(model)
        except Exception as e:
            raise ModelRunException(
                RunId(scenario.scenario_id, scenario.replication_id)
            ) from e

        try:
            output = self.extract_output(model)
        except Exception as e:
            raise OutcomeExtractionException(
                RunId(scenario.scenario_id, scenario.replication_id), self.outcomes
            ) from e
        return output


def _safe_call(
    config: RunConfiguration,
    scenario: Scenario,
    writer: Writer,
) -> tuple[Reference, None] | tuple[None, FailureInfo]:
    """Run one scenario and persist its outcome. Runs in the worker.

    Args:
        config: a RunConfiguration instance
        scenario: a Scenario instance
        writer: a Writer instance

    Returns (reference, None) on success or (None, failure_info) on a failure
    raised inside the run or the writer. Catching here means a model
    error becomes data (a FailureInfo instance) rather than an exception
    crossing the process boundary, so one failed run never aborts a parameter sweep.
    """
    run_id = RunId(scenario.scenario_id, scenario.replication_id)
    try:
        outcome = config(scenario)
    except RunStageException as e:
        cause = e.__cause__ or e
        return None, FailureInfo(
            origin=e.origin,
            exception_type=type(cause).__name__,
            message=str(cause),
            traceback="".join(traceback.format_exception(e)),
        )
    try:
        ref = writer.to_reference(run_id, outcome)
    except Exception as e:
        return None, FailureInfo(
            origin=FailureOrigin.WRITING,
            exception_type=type(e).__name__,
            message=str(e),
            traceback="".join(traceback.format_exception(e)),
        )
    return ref, None


def run_scenarios(
    scenarios: Iterable[Scenario],
    config: RunConfiguration,
    *,
    executor: Executor | None = None,
    store: Store | None = None,
    progress: bool = True,
) -> Store:
    """Run the scenarios and return a Store object.

    Args:
        scenarios: an iterable of scenarios to run
            Scenarios to execute. For replications, construct these via
            ``MyScenario.from_dataframe(df, replications=n)`` — replication is
            handled at scenario construction, not here.
        config: a RunConfiguration instance
            Per-scenario execution unit. Must be picklable when using a
            distributed executor (e.g., ProcessPoolExecutor).
        executor: an executor to run the scenarios
            Execution backend. If None, scenarios run sequentially in the
            calling thread (useful for debugging and small experiments).
            Otherwise, pass a user-constructed executor; its lifetime is the
            caller's responsibility (use a ``with`` block).
        store: the Storage backend to use
        progress: whether to display the progress
            Display a progress bar via ``tqdm`` if installed.

    Returns: a Store instance

    """
    if store is None:
        store = InMemoryStore()

    scenarios = list(scenarios)
    writer = store.writer()
    store.write_scenarios(scenarios)

    def _bar(iterable):
        if not progress:
            return iterable
        try:
            from tqdm.auto import tqdm  # noqa: PLC0415

            return tqdm(iterable, total=len(scenarios), desc="Running scenarios")
        except ImportError:
            return iterable

    def _record(scenario: Scenario, ref: Reference | None, failure: FailureInfo | None):
        """Handler for recording the return _safe_call."""
        if failure is None:
            store.mark_succeeded(ref)
        else:
            run_id = RunId(scenario.scenario_id, scenario.replication_id)
            store.mark_failed(run_id, failure)

    if executor is None:
        # Sequential: run in the loop so the bar advances per scenario.
        for scenario in _bar(scenarios):
            ref, failure_info = _safe_call(config, scenario, writer)
            _record(scenario, ref, failure_info)
    else:
        futures = {
            executor.submit(_safe_call, config, scenario, writer): scenario
            for scenario in scenarios
        }
        try:
            for future in _bar(as_completed(futures)):
                scenario = futures[future]
                try:
                    ref, failure_info = future.result()
                except BrokenExecutor:
                    # raise it again so it is handled in the dedicated outer except clause
                    raise
                except Exception as e:
                    # pickling failure or CancelledError on the return trip; record and continue
                    # RUNNING might not be the right label here....
                    ref, failure_info = (
                        None,
                        FailureInfo(
                            origin=FailureOrigin.WRITING,
                            exception_type=type(e).__name__,
                            message=str(e),
                            traceback="".join(traceback.format_exception(e)),
                        ),
                    )
                _record(scenario, ref, failure_info)
        except BrokenExecutor as e:
            cause = e.__cause__ or e
            for entry in list(store.pending()):
                store.mark_aborted(
                    entry,
                    FailureInfo(
                        origin=FailureOrigin.ABORTED,
                        exception_type=type(cause).__name__,
                        message=str(e),
                        traceback="".join(traceback.format_exception(e)),
                    ),
                )
            return store

    return store
