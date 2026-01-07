"""batchrunner for running a factorial experiment design over a model.

To take advantage of parallel execution of experiments, `batch_run` uses
multiprocessing if ``number_processes`` is larger than 1. It is strongly advised
to only run in parallel using a normal python file (so don't try to do it in a
jupyter notebook). This is because Jupyter notebooks have a different execution
model that can cause issues with Python's multiprocessing module, especially on
Windows. The main problems include the lack of a traditional __main__ entry
point, serialization issues, and potential deadlocks.

Moreover, best practice when using multiprocessing is to
put the code inside an ``if __name__ == '__main__':`` code black as shown below::

    from mesa.batchrunner import batch_run

    params = {"width": 10, "height": 10, "N": range(10, 500, 10)}

    if __name__ == '__main__':
        results = batch_run(
            MoneyModel,
            parameters=params,
            iterations=5,
            max_steps=100,
            number_processes=None,
            data_collection_period=1,
            display_progress=True,
        )

"""

import bisect
import inspect
import itertools
import multiprocessing
import warnings
from collections.abc import Iterable, Mapping, Sequence
from functools import partial
from multiprocessing import Pool
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from mesa.model import Model

SeedLike = int | np.integer | Sequence[int] | np.random.SeedSequence


def batch_run(
    model_cls: type[Model],
    parameters: Mapping[str, Any | Iterable[Any]],
    # We still retain the Optional[int] because users may set it to None (i.e. use all CPUs)
    number_processes: int | None = 1,
    iterations: int | None = None,
    data_collection_period: int = -1,
    max_steps: int = 1000,
    display_progress: bool = True,
    rng: SeedLike | Iterable[SeedLike] | None = None,
) -> list[dict[str, Any]]:
    """Batch run a mesa model with a set of parameter values.

    Args:
        model_cls (Type[Model]): The model class to batch-run
        parameters (Mapping[str, Union[Any, Iterable[Any]]]): Dictionary with model parameters over which to run the model. You can either pass single values or iterables.
        number_processes (int, optional): Number of processes used, by default 1. Set this to None if you want to use all CPUs.
        iterations (int, optional): Number of iterations for each parameter combination, by default 1
        data_collection_period (int, optional): Number of steps after which data gets collected, by default -1 (end of episode)
        max_steps (int, optional): Maximum number of model steps after which the model halts, by default 1000
        display_progress (bool, optional): Display batch run process, by default True
        rng : a valid value or iterable of values for seeding the random number generator in the model

    Returns:
        List[Dict[str, Any]]

    Notes:
        batch_run assumes the model has a `datacollector` attribute that has a DataCollector object initialized.

    """
    multiprocessing.set_start_method("spawn", force=True)

    if iterations is not None and rng is not None:
        raise ValueError(
            "you cannot use both iterations and rng at the same time. Please only use rng."
        )
    if iterations is not None:
        warnings.warn(
            "The `iterations` keyword argument is deprecated, please use `rng` instead."
            "See https://mesa.readthedocs.io/latest/migration_guide.html#batch-run",
            DeprecationWarning,
            stacklevel=2,
        )
        rng = [None] * iterations
    if not isinstance(rng, Iterable):
        rng = [rng]

    # establish to use seed or rng as name for parameter
    model_parameters = inspect.signature(model_cls).parameters
    rng_kwarg_name = "rng"
    if "seed" in model_parameters:
        rng_kwarg_name = "seed"

    runs_list = []
    run_id = 0
    for i, rng_i in enumerate(rng):
        for kwargs in _make_model_kwargs(parameters):
            kwargs[rng_kwarg_name] = rng_i
            runs_list.append((run_id, i, kwargs))
            run_id += 1

    process_func = partial(
        _model_run_func,
        model_cls,
        max_steps=max_steps,
        data_collection_period=data_collection_period,
    )

    results: list[dict[str, Any]] = []

    with tqdm(total=len(runs_list), disable=not display_progress) as pbar:
        if number_processes == 1:
            for run in runs_list:
                data = process_func(run)
                results.extend(data)
                pbar.update()
        else:
            with Pool(number_processes) as p:
                for data in p.imap_unordered(process_func, runs_list):
                    results.extend(data)
                    pbar.update()

    return results


def _make_model_kwargs(
    parameters: Mapping[str, Any | Iterable[Any]],
) -> list[dict[str, Any]]:
    """Create model kwargs from parameters dictionary.

    Parameters
    ----------
    parameters : Mapping[str, Union[Any, Iterable[Any]]]
        Single or multiple values for each model parameter name.

        Allowed values for each parameter:
        - A single value (e.g., `32`, `"relu"`).
        - A non-empty iterable (e.g., `[0.01, 0.1]`, `["relu", "sigmoid"]`).

        Not allowed:
        - Empty lists or empty iterables (e.g., `[]`, `()`, etc.). These should be removed manually.

    Returns:
    -------
    List[Dict[str, Any]]
        A list of all kwargs combinations.
    """
    parameter_list = []
    for param, values in parameters.items():
        if isinstance(values, str):
            # The values is a single string, so we shouldn't iterate over it.
            all_values = [(param, values)]
        elif isinstance(values, list | tuple | set) and len(values) == 0:
            # If it's an empty iterable, raise an error
            raise ValueError(
                f"Parameter '{param}' contains an empty iterable, which is not allowed."
            )

        else:
            try:
                all_values = [(param, value) for value in values]
            except TypeError:
                all_values = [(param, values)]
        parameter_list.append(all_values)
    all_kwargs = itertools.product(*parameter_list)
    kwargs_list = [dict(kwargs) for kwargs in all_kwargs]
    return kwargs_list


def _model_run_func(
    model_cls: type[Model],
    run: tuple[int, int, dict[str, Any]],
    max_steps: int,
    data_collection_period: int,
) -> list[dict[str, Any]]:
    """Run a single model run and collect model and agent data.

    Parameters
    ----------
    model_cls : Type[Model]
        The model class to batch-run
    run: Tuple[int, int, Dict[str, Any]]
        The run id, iteration number, and kwargs for this run
    max_steps : int
        Maximum number of model steps after which the model halts, by default 1000
    data_collection_period : int
        Number of steps after which data gets collected

    Returns:
    -------
    List[Dict[str, Any]]
        Return model_data, agent_data from the reporters
    """
    run_id, iteration, kwargs = run

    model = model_cls(**kwargs)
    while model.running and model.steps <= max_steps:
        model.step()

    data = []

    steps = list(range(0, model.steps, data_collection_period))
    if not steps or steps[-1] != model.steps - 1:
        steps.append(model.steps - 1)

    for step in steps:
        model_data, all_agents_data = _collect_data(model, step)

        # If there are agent_reporters, then create an entry for each agent
        if all_agents_data:
            stepdata = [
                {
                    "RunId": run_id,
                    "iteration": iteration,
                    "Step": step,
                    **kwargs,
                    **model_data,
                    **agent_data,
                }
                for agent_data in all_agents_data
            ]
        # If there is only model data, then create a single entry for the step
        else:
            stepdata = [
                {
                    "RunId": run_id,
                    "iteration": iteration,
                    "Step": step,
                    **kwargs,
                    **model_data,
                }
            ]
        data.extend(stepdata)

    return data


def _collect_data(
    model: Model,
    step: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Collect model and agent data from a model using mesas datacollector."""
    if not hasattr(model, "datacollector"):
        raise AttributeError(
            "The model does not have a datacollector attribute. Please add a DataCollector to your model."
        )
    dc = model.datacollector

    # Check if modern DataCollector with _collection_steps exists (handles time dilation)
    if hasattr(dc, "_collection_steps"):
        idx = bisect.bisect_right(dc._collection_steps, step) - 1
        if (
            idx >= 0
            and idx < len(dc._collection_steps)
            and dc._collection_steps[idx] == step
        ):
            # Exact match found - use the index directly
            model_data = {param: values[idx] for param, values in dc.model_vars.items()}
        else:
            # Step not found in _collection_steps
            # Use sparse collection logic: find the nearest collected step
            if idx >= 0 and idx < len(dc._collection_steps):
                # Use the most recent collected data before this step
                model_data = {
                    param: values[idx] for param, values in dc.model_vars.items()
                }
            else:
                # No data collected yet, use first available
                try:
                    model_data = {
                        param: values[0] for param, values in dc.model_vars.items()
                    }
                except IndexError:
                    model_data = {}
    else:
        # Legacy DataCollector without _collection_steps
        # Use sparse collection logic for models that collect data irregularly
        available_steps = sorted(dc._agent_records.keys())
        if step not in available_steps:
            step = max((s for s in available_steps if s <= step), default=0)

        try:
            collection_index = available_steps.index(step)
        except ValueError:
            collection_index = 0

        model_data = {
            param: values[collection_index] for param, values in dc.model_vars.items()
        }

    all_agents_data = []
    raw_agent_data = dc._agent_records.get(step, [])
    for data in raw_agent_data:
        agent_dict = {"AgentID": data[1]}
        agent_dict.update(zip(dc.agent_reporters, data[2:]))
        all_agents_data.append(agent_dict)
    return model_data, all_agents_data
