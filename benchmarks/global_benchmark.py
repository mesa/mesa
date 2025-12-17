"""runner for global performance benchmarks."""

import gc
import os
import pickle
import sys
import time
import timeit

# making sure we use this version of mesa and not one
# also installed in site_packages or so.
sys.path.insert(0, os.path.abspath(".."))

from configurations import configurations

from mesa.experimental.devs.simulator import ABMSimulator


# Generic function to initialize and run a model
def run_model(model_class, seed, parameters):
    """Run model for given seed and parameter values.

    Args:
        model_class: a model class
        seed: the seed
        parameters: parameters for the run

    Returns:
        startup time and run time
    """
    uses_simulator = ["WolfSheep"]
    start_init = timeit.default_timer()
    if model_class.__name__ in uses_simulator:
        simulator = ABMSimulator()
        model = model_class(simulator=simulator, rng=seed, **parameters)
    else:
        model = model_class(rng=seed, **parameters)

    end_init_start_run = timeit.default_timer()

    if model_class.__name__ in uses_simulator:
        simulator.run_for(config["steps"])
    else:
        for _ in range(config["steps"]):
            model.step()

    end_run = timeit.default_timer()

    return (end_init_start_run - start_init), (end_run - end_init_start_run)


# Function to run experiments and save the fastest replication for each seed
def run_experiments(model_class, config):
    """Run performance benchmarks.

    Args:
        model_class: the model class to use for the benchmark
        config: the benchmark configuration

    """
    gc.enable()
    sys.path.insert(0, os.path.abspath("."))

    init_times = []
    run_times = []
    for seed in range(1, config["seeds"] + 1):
        fastest_init = float("inf")
        fastest_run = float("inf")
        for _replication in range(1, config["replications"] + 1):
            init_time, run_time = run_model(model_class, seed, config["parameters"])
            if init_time < fastest_init:
                fastest_init = init_time
            if run_time < fastest_run:
                fastest_run = run_time
        init_times.append(fastest_init)
        run_times.append(fastest_run)

    return init_times, run_times


print(f"{time.strftime('%H:%M:%S', time.localtime())} starting benchmarks.")
results_dict = {}
for model, model_config in configurations.items():
    for size, config in model_config.items():
        results = run_experiments(model, config)

        mean_init = sum(results[0]) / len(results[0])
        mean_run = sum(results[1]) / len(results[1])

        print(
            f"{time.strftime('%H:%M:%S', time.localtime())} {model.__name__:<14} ({size}) timings: Init {mean_init:.5f} s; Run {mean_run:.4f} s"
        )

        results_dict[model, size] = results

# Change this name to anything you like
save_name = "timings"

i = 1
while os.path.exists(f"{save_name}_{i}.pickle"):
    i += 1

with open(f"{save_name}_{i}.pickle", "wb") as handle:
    pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Done benchmarking. Saved results to {save_name}_{i}.pickle.")
