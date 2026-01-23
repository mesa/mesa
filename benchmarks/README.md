# Mesa Performance Benchmarks

The `/benchmarks` directory contains tools for benchmarking the Mesa library performance on the included example models. This allows to track and compare model initialisation time and runtime between different Mesa versions.

MESA uses several example base models for benchmarking performance (BoltzmannWealth, Schelling, BoidFlockers, and WolfSheep) by calculating the initialization time and run time for each of these models. These example models can be found in the `/mesa/examples` directory.

## Available Files

- `configurations.py`: Contains model configurations for benchmarking
- `global_benchmark.py`: Main script for running benchmarks
- `compare_timings.py`: Tool to compare results between benchmark runs

## How to Use


### 1. Benchmark Configuration

The `configurations.py` file defines which models to benchmark and their parameters. Each model has:

- `small` and `large` configurations for testing different scales
- Parameters for:
  - `seeds`: Number of different random seeds to use
  - `replications`: Number of times to repeat the benchmark for each seed
  - `steps`: Number of model steps to run
  - `parameters`: Model-specific parameters

### 2. Running Benchmarks

To run the benchmarks:

```bash
pip install tabulate # if you don't have it yet
python global_benchmark.py
```

This will:
- Run all models defined in `configurations.py` with their respective configurations
- Measure initialization time and run time for each model
- Save results to a pickle file named `timings_X.pickle` (where X is an incremental number)
- Display a summary of results in the terminal


>**Noteworthy :** the pickle file created by the benchmark is not under git control. So you can run the benchmark on the master branch first, switch to your development branch, and run the benchmarks again.

#### What's being measured:

- **Initialization time**: How long it takes to create model instances
- **Run time**: How long it takes to run the model for the specified number of steps

### 3. Comparing Results

After running benchmarks at different times (e.g., before and after code changes), you can compare the results:

1. Rename your benchmark files to distinguish them (e.g., `timings_before.pickle` and `timings_after.pickle`)
2. Edit `compare_timings.py` to use your file names:
   ```python
   filename1 = "timings_1" # Or the name of your 1st timings file
   filename2 = "timings_2" # Or the name of your 2nd timings file
   ```
3. Run the comparison:
   ```bash
   python compare_timings.py
   ```

The output will show:
- Percentage changes in initialization and run times
- 95% confidence intervals for the changes
- Emojis indicate performance changes:
  - 🟢 Faster performance (>3% improvement)
  - 🔴 Slower performance (>3% regression)
  - 🔵 Insignificant change (within ±3%)

> Some care is required in the interpretation since it only shows percentage changes and not the absolute changes. The init times in general are tiny so slower performance here is not necessarily as much of an issue.


## Example Workflow

1. Run benchmarks on your current Mesa version:
   ```bash
   python global_benchmark.py
   # Results saved as timings_1.pickle
   ```

2. Make code changes (add a feature in mesa, optimize a method, etc.) **OR** switch to development branch - you can do that without duplicating your pickle file in the new branch as it is not under git control


3. Run benchmarks again:
   ```bash
   python global_benchmark.py
   # Results saved as timings_2.pickle
   ```

4. Compare results:
   ```bash
   python compare_timings.py
   # Shows performance change table
   ```

5. Use the results to validate your optimizations or identify regressions
