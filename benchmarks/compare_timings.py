"""compare timings across 2 benchmarks."""

import pickle
import argparse
import sys

import numpy as np
import pandas as pd

filename1 = "timings_1"
filename2 = "timings_2"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare timings across 2 benchmarks and display % changes with 95% CI."
    )
    parser.add_argument("--file1", default="timings_1", help="Base name of the first pickle file (without .pickle)")
    parser.add_argument("--file2", default="timings_2", help="Base name of the second pickle file (without .pickle)")
    parser.add_argument("--n", type=int, default=1000, help="Number of bootstrap samples (default: 1000)")
    parser.add_argument("--threshold", type=float, default=3.0, help="Percentage threshold for flagging significant change (default: 3.0)")
    return parser.parse_args()

def load_pickle(filename: str) -> dict:
    path = f"{filename}.pickle"
    try:
        with open(path, "rb") as handle:
            return pickle.load(handle)  # noqa: S301
    except FileNotFoundError:
        print(f"Error: File '{path}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading '{path}': {e}", file=sys.stderr)
        sys.exit(1)

with open(f"{filename1}.pickle", "rb") as handle:
    timings_1 = pickle.load(handle)  # noqa: S301

with open(f"{filename2}.pickle", "rb") as handle:
    timings_2 = pickle.load(handle)  # noqa: S301


def bootstrap_percentage_change_confidence_interval(data1, data2, n=1000):
    """Calculate the percentage change and perform bootstrap to estimate the confidence interval.

    Args:
        data1: benchmark dataset 1
        data2: benchmark dataset 2
        n: bootstrap sample size

    Returns:
        float, mean, and lower and upper bound of confidence interval.
    """
    change_samples = []
    for _ in range(n):
        sampled_indices = np.random.choice(
            range(len(data1)), size=len(data1), replace=True
        )
        sampled_data1 = np.array(data1)[sampled_indices]
        sampled_data2 = np.array(data2)[sampled_indices]
        change = 100 * (sampled_data2 - sampled_data1) / sampled_data1
        change_samples.append(np.mean(change))
    lower, upper = np.percentile(change_samples, [2.5, 97.5])
    return np.mean(change_samples), lower, upper


# DataFrame to store the results
results_df = pd.DataFrame()


def performance_emoji(lower: float, upper: float, threshold: float = 3.0) ->str:
    """Function to determine the emoji based on change and confidence interval."""
    if upper < -threshold:
        return "🟢"  # Emoji for faster performance
    elif lower > threshold:
        return "🔴"  # Emoji for slower performance
    else:
        return "🔵"  # Emoji for insignificant change


# Iterate over the models and sizes, perform analysis, and populate the DataFrame
rows=[]
for model, size in timings_1:
    model_name = model.__name__

    if (model, size) not in timings_2:
        print(f"Warning: ({model.__name__}, {size}) found in file1 but not in file2 — skipping.", file=sys.stderr)
        continue


    # Calculate percentage change and confidence interval for init times
    (
        init_change,
        init_lower,
        init_upper,
    ) = bootstrap_percentage_change_confidence_interval(
        timings_1[(model, size)][0], timings_2[(model, size)][0]
    )
    init_emoji = performance_emoji(init_lower, init_upper)
    init_summary = (
        f"{init_emoji} {init_change:+.1f}% [{init_lower:+.1f}%, {init_upper:+.1f}%]"
    )

    # Calculate percentage change and confidence interval for run times
    run_change, run_lower, run_upper = bootstrap_percentage_change_confidence_interval(
        timings_1[(model, size)][1], timings_2[(model, size)][1]
    )
    run_emoji = performance_emoji(run_lower, run_upper)
    run_summary = (
        f"{run_emoji} {run_change:+.1f}% [{run_lower:+.1f}%, {run_upper:+.1f}%]"
    )

    # Append results to DataFrame
    rows.append({
        "Model": model_name,
        "Size": size,
        "Init time [95% CI]": init_summary,
        "Run time [95% CI]": run_summary,
    })

    results_df = pd.DataFrame(rows, columns=["Model", "Size", "Init time [95% CI]", "Run time [95% CI]"])

# Convert DataFrame to markdown with specified alignments
markdown_representation = results_df.to_markdown(index=False, tablefmt="github")

# Display the markdown representation
print(markdown_representation)
