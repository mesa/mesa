"""compare timings across 2 benchmarks."""

import pickle

import numpy as np
import pandas as pd

filename1 = "timings_1"
filename2 = "timings_2"

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


def performance_emoji(lower, upper):
    """Function to determine the emoji based on change and confidence interval."""
    if upper < -3:
        return "🟢"  # Emoji for faster performance
    elif lower > 3:
        return "🔴"  # Emoji for slower performance
    else:
        return "🔵"  # Emoji for insignificant change


# Iterate over the models and sizes, perform analysis, and populate the DataFrame
for model, size in timings_1:
    model_name = model.__name__

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
    row = pd.DataFrame(
        {
            "Model": [model_name],
            "Size": [size],
            "Init time [95% CI]": [init_summary],
            "Run time [95% CI]": [run_summary],
        }
    )

    results_df = pd.concat([results_df, row], ignore_index=True)

# Convert DataFrame to markdown with specified alignments
markdown_representation = results_df.to_markdown(index=False, tablefmt="github")

# Display the markdown representation
print(markdown_representation)
