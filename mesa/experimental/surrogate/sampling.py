"""Utilities for parameter space sampling using Latin Hypercube Sampling."""

from scipy.stats import qmc


def sample_parameters(param_space, n_samples, seed=None):
    """Generates a list of parameter dictionaries using Latin Hypercube Sampling."""
    dim = len(param_space)
    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    sample = sampler.random(n=n_samples)

    l_bounds = [v[0] for v in param_space.values()]
    u_bounds = [v[1] for v in param_space.values()]
    scaled_samples = qmc.scale(sample, l_bounds, u_bounds)

    param_names = list(param_space.keys())
    is_int = [
        isinstance(v[0], int) and isinstance(v[1], int) for v in param_space.values()
    ]

    output = []
    for j in range(n_samples):
        config = {}
        for i in range(dim):
            val = scaled_samples[j, i]
            config[param_names[i]] = round(val) if is_int[i] else val
        output.append(config)
    return output
