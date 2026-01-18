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
    return [
        {param_names[i]: scaled_samples[j, i] for i in range(dim)}
        for j in range(n_samples)
    ]
