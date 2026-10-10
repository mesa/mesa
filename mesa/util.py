"""Utilities used across mesa."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from random import Random

import numpy as np

SeedLike = int | np.integer | Sequence[int] | np.random.SeedSequence
RNGLike = np.random.Generator | np.random.BitGenerator


def resolve_rng(
    random: Random | None = None,
    rng: RNGLike | SeedLike | None = None,
    *,
    stacklevel: int = 3,
) -> np.random.Generator | None:
    """Resolve the random number generator passed to a constructor.

    ``rng`` is the recommended argument and takes a ``numpy.random.Generator``
    or any value accepted by ``numpy.random.default_rng``. The stdlib
    ``random.Random`` instance passed via the deprecated ``random`` argument is
    converted to a generator seeded with its full state, so models seeded with
    ``random.Random(seed)`` stay reproducible.

    Args:
        random: a seeded stdlib random.Random instance. Deprecated in favor of rng.
        rng: a numpy.random.Generator or a value accepted by numpy.random.default_rng.
        stacklevel: the stacklevel of the deprecation warning, so it points at
            the code that instantiated the class.

    Returns:
        The numpy generator to use, or None if neither argument was passed.

    Raises:
        ValueError: if both random and rng are passed.
    """
    if random is not None:
        warnings.warn(
            "The `random` keyword argument is deprecated and will be removed in Mesa 5.0. "
            "Use `rng` instead. "
            "See https://mesa.readthedocs.io/latest/migration_guide.html#random-keyword-argument-deprecated-in-favor-of-rng",
            PendingDeprecationWarning,
            stacklevel=stacklevel,
        )
        if rng is not None:
            raise ValueError("Pass either rng or random, not both.")
        # seed with the full state of the stdlib generator so that models
        # seeded via random.Random(seed) remain reproducible
        rng = random.getstate()[1]

    if rng is None:
        return None
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)
