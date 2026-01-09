"""Base Scenario class."""

from collections.abc import MutableMapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

SeedLike = int | np.integer | Sequence[int] | np.random.SeedSequence
RNGLike = np.random.Generator | np.random.BitGenerator


if TYPE_CHECKING:
    from .model_with_scenario import ModelWithScenario

class Scenario[M: ModelWithScenario](MutableMapping):
    """A Scenario class.

    Notes:
        in essence, this is a mutable mapping with
        protection, so it cannot be mutated while
        model.running is true.

    """

    def __init__(self, *, rng:RNGLike | SeedLike | None =None, **kwargs):
        """Initialize a Scenario.

        Args:
            rng: a random number generator or valid seed value for a numpy generator.
            kwargs: all other scenario parameters

        """
        self.__dict__.update(rng=rng, **kwargs)
        self.model: M|None= None

    def __setitem__(self, key, value):# noqa: D105
        if self.model.running:
            raise ValueError("Cannot mutate scenario while model is running")
        self.__dict__[key] = value

    def __getitem__(self, key):# noqa: D105
        return self.__dict__[key]

    def __delitem__(self, key):# noqa: D105
        del self.__dict__[key]

    def __iter__(self):# noqa: D105
        return iter(self.__dict__)

    def __len__(self): # noqa: D105
        return len(self.__dict__)

