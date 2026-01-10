"""Base Scenario class."""

from collections import defaultdict
from collections.abc import MutableMapping, Sequence
from functools import partial
from itertools import count
from typing import TYPE_CHECKING, ClassVar

import numpy as np

SeedLike = int | np.integer | Sequence[int] | np.random.SeedSequence
RNGLike = np.random.Generator | np.random.BitGenerator


if TYPE_CHECKING:
    from .model_with_scenario import ModelWithScenario


class Scenario[M: ModelWithScenario](MutableMapping):
    """A Scenario class.

    Attributes:
        model : the model instance to which this scenario belongs
        scenario_id : a unique identifier for this scenario, auto-generated, starting from 0

    Notes:
        in essence, this is a mutable mapping with
        protection, so it cannot be mutated while
        model.running is true.

    """

    _ids: ClassVar[defaultdict] = defaultdict(partial(count, 0))

    __slots__ = ("__dict__", "model", "scenario_id")

    def __init__(self, *, rng: RNGLike | SeedLike | None = None, **kwargs):
        """Initialize a Scenario.

        Args:
            rng: a random number generator or valid seed value for a numpy generator.
            kwargs: all other scenario parameters

        """
        self.model: M | None = None
        self.scenario_id: int = next(self._ids[self.__class__])
        self.__dict__.update(rng=rng, **kwargs)

    def __setitem__(self, key, value):  # noqa: D105
        if self.model is not None and self.model.running:
            raise ValueError("Cannot mutate scenario while model is running")

        self.__dict__[key] = value

    def __getitem__(self, key):  # noqa: D105
        return self.__dict__[key]

    def __delitem__(self, key):  # noqa: D105
        del self.__dict__[key]

    def __iter__(self):  # noqa: D105
        return iter(self.__dict__)

    def __len__(self):  # noqa: D105
        return len(self.__dict__)

    def __setattr__(self, key, value):  # noqa: D105
        if key not in self.__slots__:
            self.__setitem__(key, value)
        else:
            super().__setattr__(key, value)

    def __delattr__(self, key):  # noqa: D105
        if key not in self.__slots__:
            self.__delitem__(key)
        else:
            super().__delattr__(key)

    def to_dict(self):
        """Return a dict representation of the scenario."""
        content = self.__dict__.copy()
        for entry in self.__slots__:
            if entry != "__dict__":
                content[entry] = getattr(self, entry)
        return content
