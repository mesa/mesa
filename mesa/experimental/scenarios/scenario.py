"""Base Scenario class."""

from collections import defaultdict
from collections.abc import Sequence
from functools import partial
from itertools import count
from typing import TYPE_CHECKING, ClassVar

import numpy as np

SeedLike = int | np.integer | Sequence[int] | np.random.SeedSequence
RNGLike = np.random.Generator | np.random.BitGenerator


if TYPE_CHECKING:
    from .model_with_scenario import ModelWithScenario


class Scenario[M: ModelWithScenario]:
    """A Scenario class.

    Attributes:
        model : the model instance to which this scenario belongs
        scenario_id : a unique identifier for this scenario, auto-generated, starting from 0

    Notes:
        all additional parameters are stored as attributes of the scenario and
        are thus available via property access.

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

    def __iter__(self):  # noqa: D105
        return iter(self.__dict__)

    def __len__(self):  # noqa: D105
        return len(self.__dict__)

    def to_dict(self):
        """Return a dict representation of the scenario."""
        data = self.__dict__.copy()
        for entry in self.__slots__:
            if entry != "__dict__":
                data[entry] = getattr(self, entry)
        return data
