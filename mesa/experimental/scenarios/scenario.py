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

    It is recommended to add a property to your agents to make scenario access
    easy inside your agent. For example:

    ::
        @property
        def scenario(self):
            return self.model.scenario

    """

    _ids: ClassVar[defaultdict] = defaultdict(partial(count, 0))
    __slots__ = ("__dict__", "model", "_scenario_id")

    def __init__(self, *, rng: RNGLike | SeedLike | None = None, **kwargs):
        """Initialize a Scenario.

        Args:
            rng: a random number generator or valid seed value for a numpy generator.
            kwargs: all other scenario parameters

        """
        self.model: M | None = None
        self._scenario_id: int = next(self._ids[self.__class__]) if "_scenario_id" not in kwargs else kwargs.pop("_scenario_id")
        self.__dict__.update(rng=rng, **kwargs)

    def __iter__(self):  # noqa: D105
        return iter(self.__dict__.items())

    def __len__(self):  # noqa: D105
        return len(self.__dict__)

    def __setattr__(self, name: str, value: object) -> None:  # noqa: D105
        try:
            if self.model.running:
                raise ValueError("Cannot change scenario parameters during model run.")
        except AttributeError:
            # happens when we do self.model = None in init
            pass
        super().__setattr__(name, value)

    def to_dict(self):
        """Return a dict representation of the scenario."""
        return {**self.__dict__, "model": self.model, "_scenario_id": self._scenario_id}