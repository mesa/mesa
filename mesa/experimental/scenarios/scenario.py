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
    from mesa.model import Model


class Scenario[M: Model]:
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
    __slots__ = ("__dict__", "_scenario_id", "model")

    @classmethod
    def _reset_counter(cls):
        """Reset the scenario counter for this class."""
        cls._ids[cls] = count(0)

    def __init__(self, *, rng: RNGLike | SeedLike | None = None, **kwargs):
        """Initialize a Scenario.

        Args:
            rng: a random number generator or valid seed value for a numpy generator.
            kwargs: all other scenario parameters

        """
        self.model: M | None = None
        self._scenario_id: int = (
            next(self._ids[self.__class__])
            if "_scenario_id" not in kwargs
            else kwargs.pop("_scenario_id")
        )
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


# def scenarios_from_dataframe(
#     experiments: pd.DataFrame, rng: int | Iterable[SeedLike]
# ) -> list[Scenario]:
#     """Turn a dataframe into a list of scenarios.
#
#     Args:
#        experiments: Dataframe containing the parameters for the scenarios.
#        rng: the number of random seeds to use or a list of seeds.
#
#     Returns:
#        a list of scenario instances
#
#     If rng is an integer, numpy will be used to generate that many seed values.
#
#     """
#     if not isinstance(rng, Iterable):
#         rng = np.random.default_rng(42).integers(0, high=sys.maxsize, size=(rng,))
#
#     scenarios = []
#     for i, entry in enumerate(experiments.to_dict(orient="records")):
#         for seed in rng:
#             scenarios.append(Scenario(rng=seed, _experiment_id=i, **entry))
#
#     return scenarios


# def scenarios_from_numpy(
#     experiments: np.ndarray, parameter_names: list[str], rng: int | Iterable[SeedLike]
# ) -> list[Scenario]:
#     """Turn a numpy array into a list of scenarios.
#
#     Args:
#        experiments: Dataframe containing the parameters for the scenarios.
#        parameter_names: the names of the parameters
#        rng: the number of random seeds to use or a list of seeds.
#
#     Returns:
#        a list of scenario instances
#
#     If rng is an integer, numpy will be used to generate that many seed values.
#
#     """
#     if len(parameter_names) != experiments.shape[1]:
#         raise ValueError(
#             "The number of parameter names does not match the number of columns in the numpy array."
#         )
#
#     return scenarios_from_dataframe(
#         pd.DataFrame(experiments, columns=parameter_names), rng
#     )
