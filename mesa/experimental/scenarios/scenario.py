"""Base Scenario class."""

from collections import defaultdict
from collections.abc import Sequence
from functools import partial
from itertools import count
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

SeedLike = int | np.integer | Sequence[int] | np.random.SeedSequence
RNGLike = np.random.Generator | np.random.BitGenerator


if TYPE_CHECKING:
    from mesa.model import Model


class Scenario[M: Model]:
    """A Scenario class for defining model parameters and experiments.

    Supports both simple instantiation and type-hinted subclassing:

        # Simple usage
        scenario = Scenario(rng=42, density=0.8, minority_pc=0.5)

        # Type-hinted subclass (recommended for complex models)
        class MyScenario(Scenario):
            citizen_density: float = 0.7
            cop_vision: int = 7
            movement: bool = True

        scenario = MyScenario(rng=42, cop_vision=10)  # Override defaults

    Attributes:
        model: The model instance to which this scenario belongs
        scenario_id: A unique identifier for this scenario, auto-generated starting from 0
        rng: Random number generator or seed value

    Notes:
        All parameters are accessible via attribute access (scenario.param).
        Class-level attributes in subclasses serve as default values.
        Scenario parameters cannot be modified during model execution.
    """

    _ids: ClassVar[defaultdict] = defaultdict(partial(count, 0))
    _scenario_defaults: ClassVar[dict[str, Any]] = {}
    __slots__ = ("__dict__", "_scenario_id", "model")

    @classmethod
    def __init_subclass__(cls):
        """Called once when a subclass is created."""
        # Collect defaults once and cache on the class
        defaults = {}
        for base in reversed(cls.__mro__):
            if base is Scenario or base is object:
                continue
            annotations = getattr(base, "__annotations__", {})
            for key in annotations:
                if hasattr(base, key) and not key.startswith("_"):
                    defaults[key] = getattr(base, key)

        # Cache on the class itself
        cls._scenario_defaults = defaults

    @classmethod
    def _reset_counter(cls):
        """Reset the scenario counter for this class."""
        cls._ids[cls] = count(0)

    def __init__(self, *, rng: RNGLike | SeedLike | None = None, **kwargs):
        """Initialize a Scenario.

        Args:
            rng: Random number generator or valid seed value
            **kwargs: All other scenario parameters (override class-level defaults)
        """
        self.model: M | None = None
        self._scenario_id: int = (
            next(self._ids[self.__class__])
            if "_scenario_id" not in kwargs
            else kwargs.pop("_scenario_id")
        )
        self.__dict__.update(self._scenario_defaults)
        self.__dict__.update(kwargs)
        self.__dict__["rng"] = rng

    def __iter__(self):
        """Iterate over (key, value) pairs."""
        return iter(self.__dict__.items())

    def __len__(self):
        """Return number of parameters."""
        return len(self.__dict__)

    def __setattr__(self, name: str, value: object) -> None:
        """Prevent modification during model execution."""
        try:
            if self.model and self.model.running:
                raise ValueError(
                    f"Cannot change scenario parameter '{name}' during model run."
                )
        except AttributeError:
            # During initialization when self.model doesn't exist yet
            pass
        super().__setattr__(name, value)

    def to_dict(self) -> dict[str, Any]:
        """Return dict representation of the scenario."""
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
