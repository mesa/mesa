"""Base Scenario class."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from functools import partial
from itertools import count
from typing import Any, ClassVar

import numpy as np
import pandas as pd

SeedLike = int | np.integer | Sequence[int] | np.random.SeedSequence
RNGLike = np.random.Generator | np.random.BitGenerator


def rescale_samples(
    samples: np.ndarray,
    ranges: np.ndarray,
    *,
    inplace: bool = False,
) -> np.ndarray:
    """Rescale samples from the unit interval [0, 1] to parameter ranges.

    Parameters
    ----------
    samples : ndarray (n, d)
        Samples drawn from the unit interval.
    ranges : ndarray (d, 2)
        Parameter ranges given as [[min, max], ...].
    inplace : bool, optional
        If True, the input ``samples`` array is modified in place.
        If False (default), a new array containing the rescaled samples
        is returned.

    Returns:
    -------
    ndarray (n, d)
        Rescaled samples.

    Notes:
    -----
    The rescaling is performed using NumPy broadcasting. If ``inplace=True``,
    the original ``samples`` array is overwritten.
    """
    samples = np.asarray(samples)
    ranges = np.asarray(ranges)

    mins = ranges[:, 0]
    scale = ranges[:, 1] - mins

    if inplace:
        samples *= scale
        samples += mins
        return samples

    return samples * scale + mins


class Scenario:
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
        scenario_id: A unique identifier for this scenario, auto-generated starting from 0
        experiment_id: Identifies the design point (e.g., row in a QMC sample matrix)
        replication_id: Identifies the stochastic replication within a design point
        rng: Random number generator seed value

    Notes:
        All parameters are accessible via attribute access (scenario.param).
        Class-level attributes in subclasses serve as default values.
        Scenario instances are frozen after initialisation; parameters cannot be modified.
        To create replications with derived seeds, use replicate().
    """

    _ids: ClassVar[defaultdict] = defaultdict(partial(count, 0))
    _scenario_defaults: ClassVar[dict[str, Any]] = {}
    __slots__ = (
        "__dict__",
        "_frozen",
        "initial_rng_state",
        "replication_id",
        "rng",
        "scenario_id",
    )

    @classmethod
    def __init_subclass__(cls):
        """Called once when a subclass is created."""
        defaults = {}
        for base in reversed(cls.__mro__):
            if base is Scenario or base is object:
                continue
            annotations = getattr(base, "__annotations__", {})
            for key in annotations:
                if hasattr(base, key) and not key.startswith("_"):
                    defaults[key] = getattr(base, key)

        cls._scenario_defaults = defaults

    @classmethod
    def _reset_counter(cls):
        """Reset the scenario counter for this class."""
        cls._ids[cls] = count(0)

    def __init__(
        self,
        *,
        rng: RNGLike | SeedLike | None = None,
        scenario_id: int | None = None,
        replication_id: int | None = None,
        **kwargs,
    ):
        """Initialize a Scenario.

        Args:
            rng: Seed for the random number generator. Accepts any value accepted by
                numpy.random.default_rng(). scenario.rng is always a Generator after
                initialisation. The initial rng state is stored in scenario.initial_rng_state
                and used by spawn_replications() to derive child seeds.
            scenario_id: Index of the design point in the experiment matrix.
            replication_id: Index of the stochastic replication for this design point.
            **kwargs: All other scenario parameters (override class-level defaults).
        """
        self._frozen = False
        self.scenario_id = (
            next(self._ids[type(self)]) if scenario_id is None else scenario_id
        )
        self.replication_id = replication_id
        self.rng = np.random.default_rng(rng)
        self.initial_rng_state = self.rng.bit_generator.state

        self.__dict__.update(self._scenario_defaults)
        self.__dict__.update(kwargs)
        self._frozen = True

    def __setattr__(self, name: str, value: object) -> None:
        """Prevent any modification after initialisation."""
        if getattr(self, "_frozen", False):
            raise TypeError(
                f"Scenario is frozen; cannot set '{name}'. "
                "Create a new Scenario instance instead."
            )
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        """Prevent deletion of attributes after initialisation."""
        if getattr(self, "_frozen", False):
            raise TypeError(f"Scenario is frozen; cannot delete '{name}'.")
        super().__delattr__(name)

    def __getstate__(self):
        """Return state for pickling."""
        return (
            self.__dict__.copy(),
            self.scenario_id,
            self.replication_id,
            self.initial_rng_state,
        )

    def __setstate__(self, state):
        """Restore state when unpickling."""
        dict_state, scenario_id, replication_id, initial_rng_state = state
        self._frozen = False
        self.scenario_id = scenario_id
        self.replication_id = replication_id
        self.initial_rng_state = initial_rng_state
        bg_class = getattr(np.random, initial_rng_state["bit_generator"])
        bg = bg_class()
        bg.state = initial_rng_state
        self.rng = np.random.Generator(bg)
        self.__dict__.update(dict_state)
        self._frozen = True

    @property
    def _stdlib_seed(self) -> int:
        """Derive a reproducible stdlib seed from the initial rng state."""
        inner = self.initial_rng_state["state"]["state"]
        if hasattr(inner, "tolist"):
            return int(inner.tolist()[0]) % (2**31)
        return int(inner) % (2**31)

    def __iter__(self):
        """Iterate over (key, value) pairs of the user specified parameters (excluding rng)."""
        return iter(self.__dict__.items())

    def __len__(self):
        """Return number of user defined parameters (excluding rng)."""
        return len(self.__dict__)

    def to_dict(self) -> dict[str, Any]:
        """Return dict representation of the scenario."""
        return {
            **self.__dict__,
            "scenario_id": self.scenario_id,
            "replication_id": self.replication_id,
            "initial_rng_state": self.initial_rng_state,
        }

    def spawn_replications(self, n: int) -> list[Scenario]:
        """Spawn n replications of this scenario with deterministically derived seeds.

        Each replication has identical user provided parameters but a unique random number generator and replication_id.
        The rng is spawned from the original rng of the base scenario instance.

        Args:
            n: Number of replications to create.

        Returns:
            A list of n Scenario instances with replication_id 0..n-1.
        """
        inner = self.initial_rng_state["state"]["state"]
        entropy = inner.tolist() if hasattr(inner, "tolist") else inner
        child_seeds = np.random.SeedSequence(entropy).spawn(n)
        return [
            self.__class__(
                rng=child_seeds[i],
                scenario_id=self.scenario_id,
                replication_id=i,
                **self.__dict__,
            )
            for i in range(n)
        ]

    @classmethod
    def from_dataframe(
        cls,
        experiments: pd.DataFrame,
        *,
        rng: SeedLike | None = None,
        replications: int | None = None,
    ) -> list[Scenario]:
        """Turn a dataframe into a list of scenarios.

        Args:
           experiments: Dataframe containing the parameters for the scenarios.
           rng: the number of random seeds to use or a list of seeds.
           replications: the number of replications to create for each scenario

        Returns:
           a list of scenario instances

        If rng is an integer, numpy will be used to generate that many seed values.

        """
        scenarios = []

        for i, entry in enumerate(experiments.to_dict(orient="records")):
            scenario = cls(rng=rng, scenario_id=i, **entry)
            if replications is None:
                scenarios.append(scenario)
            else:
                for replication in scenario.spawn_replications(replications):
                    scenarios.append(replication)

        return scenarios

    @classmethod
    def from_ndarray(
        cls,
        experiments: np.ndarray,
        parameter_names: list[str],
        *,
        rng: SeedLike | None = None,
        replications: int | None = None,
    ) -> list[Scenario]:
        """Turn a numpy array into a list of scenarios.

        Args:
           experiments: Dataframe containing the parameters for the scenarios.
           parameter_names: the names of the parameters
           rng: the number of random seeds to use or a list of seeds.
           replications: the number of replications to create for each scenario

        Returns:
           a list of scenario instances

        If rng is an integer, numpy will be used to generate that many seed values.

        """
        if len(parameter_names) != experiments.shape[1]:
            raise ValueError(
                "The number of parameter names does not match the number of columns in the numpy array."
            )

        return cls.from_dataframe(
            pd.DataFrame(experiments, columns=parameter_names),
            rng=rng,
            replications=replications,
        )
