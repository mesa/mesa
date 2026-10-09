"""Agent related classes.

Core Objects: Agent.
"""

# Postpone annotation evaluation to avoid NameError from forward references (PEP 563). Remove once Python 3.14+ is required.
from __future__ import annotations

import contextlib
import itertools
from random import Random
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from mesa.model import Model

from mesa.agentset import AgentSet, _resolve_per_agent_values


class Agent[M: Model]:
    """Base class for a model agent in Mesa.

    Attributes:
        model (Model): A reference to the model instance.
        unique_id (int): A unique identifier for this agent.

    Notes:
        Agents must be hashable to be used in an AgentSet.
        In Python 3, defining `__eq__` without `__hash__` makes an object unhashable,
        which will break AgentSet usage.
        unique_id is unique relative to a model instance and starts from 1

    """

    _datasets: ClassVar = set()
    _repr_excluded_fields: ClassVar[set[str]] = {"model", "current_action", "unique_id"}

    def __init_subclass__(cls, **kwargs):
        """Called when Agent is subclassed, giving each subclass its own dataset set."""
        super().__init_subclass__(**kwargs)
        # Each subclass gets its own dataset set
        # we use strings on this to avoid memory leaks
        # and ensure the retrieved dataset belongs to the same
        # model instance as the agent
        cls._datasets = set()

    def __init__(self, model: M, *args, **kwargs) -> None:
        """Create a new agent.

        Args:
            model (Model): The model instance in which the agent exists.
            args: Passed on to super.
            kwargs: Passed on to super.

        Notes:
            to make proper use of python's super, in each class remove the arguments and
            keyword arguments you need and pass on the rest to super
        """
        super().__init__(*args, **kwargs)

        self.model: M = model
        self.unique_id = None
        self.model.register_agent(self)

        for dataset in self._datasets:
            self.model.data_registry[dataset].add_agent(self)

    def remove(self) -> None:
        """Remove and delete the agent from the model."""
        with contextlib.suppress(KeyError):
            self.model.deregister_agent(self)

        # ensures models are also removed from datasets
        for dataset in self._datasets:
            self.model.data_registry[dataset].remove_agent(self)

    def step(self) -> None:
        """A single step of the agent."""

    def advance(self) -> None:  # noqa: D102
        pass

    @classmethod
    def create_agents[T: Agent](
        cls: type[T], model: Model, n: int, *args, **kwargs
    ) -> AgentSet[T]:
        """Create N agents.

        Args:
            model: the model to which the agents belong
            args: arguments to pass onto agent instances
                  each arg is either a single object or a sequence of length n
            n: the number of agents to create
            kwargs: keyword arguments to pass onto agent instances
                   each keyword arg is either a single object or a sequence of length n

        Returns:
            AgentSet containing the agents created.

        Warning:
            A list, tuple, ndarray, or pandas Series argument is treated as one
            value per agent and must have length n; a length mismatch raises
            ValueError. This is especially easy to hit with coordinate tuples:
            create_agents(model, 2, pos=(10, 20)) does NOT give both agents
            pos=(10, 20); it gives agent 0 pos=10 and agent 1 pos=20, since the
            tuple's length (2) matches n (2).

            To assign the same sequence value to every agent, broadcast it
            explicitly as a length-n list of that value, e.g.:
            create_agents(model, 2, pos=[(10, 20)] * 2)

        Raises:
            ValueError: If a sequence argument's length does not match n.

        """
        agents = []

        if not args and not kwargs:
            for _ in range(n):
                agents.append(cls(model))
            return AgentSet(agents, random=model.random)

        # Prepare positional argument iterators. A sequence must have length n
        # (assigned per agent); a length mismatch raises. Anything else is broadcast.
        arg_iters = [_resolve_per_agent_values(arg, n) for arg in args]

        # Prepare keyword argument iterators
        kw_keys = list(kwargs.keys())
        kw_val_iters = [_resolve_per_agent_values(v, n) for v in kwargs.values()]

        # If arg_iters is empty, zip(*[]) returns nothing, so we use repeat(())
        pos_iter = zip(*arg_iters) if arg_iters else itertools.repeat(())

        kw_iter = zip(*kw_val_iters) if kw_val_iters else itertools.repeat(())

        # We rely on range(n) to drive the loop length
        if kwargs:
            for _, p_args, k_vals in zip(range(n), pos_iter, kw_iter):
                agents.append(cls(model, *p_args, **dict(zip(kw_keys, k_vals))))
        else:
            for _, p_args in zip(range(n), pos_iter):
                agents.append(cls(model, *p_args))

        return AgentSet(agents, random=model.random)

    @classmethod
    def from_dataframe[T: Agent](
        cls: type[T], model: Model, df: pd.DataFrame, **kwargs
    ) -> AgentSet[T]:
        """Create agents from a pandas DataFrame.

        Each row of the DataFrame represents one agent. The DataFrame columns are
        mapped to the agent's constructor as keyword arguments. Additional keyword
        arguments (`**kwargs`) can be used to set constant attributes for all agents.

        Args:
            model: The model instance.
            df: The pandas DataFrame. Each row represents an agent.
            **kwargs: Constant values to pass to every agent's constructor.
                Only non-sequence data is allowed in kwargs to avoid ambiguity
                with DataFrame columns.

        Returns:
            AgentSet containing the agents created.

        Note:
            If you need to pass variable data or sequences, add them as columns
            to the DataFrame before calling this method.
        """
        for key, value in kwargs.items():
            if isinstance(value, (list, np.ndarray, tuple, pd.Series)):
                raise TypeError(
                    f"from_dataframe does not support sequence data in kwargs ('{key}'). "
                    "Please add this data to the DataFrame before calling from_dataframe."
                )

        agents = [
            cls(model, **{**record, **kwargs})
            for record in df.to_dict(orient="records")
        ]

        return AgentSet(agents, random=model.random)

    def __str__(self) -> str:
        """Return a human-readable string representation of the agent."""
        return f"{self.__class__.__name__}, agent_id = {self.unique_id}"

    def __repr__(self) -> str:
        """Return an unambiguous string representation including agent state."""
        # Get excluded fields (allows subclasses to override)
        excluded = self._repr_excluded_fields

        # Get user-defined attributes (exclude private and Mesa fields)
        user_attrs = {
            k: v
            for k, v in self.__dict__.items()
            if not k.startswith("_") and k not in excluded
        }

        if user_attrs:
            attr_str = ", ".join(f"{k}={v!r}" for k, v in user_attrs.items())
            return f"<{self.__class__.__name__} id={self.unique_id} {attr_str}>"
        else:
            return f"<{self.__class__.__name__} id={self.unique_id}>"

    @property
    def random(self) -> Random:
        """Return a seeded stdlib rng."""
        return self.model.random

    @property
    def rng(self) -> np.random.Generator:
        """Return a seeded np.random rng."""
        return self.model.rng

    @property
    def scenario(self):
        """Return the scenario associated with the model."""
        return self.model.scenario
