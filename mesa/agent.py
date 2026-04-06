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
    from mesa.experimental.actions import Action
    from mesa.model import Model

from mesa.agentset import AgentSet


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

    def __init_subclass__(cls, **kwargs):
        """Called when DatasetTrackedAgent is subclassed."""
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
        self.current_action: Action | None = None
        self.model.register_agent(self)

        for dataset in self._datasets:
            self.model.data_registry[dataset].add_agent(self)

    def remove(self) -> None:
        """Remove and delete the agent from the model.

        If the agent is currently performing an action, the action's
        scheduled completion event is cancelled silently. The action's
        on_interrupt() callback is NOT fired, because the agent is being
        destroyed — not making a behavioral decision. The action moves
        to no defined end state; it is simply abandoned.

        If your action holds external resources (e.g., a Resource slot,
        a reservation, a lock), override Agent.remove() and call
        self.cancel_action() before super().remove() to ensure
        on_interrupt() fires and cleanup logic runs:

            def remove(self):
                self.cancel_action()  # Fires on_interrupt for cleanup
                super().remove()

        Notes:
            This is a deliberate design choice. The default silent
            cleanup is safe and avoids callbacks touching agent state
            during teardown. Models that need cleanup should opt in
            explicitly.
        """
        if self.current_action is not None:
            self.current_action._cancel_event()  # Silent cleanup, no callback
            self.current_action = None

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

        """
        agents = []

        if not args and not kwargs:
            for _ in range(n):
                agents.append(cls(model))
            return AgentSet(agents, random=model.random)

        # Prepare positional argument iterators
        arg_iters = []
        for arg in args:
            if isinstance(arg, (list, np.ndarray, tuple, pd.Series)) and len(arg) == n:
                arg_iters.append(arg)
            else:
                arg_iters.append(itertools.repeat(arg, n))

        # Prepare keyword argument iterators
        kw_keys = list(kwargs.keys())
        kw_val_iters = []
        for v in kwargs.values():
            if isinstance(v, (list, np.ndarray, tuple, pd.Series)) and len(v) == n:
                kw_val_iters.append(v)
            else:
                kw_val_iters.append(itertools.repeat(v, n))

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

    # Actions methods
    def start_action(self, action: Action) -> Action:
        """Start performing an action.

        The action must be in PENDING or INTERRUPTED state and the agent
        must not be currently performing another action.

        Args:
            action: The Action to perform. Must have been created with
                this agent as its agent.

        Returns:
            The started Action.

        Raises:
            ValueError: If the agent is already performing an action,
                or if the action doesn't belong to this agent.
        """
        if self.current_action is not None:
            raise ValueError(
                f"Agent {self.unique_id} is already performing an action "
                f"({self.current_action!r}). Use interrupt_for() or "
                f"cancel_action() first."
            )

        if action.agent is not self:
            raise ValueError(
                f"Action's agent (id={action.agent.unique_id}) does not match "
                f"this agent (id={self.unique_id})."
            )

        self.current_action = action
        action.start()

        # If the action completed instantly (duration=0), start() already
        # called _do_complete which cleared current_action via the Action.
        return action

    def interrupt_for(self, new_action: Action) -> bool:
        """Interrupt the current action and start a new one.

        If there is no current action, simply starts the new one. If the
        current action is non-interruptible, returns False and does nothing.

        Args:
            new_action: The Action to perform instead.

        Returns:
            True if the new action was started (either no current action,
            or the current one was successfully interrupted). False if the
            current action is non-interruptible.
        """
        if self.current_action is not None and not self.current_action.interrupt():
            return False
            # interrupt() already cleared current_action

        self.start_action(new_action)
        return True

    def cancel_action(self) -> bool:
        """Cancel the current action, ignoring interruptible flag.

        Calls on_interrupt with partial progress. Returns False only if
        there is no current action.

        Returns:
            True if an action was cancelled, False if idle.
        """
        if self.current_action is None:
            return False

        self.current_action.cancel()
        # cancel() already cleared current_action
        return True

    @property
    def is_busy(self) -> bool:
        """Whether the agent is currently performing an action."""
        return self.current_action is not None
