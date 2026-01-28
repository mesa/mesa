"""Agent related classes.

Core Objects: Agent and AgentSet.
"""

# Postpone annotation evaluation to avoid NameError from forward references (PEP 563). Remove once Python 3.14+ is required.
from __future__ import annotations

import contextlib
import copy
import itertools
import operator
import warnings
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Iterator, MutableSet, Sequence
from random import Random

# mypy
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np

if TYPE_CHECKING:
    from mesa.model import Model
    from mesa.space import Position


class Agent[M: Model]:
    """Base class for a model agent in Mesa.

    Attributes:
        model (Model): A reference to the model instance.
        unique_id (int): A unique identifier for this agent.
        pos (Position): A reference to the position where this agent is located.

    Notes:
        Agents must be hashable to be used in an AgentSet.
        In Python 3, defining `__eq__` without `__hash__` makes an object unhashable,
        which will break AgentSet usage.
        unique_id is unique relative to a model instance and starts from 1

    """

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
        self.pos: Position | None = None
        self.model.register_agent(self)

    def remove(self) -> None:
        """Remove and delete the agent from the model.

        Notes:
            If you need to do additional cleanup when removing an agent by for example removing
            it from a space, consider extending this method in your own agent class.

        """
        with contextlib.suppress(KeyError):
            self.model.deregister_agent(self)

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
            if isinstance(arg, (list, np.ndarray, tuple)) and len(arg) == n:
                arg_iters.append(arg)
            else:
                arg_iters.append(itertools.repeat(arg, n))

        # Prepare keyword argument iterators
        kw_keys = list(kwargs.keys())
        kw_val_iters = []
        for v in kwargs.values():
            if isinstance(v, (list, np.ndarray, tuple)) and len(v) == n:
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


class AbstractAgentSet[A: Agent](ABC, MutableSet[A]):
    """An abstract base collection class that represents an ordered set of agents within an agent-based model (ABM).

    This class defines the minimal interface that all AgentSet implementations must follow.
    Subclasses are free to override methods with optimized implementations based on their
    storage mechanism (weak references vs strong references).

    Attributes:
        model (Model): The ABM model instance to which this AbstractAgentSet belongs.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of agents in the AbstractAgentSet."""
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[A]:
        """Provide an iterator over the agents in the AbstractAgentSet."""
        ...

    @abstractmethod
    def __contains__(self, agent: A) -> bool:
        """Check if an agent is in the AgentSet."""
        ...

    @abstractmethod
    def _update(self, agents: Iterable[A]) -> AbstractAgentSet[A]:
        """Update the AbstractAgentSet a with new set of agents."""
        ...

    def select(
        self,
        filter_func: Callable[[A], bool] | None = None,
        at_most: int | float = float("inf"),
        inplace: bool = False,
        agent_type: type[A] | None = None,
    ) -> AbstractAgentSet[A]:
        """Select a subset of agents from the AbstractAgentSet based on a filter function and/or quantity limit.

        Args:
            filter_func (Callable[[Agent], bool], optional): A function that takes an Agent and returns True if the
                agent should be included in the result. Defaults to None, meaning no filtering is applied.
            at_most (int | float, optional): The maximum amount of agents to select. Defaults to infinity.
              - If an integer, at most the first number of matching agents are selected.
              - If a float between 0 and 1, at most that fraction of original the agents are selected.
            inplace (bool, optional): If True, modifies the current AbstractAgentSet; otherwise, returns a new AbstractAgentSet. Defaults to False.
            agent_type (type[Agent], optional): The class type of the agents to select. Defaults to None, meaning no type filtering is applied.

        Returns:
            AbstractAgentSet: A new AbstractAgentSet containing the selected agents, unless inplace is True, in which case the current AbstractAgentSet is updated.

        Notes:
            - at_most just return the first n or fraction of agents. To take a random sample, shuffle() beforehand.
            - at_most is an upper limit. When specifying other criteria, the number of agents returned can be smaller.
        """
        inf = float("inf")
        if filter_func is None and agent_type is None and at_most == inf:
            return self if inplace else copy.copy(self)

        # Check if at_most is of type float
        if at_most <= 1.0 and isinstance(at_most, float):
            at_most = int(len(self) * at_most)  # Note that it rounds down (floor)

        def agent_generator(filter_func, agent_type, at_most):
            count = 0
            for agent in self:
                if count >= at_most:
                    break
                if (not filter_func or filter_func(agent)) and (
                    not agent_type or isinstance(agent, agent_type)
                ):
                    yield agent
                    count += 1

        agents = agent_generator(filter_func, agent_type, at_most)

        # Use type(self) to ensure we return the correct subclass (AgentSet vs StrongAgentSet)
        return self._update(agents) if inplace else type(self)(agents, self.random)

    def agg(
        self, attribute: str, func: Callable | Iterable[Callable]
    ) -> Any | list[Any]:
        """Aggregate an attribute of all agents in the AgentSet using one or more functions.

        Args:
            attribute (str): The name of the attribute to aggregate.
            func (Callable | Iterable[Callable]):
                - If Callable: A single function to apply to the attribute values (e.g., min, max, sum, np.mean)
                - If Iterable: Multiple functions to apply to the attribute values

        Returns:
            Any | [Any, ...]: Result of applying the function(s) to the attribute values.

        Examples:
            # Single function
            avg_energy = model.agents.agg("energy", np.mean)

            # Multiple functions
            min_wealth, max_wealth, total_wealth = model.agents.agg("wealth", [min, max, sum])
        """
        values = self.get(attribute)

        if isinstance(func, Callable):
            return func(values)
        else:
            return [f(values) for f in func]

    @overload
    def get(
        self,
        attr_names: str,
        handle_missing: Literal["error", "default"] = "error",
        default_value: Any = None,
    ) -> list[Any]: ...

    @overload
    def get(
        self,
        attr_names: list[str],
        handle_missing: Literal["error", "default"] = "error",
        default_value: Any = None,
    ) -> list[list[Any]]: ...

    def get(
        self,
        attr_names,
        handle_missing="error",
        default_value=None,
    ):
        """Retrieve the specified attribute(s) from each agent in the AgentSet.

        Args:
            attr_names (str | list[str]): The name(s) of the attribute(s) to retrieve from each agent.
            handle_missing (str, optional): How to handle missing attributes. Can be:
                                            - 'error' (default): raises an AttributeError if attribute is missing.
                                            - 'default': returns the specified default_value.
            default_value (Any, optional): The default value to return if 'handle_missing' is set to 'default'
                                           and the agent does not have the attribute.

        Returns:
            list[Any]: A list with the attribute value for each agent if attr_names is a str.
            list[list[Any]]: A list with a lists of attribute values for each agent if attr_names is a list of str.

        Raises:
            AttributeError: If 'handle_missing' is 'error' and the agent does not have the specified attribute(s).
            ValueError: If an unknown 'handle_missing' option is provided.
        """
        is_single_attr = isinstance(attr_names, str)

        if handle_missing == "error":
            if is_single_attr:
                return [getattr(agent, attr_names) for agent in self._agents]
            else:
                return [
                    [getattr(agent, attr) for attr in attr_names]
                    for agent in self._agents
                ]

        elif handle_missing == "default":
            if is_single_attr:
                return [
                    getattr(agent, attr_names, default_value) for agent in self._agents
                ]
            else:
                return [
                    [getattr(agent, attr, default_value) for attr in attr_names]
                    for agent in self._agents
                ]

        else:
            raise ValueError(
                f"Unknown handle_missing option: {handle_missing}, "
                "should be one of 'error' or 'default'"
            )

    def set(self, attr_name: str, value: Any) -> AgentSet[A]:
        """Set a specified attribute to a given value for all agents in the AgentSet.

        Args:
            attr_name (str): The name of the attribute to set.
            value (Any): The value to set the attribute to.

        Returns:
            AgentSet: The AgentSet instance itself, after setting the attribute.
        """
        for agent in self:
            setattr(agent, attr_name, value)
        return self

    @abstractmethod
    def add(self, agent: A):
        """Add an agent to the AbstractAgentSet.

        Args:
            agent (Agent): The agent to add to the set.

        Note:
            This method is an implementation of the abstract method from MutableSet.
        """
        ...

    @abstractmethod
    def discard(self, agent: A):
        """Remove an agent from the AbstractAgentSet if it exists.

        This method does not raise an error if the agent is not present.

        Args:
            agent (Agent): The agent to remove from the set.

        Note:
            This method is an implementation of the abstract method from MutableSet.
        """
        ...

    @abstractmethod
    def remove(self, agent: A):
        """Remove an agent from the AbstractAgentSet.

        Raises:
            An Exception if the agent is not present.

        Args:
            agent (Agent): The agent to remove from the set.

        Note:
            This method is an implementation of the abstract method from MutableSet.
        """
        ...

    def groupby(
        self, by: Callable | str, result_type: Literal["agentset", "list"] = "agentset"
    ) -> GroupBy:
        """Group agents by the specified attribute or return from the callable.

        Args:
            by (Callable, str): used to determine what to group agents by

                                * if ``by`` is a callable, it will be called for each agent and the return is used
                                  for grouping
                                * if ``by`` is a str, it should refer to an attribute on the agent and the value
                                  of this attribute will be used for grouping

            result_type (str, optional): The datatype for the resulting groups {"agentset", "list"}

        Returns:
            GroupBy


        Notes:
            There might be performance benefits to using `result_type='list'` if you don't need the advanced functionality
            of an AbstractAgentSet.
        """
        groups = defaultdict(list)

        if isinstance(by, Callable):
            for agent in self:
                groups[by(agent)].append(agent)
        else:
            for agent in self:
                groups[getattr(agent, by)].append(agent)

        if result_type == "agentset":
            return GroupBy(
                {k: type(self)(v, random=self.random) for k, v in groups.items()}
            )
        else:
            return GroupBy(groups)

    # Performance-critical methods are left abstract

    @abstractmethod
    def shuffle(self, inplace: bool = False) -> AbstractAgentSet[A]:
        """Randomly shuffle the order of agents in the AbstractAgentSet."""
        ...

    @abstractmethod
    def sort(
        self,
        key: Callable[[A], Any] | str,
        ascending: bool = False,
        inplace: bool = False,
    ) -> AbstractAgentSet[A]:
        """Sort the agents in the AbstractAgentSet based on a specified attribute or custom function."""
        ...

    @abstractmethod
    def do(self, method: str | Callable, *args, **kwargs) -> AbstractAgentSet[A]:
        """Invoke a method or function on each agent in the AbstractAgentSet."""
        ...

    @abstractmethod
    def shuffle_do(
        self, method: str | Callable, *args, **kwargs
    ) -> AbstractAgentSet[A]:
        """Shuffle the agents in the AbstractAgentSet and then invoke a method or function on each agent."""
        ...

    @abstractmethod
    def map(self, method: str | Callable, *args, **kwargs) -> list[Any]:
        """Invoke a method or function on each agent in the AbstractAgentSet and return the results."""
        ...


class AgentSet[A: Agent](AbstractAgentSet[A], Sequence[A]):
    """A collection class that represents an ordered set of agents using weak references.

    This implementation uses weak references to agents, allowing for efficient management
    of agent lifecycles without preventing garbage collection.

    Attributes:
        random (Random): The random number generator for this agent set.

    Notes:
        The AgentSet maintains weak references to agents, which means that agents not
        referenced elsewhere in the program may be automatically removed from the AgentSet.
        This is the default implementation for most use cases where automatic cleanup is desired.

        Performance-critical methods are optimized to work directly with weak references,
        avoiding the overhead of creating strong references during iteration.
    """

    def __init__(
        self,
        agents: Iterable[A],
        random: Random | None = None,
    ):
        """Initialize the AgentSet with weak references to agents.

        Args:
            agents (Iterable[Agent]): An iterable of Agent objects to be included in the set.
            random (Random | None): The random number generator for this agent set.
        """
        self._agents = weakref.WeakKeyDictionary(dict.fromkeys(agents))
        if (len(self._agents) == 0) and random is None:
            warnings.warn(
                "No Agents specified in creation of AgentSet and no random number generator specified. "
                "This can make models non-reproducible. Please pass a random number generator explicitly",
                UserWarning,
                stacklevel=2,
            )
            random = Random()

        if random is not None:
            self.random = random
        else:
            # all agents in an AgentSet should share the same model, just take it from first
            self.random = self._agents.keys().__next__().model.random

    def __len__(self) -> int:
        """Return the number of agents in the AgentSet."""
        return len(self._agents)

    def __iter__(self) -> Iterator[A]:
        """Provide an iterator over the agents in the AgentSet."""
        return self._agents.keys()

    def __contains__(self, agent: A) -> bool:
        """Check if an agent is in the AgentSet. Can be used like `agent in agentset`."""
        return agent in self._agents

    def shuffle(self, inplace: bool = False) -> AgentSet[A]:
        """Randomly shuffle the order of agents in the AgentSet.

        Args:
            inplace (bool, optional): If True, shuffles the agents in the current AgentSet; otherwise, returns a new shuffled AgentSet. Defaults to False.

        Returns:
            AgentSet: A shuffled AgentSet. Returns the current AgentSet if inplace is True.

        Note:
            Using inplace = True is more performant

        """
        weakrefs = list(self._agents.keyrefs())
        self.random.shuffle(weakrefs)

        if inplace:
            self._agents.data = dict.fromkeys(weakrefs)
            return self
        else:
            return AgentSet(
                (agent for ref in weakrefs if (agent := ref()) is not None), self.random
            )

    def sort(
        self,
        key: Callable[[A], Any] | str,
        ascending: bool = False,
        inplace: bool = False,
    ) -> AgentSet[A]:
        """Sort the agents in the AgentSet based on a specified attribute or custom function.

        Args:
            key (Callable[[Agent], Any] | str): A function or attribute name based on which the agents are sorted.
            ascending (bool, optional): If True, the agents are sorted in ascending order. Defaults to False.
            inplace (bool, optional): If True, sorts the agents in the current AgentSet; otherwise, returns a new sorted AgentSet. Defaults to False.

        Returns:
            AgentSet: A sorted AgentSet. Returns the current AgentSet if inplace is True.
        """
        if isinstance(key, str):
            key = operator.attrgetter(key)

        sorted_agents = sorted(self._agents.keys(), key=key, reverse=not ascending)

        return (
            AgentSet(sorted_agents, self.random)
            if not inplace
            else self._update(sorted_agents)
        )

    def _update(self, agents: Iterable[A]):
        """Update the AgentSet with a new set of agents.

        This is a private method primarily used internally by other methods like select, shuffle, and sort.
        """
        self._agents = weakref.WeakKeyDictionary(dict.fromkeys(agents))
        return self

    def do(self, method: str | Callable, *args, **kwargs) -> AgentSet[A]:
        """Invoke a method or function on each agent in the AgentSet.

        Args:
            method (str, callable): the callable to do on each agent

                                        * in case of str, the name of the method to call on each agent.
                                        * in case of callable, the function to be called with each agent as first argument

            *args: Variable length argument list passed to the callable being called.
            **kwargs: Arbitrary keyword arguments passed to the callable being called.

        Returns:
            AgentSet: The AgentSet instance itself.
        """
        # we iterate over the actual weakref keys and check if weakref is alive before calling the method
        if isinstance(method, str):
            for agentref in self._agents.keyrefs():
                if (agent := agentref()) is not None:
                    getattr(agent, method)(*args, **kwargs)
        else:
            for agentref in self._agents.keyrefs():
                if (agent := agentref()) is not None:
                    method(agent, *args, **kwargs)

        return self

    def shuffle_do(self, method: str | Callable, *args, **kwargs) -> AgentSet[A]:
        """Shuffle the agents in the AgentSet and then invoke a method or function on each agent.

        It's a fast, optimized version of calling shuffle() followed by do().
        """
        weakrefs = list(self._agents.keyrefs())
        self.random.shuffle(weakrefs)

        if isinstance(method, str):
            for ref in weakrefs:
                if (agent := ref()) is not None:
                    getattr(agent, method)(*args, **kwargs)
        else:
            for ref in weakrefs:
                if (agent := ref()) is not None:
                    method(agent, *args, **kwargs)

        return self

    def map(self, method: str | Callable, *args, **kwargs) -> list[Any]:
        """Invoke a method or function on each agent in the AgentSet and return the results.

        Args:
            method (str, callable): the callable to apply on each agent

                                        * in case of str, the name of the method to call on each agent.
                                        * in case of callable, the function to be called with each agent as first argument

            *args: Variable length argument list passed to the callable being called.
            **kwargs: Arbitrary keyword arguments passed to the callable being called.

        Returns:
           list[Any]: The results of the callable calls
        """
        # we iterate over the actual weakref keys and check if weakref is alive before calling the method
        if isinstance(method, str):
            res = [
                getattr(agent, method)(*args, **kwargs)
                for agentref in self._agents.keyrefs()
                if (agent := agentref()) is not None
            ]
        else:
            res = [
                method(agent, *args, **kwargs)
                for agentref in self._agents.keyrefs()
                if (agent := agentref()) is not None
            ]

        return res

    def to_list(self) -> list[A]:
        """Convert the AgentSet to a list.

        Returns:
            list[Agent]: A list containing all agents in the AgentSet.

        Notes:
            This method provides an explicit way to convert the AgentSet to a list.
            It is the recommended approach when list operations (indexing, slicing)
            are needed, as direct sequence operations on AgentSet are deprecated
            and will be removed in Mesa 4.0.
        """
        return list(self._agents.keys())

    @overload
    def __getitem__(self, item: int) -> A: ...

    @overload
    def __getitem__(self, item: slice) -> list[A]: ...

    def __getitem__(self, item):
        """Retrieve an agent or a slice of agents from the AgentSet.

        Args:
            item (int | slice): The index or slice for selecting agents.

        Returns:
            Agent | list[Agent]: The selected agent or list of agents based on the index or slice provided.

        .. deprecated::
            Sequence behavior (indexing/slicing) is deprecated and will be removed in Mesa 4.0.
            Use :meth:`to_list` instead: ``agentset.to_list()[index]`` or ``agentset.to_list()[start:stop]``.
        """
        warnings.warn(
            "AgentSet.__getitem__ is deprecated and will be removed in Mesa 4.0. "
            "Use AgentSet.to_list()[index] instead. "
            "See https://mesa.readthedocs.io/latest/migration_guide.html#AgentSet-sequence-behavior",
            PendingDeprecationWarning,
            stacklevel=2,
        )
        return list(self._agents.keys())[item]

    def add(self, agent: A):
        """Add an agent to the AgentSet.

        Args:
            agent (Agent): The agent to add to the set.

        Note:
            This method is an implementation of the abstract method from MutableSet.
        """
        self._agents[agent] = None

    def discard(self, agent: A):
        """Remove an agent from the AgentSet if it exists.

        This method does not raise an error if the agent is not present.

        Args:
            agent (Agent): The agent to remove from the set.

        Note:
            This method is an implementation of the abstract method from MutableSet.
        """
        with contextlib.suppress(KeyError):
            del self._agents[agent]

    def remove(self, agent: A):
        """Remove an agent from the AgentSet.

        This method raises an error if the agent is not present.

        Args:
            agent (Agent): The agent to remove from the set.

        Note:
            This method is an implementation of the abstract method from MutableSet.
        """
        del self._agents[agent]

    def __getstate__(self):
        """Retrieve the state of the AgentSet for serialization.

        Returns:
            dict: A dictionary representing the state of the AgentSet.
        """
        return {"agents": list(self._agents.keys()), "random": self.random}

    def __setstate__(self, state):
        """Set the state of the AgentSet during deserialization.

        Args:
            state (dict): A dictionary representing the state to restore.
        """
        self.random = state["random"]
        self._update(state["agents"])


class GroupBy:
    """Helper class for AgentSet.groupby.

    Attributes:
        groups (dict): A dictionary with the group_name as key and group as values

    """

    def __init__(self, groups: dict[Any, list | AbstractAgentSet]):
        """Initialize a GroupBy instance.

        Args:
            groups (dict): A dictionary with the group_name as key and group as values

        """
        self.groups: dict[Any, list | AbstractAgentSet] = groups

    def map(self, method: Callable | str, *args, **kwargs) -> dict[Any, Any]:
        """Apply the specified callable to each group and return the results.

        Args:
            method (Callable, str): The callable to apply to each group,

                                    * if ``method`` is a callable, it will be called it will be called with the group as first argument
                                    * if ``method`` is a str, it should refer to a method on the group

                                    Additional arguments and keyword arguments will be passed on to the callable.
            args: arguments to pass to the callable
            kwargs: keyword arguments to pass to the callable

        Returns:
            dict with group_name as key and the return of the method as value

        Notes:
            this method is useful for methods or functions that do return something. It
            will break method chaining. For that, use ``do`` instead.

        """
        if isinstance(method, str):
            return {
                k: getattr(v, method)(*args, **kwargs) for k, v in self.groups.items()
            }
        else:
            return {k: method(v, *args, **kwargs) for k, v in self.groups.items()}

    def do(self, method: Callable | str, *args, **kwargs) -> GroupBy:
        """Apply the specified callable to each group.

        Args:
            method (Callable, str): The callable to apply to each group,

                                    * if ``method`` is a callable, it will be called it will be called with the group as first argument
                                    * if ``method`` is a str, it should refer to a method on the group

                                    Additional arguments and keyword arguments will be passed on to the callable.
            args: arguments to pass to the callable
            kwargs: keyword arguments to pass to the callable

        Returns:
            the original GroupBy instance

        Notes:
            this method is useful for methods or functions that don't return anything and/or
            if you want to chain multiple do calls

        """
        if isinstance(method, str):
            for v in self.groups.values():
                getattr(v, method)(*args, **kwargs)
        else:
            for v in self.groups.values():
                method(v, *args, **kwargs)

        return self

    def count(self) -> dict[Any, int]:
        """Return the count of agents in each group.

        Returns:
            dict: A dictionary mapping group names to the number of agents in each group.
        """
        return {k: len(v) for k, v in self.groups.items()}

    def agg(self, attr_name: str, func: Callable) -> dict[Hashable, Any]:
        """Aggregate the values of a specific attribute across each group using the provided function.

        Args:
            attr_name (str): The name of the attribute to aggregate.
            func (Callable): The function to apply (e.g., sum, min, max, mean).

        Returns:
            dict[Hashable, Any]: A dictionary mapping group names to the result of applying the aggregation function.
        """
        return {
            group_name: func([getattr(agent, attr_name) for agent in group])
            for group_name, group in self.groups.items()
        }

    def __iter__(self):  # noqa: D105
        return iter(self.groups.items())

    def __len__(self):  # noqa: D105
        return len(self.groups)
