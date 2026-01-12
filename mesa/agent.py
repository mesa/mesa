"""Agent related classes.

Core Objects: Agent and AgentSet.
"""

# Postpone annotation evaluation to avoid NameError from forward references (PEP 563). Remove once Python 3.14+ is required.
from __future__ import annotations

import contextlib
import copy
import functools
import itertools
import operator
import warnings
import weakref
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Iterator, MutableSet, Sequence
from random import Random

# mypy
from typing import TYPE_CHECKING, Any, ClassVar, Literal, overload

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

    # this is a class level attribute
    # it is a dictionary, indexed by model instance
    # so, unique_id is unique relative to a model, and counting starts from 1
    _ids: ClassVar[defaultdict] = defaultdict(functools.partial(itertools.count, 1))

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
        self.unique_id: int = next(self._ids[model])
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

        class ListLike:
            """Make default arguments act as if they are in a list of length N.

            This is a helper class.
            """

            def __init__(self, value):
                self.value = value

            def __getitem__(self, i):
                return self.value

        listlike_args = []
        for arg in args:
            if isinstance(arg, (list | np.ndarray | tuple)) and len(arg) == n:
                listlike_args.append(arg)
            else:
                listlike_args.append(ListLike(arg))

        listlike_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, (list | np.ndarray | tuple)) and len(v) == n:
                listlike_kwargs[k] = v
            else:
                listlike_kwargs[k] = ListLike(v)

        agents = []
        for i in range(n):
            instance_args = [arg[i] for arg in listlike_args]
            instance_kwargs = {k: v[i] for k, v in listlike_kwargs.items()}
            agent = cls(model, *instance_args, **instance_kwargs)
            agents.append(agent)
        return AgentSet(agents, random=model.random)

    @property
    def random(self) -> Random:
        """Return a seeded stdlib rng."""
        return self.model.random

    @property
    def rng(self) -> np.random.Generator:
        """Return a seeded np.random rng."""
        return self.model.rng


class AgentSet[A: Agent](MutableSet[A], Sequence[A]):
    """A collection class that represents an ordered set of agents within an agent-based model (ABM).

    This class extends both MutableSet and Sequence, providing set-like functionality with order preservation and
    sequence operations.

    Attributes:
        model (Model): The ABM model instance to which this AgentSet belongs.

    Notes:
        The AgentSet maintains weak references to agents, allowing for efficient management of agent lifecycles
        without preventing garbage collection. It is associated with a specific model instance, enabling
        interactions with the model's environment and other agents.The implementation uses a WeakKeyDictionary to store agents,
        which means that agents not referenced elsewhere in the program may be automatically removed from the AgentSet.

    Notes:
        If random is None then the random number generator in the model of the first agent is used.
        If the agents list is empty and random is also None a user warning is issued and the AgentSet
        is an empty list and a default random number generator.  This can make models non-reproducible.
        If your code may create an AgentSet with no agents please pass a random number generator explicitly.

    """

    def __init__(
        self,
        agents: Iterable[A],
        random: Random | None = None,
    ):
        """Initializes the AgentSet with a collection of agents and a reference to the model.

        Args:
            agents (Iterable[Agent]): An iterable of Agent objects to be included in the set.
            random (Random | np.random.Generator | None): the random number generator
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

    def select(
        self,
        filter_func: Callable[[A], bool] | None = None,
        at_most: int | float = float("inf"),
        inplace: bool = False,
        agent_type: type[A] | None = None,
    ) -> AgentSet[A]:
        """Select a subset of agents from the AgentSet based on a filter function and/or quantity limit.

        Args:
            filter_func (Callable[[Agent], bool], optional): A function that takes an Agent and returns True if the
                agent should be included in the result. Defaults to None, meaning no filtering is applied.
            at_most (int | float, optional): The maximum amount of agents to select. Defaults to infinity.
              - If an integer, at most the first number of matching agents are selected.
              - If a float between 0 and 1, at most that fraction of original the agents are selected.
            inplace (bool, optional): If True, modifies the current AgentSet; otherwise, returns a new AgentSet. Defaults to False.
            agent_type (type[Agent], optional): The class type of the agents to select. Defaults to None, meaning no type filtering is applied.

        Returns:
            AgentSet: A new AgentSet containing the selected agents, unless inplace is True, in which case the current AgentSet is updated.

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

        return AgentSet(agents, self.random) if not inplace else self._update(agents)

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
            AgentSet | list[Any]: The results of the callable calls if return_results is True, otherwise the AgentSet itself.
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
        """
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

    def groupby(self, by: Callable | str, result_type: str = "agentset") -> GroupBy:
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
        of an AgentSet.

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
                {k: AgentSet(v, random=self.random) for k, v in groups.items()}
            )
        else:
            return GroupBy(groups)

    # consider adding for performance reasons
    # for Sequence: __reversed__, index, and count
    # for MutableSet clear, pop, remove, __ior__, __iand__, __ixor__, and __isub__


class GroupBy:
    """Helper class for AgentSet.groupby.

    Attributes:
        groups (dict): A dictionary with the group_name as key and group as values

    """

    def __init__(self, groups: dict[Any, list | AgentSet]):
        """Initialize a GroupBy instance.

        Args:
            groups (dict): A dictionary with the group_name as key and group as values

        """
        self.groups: dict[Any, list | AgentSet] = groups

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
