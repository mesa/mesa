"""
Implementation of Mesa's meta agent capability.

Overview:
Meta-agents allow modeling hierarchical systems where agents are composed of
other agents (e.g., organizations, cities, ecosystems).

This module provides:
- Helper functions to find and evaluate agent combinations
- Utilities to dynamically create meta-agents
- A MetaAgent class that can contain other agents
"""

import itertools
from collections.abc import Callable, Iterable, Iterator
from types import MethodType
from typing import Any

from mesa.agent import Agent, AgentSet


def evaluate_combination(
    candidate_group: tuple[Agent, ...],
    model: Any,
    evaluation_func: Callable[[tuple[Agent, ...]], float] | None,
) -> tuple[tuple[Agent, ...], float] | None:
    """Evaluate a combination of agents.

    Args:
        candidate_group: The group of agents to evaluate.
        model: The model instance.
        evaluation_func: Function used to evaluate the group.

    Returns:
        (group, score) if evaluation_func is provided, else None.
    """
    if evaluation_func:
        value = evaluation_func(candidate_group)
        return candidate_group, value
    return None


def find_combinations(
    model: Any,
    group: Iterable[Agent],
    size: int | tuple[int, int] = (2, 5),
    evaluation_func: Callable[[tuple[Agent, ...]], float] | None = None,
    filter_func: Callable[
        [list[tuple[tuple[Agent, ...], float]]],
        list[tuple[tuple[Agent, ...], float]],
    ]
    | None = None,
) -> list[tuple[tuple[Agent, ...], float]]:
    """Find valuable combinations of agents.

    Args:
        model: The model instance.
        group: Agents to evaluate.
        size: Combination size or range.
        evaluation_func: Function to score combinations.
        filter_func: Optional filtering function.

    Returns:
        List of (agent_group, score) tuples.
    """
    combinations: list[tuple[tuple[Agent, ...], float]] = []

    size_range = (size, size + 1) if isinstance(size, int) else size

    for group_size in range(*size_range):
        for candidate_group in itertools.combinations(group, group_size):
            result = evaluate_combination(candidate_group, model, evaluation_func)
            if result is not None:
                evaluated_group, score = result
                if score is not None:
                    combinations.append((evaluated_group, score))

    if combinations and filter_func:
        return filter_func(combinations)

    return combinations


def extract_class(
    agents_by_type: dict[type, list[Agent]],
    new_agent_class: str,
) -> type[Agent] | None:
    """Extract agent class by name.

    Args:
        agents_by_type: Mapping of agent types to instances.
        new_agent_class: Name of class to extract.

    Returns:
        Agent class if found, else None.
    """
    agent_types_by_name: dict[str, type] = {
        agent.__name__: agent for agent in agents_by_type
    }

    if new_agent_class in agent_types_by_name:
        agent_type = agent_types_by_name[new_agent_class]
        return type(next(iter(agents_by_type[agent_type])))

    return None


def create_meta_agent(
    model: Any,
    new_agent_class: str,
    agents: Iterable[Agent],
    mesa_agent_type: type[Agent] | None,
    meta_attributes: dict[str, Any] | None = None,
    meta_methods: dict[str, Callable] | None = None,
    assume_constituting_agent_methods: bool = False,
    assume_constituting_agent_attributes: bool = False,
) -> Any | None:
    """Create or update a meta-agent instance."""

    # Ensure uniqueness while preserving order
    agents = list(dict.fromkeys(agents))

    # Normalize base class
    if not mesa_agent_type:
        mesa_agent_type = (Agent,)
    elif not isinstance(mesa_agent_type, tuple):
        mesa_agent_type = (mesa_agent_type,)

    def add_methods(
        meta_agent_instance: Any,
        agents: Iterable[Agent],
        meta_methods: dict[str, Callable] | None,
    ) -> None:
        meta_methods = meta_methods or {}

        if assume_constituting_agent_methods:
            agent_classes = {type(agent) for agent in agents}
            for agent_class in agent_classes:
                for name, value in agent_class.__dict__.items():
                    if callable(value) and not name.startswith("__"):
                        meta_methods[name] = value

        for name, method in meta_methods.items():
            bound_method = MethodType(method, meta_agent_instance)
            setattr(meta_agent_instance, name, bound_method)

    def add_attributes(
        meta_agent_instance: Any,
        agents: Iterable[Agent],
        meta_attributes: dict[str, Any] | None,
    ) -> None:
        meta_attributes = meta_attributes or {}

        mesa_reserved = {"unique_id", "model", "pos", "name", "random", "rng"}

        if assume_constituting_agent_attributes:
            for agent in agents:
                for name, value in agent.__dict__.items():
                    if (
                        not callable(value)
                        and name not in mesa_reserved
                        and not name.startswith("_")
                    ):
                        meta_attributes[name] = value

        for key, value in meta_attributes.items():
            setattr(meta_agent_instance, key, value)

    # Check existing meta-agents
    existing_meta_agents = []
    for agent in agents:
        if hasattr(agent, "meta_agents"):
            for meta_agent in agent.meta_agents:
                if (
                    meta_agent.__class__.__name__ == new_agent_class
                    and meta_agent not in existing_meta_agents
                ):
                    existing_meta_agents.append(meta_agent)

    if existing_meta_agents:
        meta_agent = sorted(existing_meta_agents, key=lambda x: x.unique_id)[0]
        add_attributes(meta_agent, agents, meta_attributes)
        add_methods(meta_agent, agents, meta_methods)
        meta_agent.add_constituting_agents(agents)
        return meta_agent

    # Try existing class
    agent_class = extract_class(model.agents_by_type, new_agent_class)

    if agent_class:
        instance = agent_class(model, agents)
    else:
        meta_agent_class = type(
            new_agent_class,
            (MetaAgent, *mesa_agent_type),
            {"unique_id": None, "_constituting_set": None},
        )
        instance = meta_agent_class(model, agents)

    add_attributes(instance, agents, meta_attributes)
    add_methods(instance, agents, meta_methods)

    return instance


class MetaAgent(Agent):
    """An agent composed of other agents."""

    def __init__(
        self,
        model: Any,
        agents: Iterable[Agent] | None = None,
        name: str = "MetaAgent",
    ):
        super().__init__(model)
        self._constituting_set = AgentSet(agents or [], random=model.random)
        self.name = name

        for agent in self._constituting_set:
            self._link_agent(agent)

    def _link_agent(self, agent: Agent) -> None:
        """Link agent to this meta-agent."""
        if not hasattr(agent, "meta_agents"):
            agent.meta_agents = set()
        agent.meta_agents.add(self)
        agent.meta_agent = self

    def __len__(self) -> int:
        return len(self._constituting_set)

    def __iter__(self) -> Iterator[Agent]:
        return iter(self._constituting_set)

    def __contains__(self, agent: Agent) -> bool:
        return agent in self._constituting_set

    @property
    def agents(self) -> AgentSet:
        return self._constituting_set

    @property
    def constituting_agents_by_type(self) -> dict[type, list[Agent]]:
        result: dict[type, list[Agent]] = {}
        for agent in self._constituting_set:
            result.setdefault(type(agent), []).append(agent)
        return result

    @property
    def constituting_agent_types(self) -> set[type]:
        return {type(agent) for agent in self._constituting_set}

    def get_constituting_agent_instance(self, agent_type: type[Agent]) -> Agent:
        try:
            return self.constituting_agents_by_type[agent_type][0]
        except KeyError:
            raise ValueError(f"No constituting_agent of type {agent_type} found.") from None

    def add_constituting_agents(self, new_agents: Iterable[Agent]) -> None:
        for agent in new_agents:
            self._constituting_set.add(agent)
            self._link_agent(agent)

    def remove_constituting_agents(self, remove_agents: Iterable[Agent]) -> None:
        for agent in remove_agents:
            self._constituting_set.discard(agent)
            if hasattr(agent, "meta_agents"):
                agent.meta_agents.discard(self)
                if agent.meta_agents:
                    agent.meta_agent = sorted(
                        agent.meta_agents, key=lambda x: x.unique_id or 0
                    )[0]
                else:
                    agent.meta_agent = None

    def step(self) -> None:
        """Override to define behavior."""
        pass