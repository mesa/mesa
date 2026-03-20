"""Implementation of Mesa's meta agent capability.

Overview: Complex systems often have multiple levels of components. An
organization is not one entity, but is made of departments, sub-departments,
and people. A person is not a single entity, but it is made of micro biomes,
organs and cells. A city is not a single entity, but it is made of districts,
neighborhoods, buildings, and people. A forest comprises an ecosystem of
trees, plants, animals, and microorganisms.

This reality is the motivation for meta-agents. It allows users to represent
these multiple levels, where each level can have agents with constituting_agents.

To demonstrate meta-agents capability there are two examples:
1 - Alliance formation which shows emergent meta-agent formation in
advanced examples:
https://github.com/mesa/mesa/tree/main/mesa/examples/advanced/alliance_formation
2 - Warehouse model in the Mesa example's repository
https://github.com/mesa/mesa-examples/tree/main/examples/warehouse

To accomplish this the MetaAgent module is as follows:

This contains four  helper functions and a MetaAgent class that can be used to
create agents that contain other agents as components.

Helper methods:
1 - find_combinations: Find combinations of agents to create a meta-agent
constituting_set.
2- evaluate_combination: Evaluate combinations of agents by some user based
criteria to determine if it should be a constituting_set of agents.
3- extract_class: Helper function for create_meta-agent. Extracts the types of
agent being created to create a new instance of that agent type.
4- create_meta_agent: Create a new meta-agent class and instantiate
agents in that class.

Meta-Agent class (MetaAgent): An agent that contains other agents
as components.

.
"""



import itertools
from collections import defaultdict
from collections.abc import Callable, Iterable
from types import MethodType
from typing import Any, Type

from mesa.agent import Agent, AgentSet


def evaluate_combination(
    candidate_group: tuple[Agent, ...],
    model: Any,
    evaluation_func: Callable[[tuple[Agent, ...]], float] | None,
) -> tuple[tuple[Agent, ...], float] | None:
    """Evaluate a combination of agents using a user-provided function."""
    if evaluation_func:
        value = evaluation_func(candidate_group)
        return candidate_group, value
    return None


def find_combinations(
    model: Any,
    group: Iterable[Agent],
    size: int | tuple[int, int] = (2, 5),
    evaluation_func: Callable[[tuple[Agent, ...]], float] | None = None,
    filter_func: Callable[[list[tuple[tuple[Agent, ...], float]]], list[tuple[tuple[Agent, ...], float]]] | None = None,
) -> list[tuple[tuple[Agent, ...], float]]:
    """Find valuable combinations of agents in a group.

    Args:
        model: The model instance.
        group: The set of agents to combine.
        size: Size or range of combination sizes (default (2,5)).
        evaluation_func: Function to evaluate combinations.
        filter_func: Optional function to filter resulting combinations.

    Returns:
        List of tuples: ((agents...), value)
    """
    combinations = []
    size_range = (size, size + 1) if isinstance(size, int) else size

    for candidate_group in itertools.chain.from_iterable(
        itertools.combinations(group, s) for s in range(*size_range)
    ):
        evaluation_result = evaluate_combination(candidate_group, model, evaluation_func)
        if evaluation_result is not None:
            evaluated_group, result = evaluation_result
            if result is not None:
                combinations.append((evaluated_group, result))

    if combinations and filter_func:
        return filter_func(combinations)

    return combinations


def extract_class(agents_by_type: dict, new_agent_class: str) -> Type[Agent] | None:
    """Extract an existing agent type from a dict of agents by type."""
    agent_type_names = {agent.__name__: agent for agent in agents_by_type}
    if new_agent_class in agent_type_names:
        agent_list = agents_by_type[agent_type_names[new_agent_class]]
        if agent_list:
            return type(next(iter(agent_list)))
    return None


def _add_methods_to_meta(
    meta_agent_instance: Agent,
    agents: Iterable[Agent],
    meta_methods: dict[str, Callable] | None,
    assume_methods: bool,
) -> None:
    """Bind methods from agents or meta_methods to a meta-agent instance."""
    if assume_methods:
        agent_classes = {type(agent) for agent in agents}
        if meta_methods is None:
            meta_methods = {}
        for cls in agent_classes:
            for name, meth in cls.__dict__.items():
                if callable(meth) and not name.startswith("__"):
                    meta_methods[name] = meth
    if meta_methods:
        for name, meth in meta_methods.items():
            setattr(meta_agent_instance, name, MethodType(meth, meta_agent_instance))


def _add_attributes_to_meta(
    meta_agent_instance: Agent,
    agents: Iterable[Agent],
    meta_attributes: dict[str, Any] | None,
    assume_attributes: bool,
) -> None:
    """Add attributes from agents or meta_attributes to a meta-agent instance."""
    reserved = {"unique_id", "model", "pos", "name", "random", "rng"}

    if assume_attributes:
        if meta_attributes is None:
            meta_attributes = {}
        for agent in agents:
            for name, value in agent.__dict__.items():
                if not callable(value) and name not in reserved and not name.startswith("_"):
                    meta_attributes[name] = value

    if meta_attributes:
        for key, value in meta_attributes.items():
            setattr(meta_agent_instance, key, value)


def create_meta_agent(
    model: Any,
    new_agent_class: str,
    agents: Iterable[Agent],
    mesa_agent_type: Type[Agent] | None = None,
    meta_attributes: dict[str, Any] | None = None,
    meta_methods: dict[str, Callable] | None = None,
    assume_constituting_agent_methods: bool = False,
    assume_constituting_agent_attributes: bool = False,
) -> Agent | None:
    """Create or retrieve a meta-agent from a group of agents."""
    agents = list(dict.fromkeys(agents))  # Ensure uniqueness while preserving order

    # Ensure mesa_agent_type is a tuple
    if not mesa_agent_type:
        mesa_agent_type = (Agent,)
    elif not isinstance(mesa_agent_type, tuple):
        mesa_agent_type = (mesa_agent_type,)

    # Path 1: Attach to existing meta-agent
    existing_meta_agents = []
    for a in agents:
        if hasattr(a, "meta_agents"):
            for ma in a.meta_agents:
                if ma.__class__.__name__ == new_agent_class and ma not in existing_meta_agents:
                    existing_meta_agents.append(ma)

    if existing_meta_agents:
        meta_agent = sorted(existing_meta_agents, key=lambda x: x.unique_id)[0]
        _add_attributes_to_meta(meta_agent, agents, meta_attributes, assume_constituting_agent_attributes)
        _add_methods_to_meta(meta_agent, agents, meta_methods, assume_constituting_agent_methods)
        meta_agent.add_constituting_agents(set(agents))
        return meta_agent

    # Path 2: Use existing agent class
    agent_class = extract_class(model.agents_by_type, new_agent_class)
    if agent_class:
        meta_agent_instance = agent_class(model, agents)
        _add_attributes_to_meta(meta_agent_instance, agents, meta_attributes, assume_constituting_agent_attributes)
        _add_methods_to_meta(meta_agent_instance, agents, meta_methods, assume_constituting_agent_methods)
        return meta_agent_instance

    # Path 3: Create new meta-agent class
    meta_agent_class = type(
        new_agent_class,
        (MetaAgent, *mesa_agent_type),
        {"unique_id": None, "_constituting_set": None},
    )
    meta_agent_instance = meta_agent_class(model, agents)
    _add_attributes_to_meta(meta_agent_instance, agents, meta_attributes, assume_constituting_agent_attributes)
    _add_methods_to_meta(meta_agent_instance, agents, meta_methods, assume_constituting_agent_methods)
    return meta_agent_instance


class MetaAgent(Agent):
    """A MetaAgent contains other agents as components."""

    def __init__(self, model: Any, agents: set[Agent] | None = None, name: str = "MetaAgent"):
        super().__init__(model)
        self._constituting_set = AgentSet(agents or [], random=model.random)
        self.name = name

        # Reference back to meta_agent
        for agent in self._constituting_set:
            if not hasattr(agent, "meta_agents"):
                agent.meta_agents = set()
            agent.meta_agents.add(self)
            agent.meta_agent = self

    def __len__(self) -> int:
        return len(self._constituting_set)

    def __iter__(self):
        return iter(self._constituting_set)

    def __contains__(self, agent: Agent) -> bool:
        return agent in self._constituting_set

    @property
    def agents(self) -> AgentSet:
        return self._constituting_set

    @property
    def constituting_agents_by_type(self) -> dict[type, list[Agent]]:
        by_type = defaultdict(list)
        for agent in self._constituting_set:
            by_type[type(agent)].append(agent)
        return dict(by_type)

    @property
    def constituting_agent_types(self) -> set[type]:
        return {type(agent) for agent in self._constituting_set}

    def get_constituting_agent_instance(self, agent_type: type) -> Agent:
        agents = self.constituting_agents_by_type.get(agent_type)
        if not agents:
            raise ValueError(f"No constituting agent of type {agent_type} found.")
        return agents[0]

    def add_constituting_agents(self, new_agents: set[Agent]) -> None:
        for agent in new_agents:
            self._constituting_set.add(agent)
            if not hasattr(agent, "meta_agents"):
                agent.meta_agents = set()
            agent.meta_agents.add(self)
            agent.meta_agent = self

    def remove_constituting_agents(self, remove_agents: set[Agent]) -> None:
        for agent in remove_agents:
            self._constituting_set.discard(agent)
            if hasattr(agent, "meta_agents"):
                agent.meta_agents.discard(self)
                agent.meta_agent = sorted(agent.meta_agents, key=lambda x: x.unique_id or 0)[0] if agent.meta_agents else None

    def step(self) -> None:
        """Define MetaAgent behavior here. By default, does nothing."""
        pass