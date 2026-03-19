"""Implementation of Mesa's meta agent capability."""

import itertools
from collections.abc import Callable, Iterable
from types import MethodType
from typing import Any

from mesa.agent import Agent, AgentSet


def evaluate_combination(
    candidate_group: tuple[Agent, ...],
    model,
    evaluation_func: Callable[[tuple[Agent, ...]], float] | None,
) -> tuple[tuple[Agent, ...], float] | None:
    """Evaluate a combination of agents."""
    if evaluation_func:
        value = evaluation_func(candidate_group)
        return candidate_group, value
    return None


def find_combinations(
    model,
    group: Iterable,
    size: int | tuple[int, int] = (2, 5),
    evaluation_func: Callable[[tuple[Agent, ...]], float] | None = None,
    filter_func: Callable[
        [list[tuple[tuple[Agent, ...], float]]], list[tuple[tuple[Agent, ...], float]]
    ]
    | None = None,
) -> list[tuple[tuple[Agent, ...], float]]:
    """Find valuable combinations of agents in this set."""
    combinations = []
    size_range = (size, size + 1) if isinstance(size, int) else size

    for candidate_group in itertools.chain.from_iterable(
        itertools.combinations(group, size) for size in range(*size_range)
    ):
        evaluation_result = evaluate_combination(
            candidate_group, model, evaluation_func
        )
        if evaluation_result is not None:
            evaluated_group, result = evaluation_result
            if result is not None:
                combinations.append((evaluated_group, result))

    if len(combinations) > 0 and filter_func:
        filtered_combinations = filter_func(combinations)
        return filtered_combinations

    return combinations


def extract_class(agents_by_type: dict, new_agent_class: object) -> type[Agent] | None:
    """Helper function for create_meta_agents extracts the types of agents."""
    agent_type_names = {}
    for agent in agents_by_type:
        agent_type_names[agent.__name__] = agent

    if new_agent_class in agent_type_names:
        return type(next(iter(agents_by_type[agent_type_names[new_agent_class]])))
    return None


def create_meta_agent(
    model: Any,
    new_agent_class: str,
    agents: Iterable[Any],
    mesa_agent_type: type[Agent] | None,
    meta_attributes: dict[str, Any] | None = None,
    meta_methods: dict[str, Callable] | None = None,
    assume_constituting_agent_methods: bool = False,
    assume_constituting_agent_attributes: bool = False,
) -> Any | None:
    """Create a new meta-agent class and instantiate agents."""
    agents = list(dict.fromkeys(agents).keys())

    if not mesa_agent_type:
        mesa_agent_type = (Agent,)
    elif not isinstance(mesa_agent_type, tuple):
        mesa_agent_type = (mesa_agent_type,)

    def add_methods(
        meta_agent_instance: Any,
        agents: Iterable[Any],
        meta_methods: dict[str, Callable],
    ) -> None:
        if assume_constituting_agent_methods:
            agent_classes = {type(agent) for agent in agents}
            if meta_methods is None:
                meta_methods = {}
            for agent_class in agent_classes:
                for name in agent_class.__dict__:
                    if callable(getattr(agent_class, name)) and not name.startswith("__"):
                        original_method = getattr(agent_class, name)
                        meta_methods[name] = original_method

        if meta_methods is not None:
            for name, meth in meta_methods.items():
                bound_method = MethodType(meth, meta_agent_instance)
                setattr(meta_agent_instance, name, bound_method)

    def add_attributes(
        meta_agent_instance: Any, agents: Iterable[Any], meta_attributes: dict[str, Any]
    ) -> None:
        mesa_primitives = ["unique_id", "model", "pos", "name", "random", "rng"]
        if assume_constituting_agent_attributes:
            if meta_attributes is None:
                meta_attributes = {}
            for agent in agents:
                for name, value in agent.__dict__.items():
                    if (
                        not callable(value)
                        and name not in mesa_primitives
                        and not name.startswith("_")
                    ):
                        meta_attributes[name] = value

        if meta_attributes is not None:
            for key, value in meta_attributes.items():
                setattr(meta_agent_instance, key, value)

    existing_meta_agents = []
    for a in agents:
        if hasattr(a, "meta_agents"):
            for ma in a.meta_agents:
                if (
                    ma.__class__.__name__ == new_agent_class
                    and ma not in existing_meta_agents
                ):
                    existing_meta_agents.append(ma)

    if len(existing_meta_agents) > 0:
        # Optimized O(n) selection and string conversion to prevent TypeError
        meta_agent = min(existing_meta_agents, key=lambda x: str(x.unique_id))
        add_attributes(meta_agent, agents, meta_attributes)
        add_methods(meta_agent, agents, meta_methods)
        meta_agent.add_constituting_agents(agents)
        return meta_agent
    else:
        agent_class = extract_class(model.agents_by_type, new_agent_class)
        if agent_class:
            meta_agent_instance = agent_class(model, agents)
            add_attributes(meta_agent_instance, agents, meta_attributes)
            add_methods(meta_agent_instance, agents, meta_methods)
            return meta_agent_instance
        else:
            meta_agent_class = type(
                new_agent_class,
                (MetaAgent, *mesa_agent_type),
                {"unique_id": None, "_constituting_set": None},
            )
            meta_agent_instance = meta_agent_class(model, agents)
            add_attributes(meta_agent_instance, agents, meta_attributes)
            add_methods(meta_agent_instance, agents, meta_methods)
            return meta_agent_instance


class MetaAgent(Agent):
    """A MetaAgent is an agent that contains other agents as components."""

    def __init__(self, model, agents: set[Agent] | None = None, name: str = "MetaAgent"):
        super().__init__(model)
        self._constituting_set = AgentSet(agents or [], random=model.random)
        self.name = name

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
        constituting_agents_by_type = {}
        for agent in self._constituting_set:
            agent_type = type(agent)
            if agent_type not in constituting_agents_by_type:
                constituting_agents_by_type[agent_type] = []
            constituting_agents_by_type[agent_type].append(agent)
        return constituting_agents_by_type

    @property
    def constituting_agent_types(self) -> set[type]:
        """Get the types of all constituting_agents."""
        return {type(agent) for agent in self._constituting_set}

    def get_constituting_agent_instance(self, agent_type: type[Agent]) -> Agent:
        """Get the instance of a constituting_agent of the specified type."""
        try:
            return self.constituting_agents_by_type[agent_type][0]
        except (KeyError, IndexError):
            raise ValueError(
                f"No constituting_agent of type {agent_type} found."
            ) from None

    def add_constituting_agents(self, new_agents: set[Agent]):
        for agent in new_agents:
            self._constituting_set.add(agent)
            if not hasattr(agent, "meta_agents"):
                agent.meta_agents = set()
            agent.meta_agents.add(self)
            agent.meta_agent = self

    def remove_constituting_agents(self, remove_agents: set[Agent]):
        """Remove agents as components."""
        for agent in remove_agents:
            self._constituting_set.discard(agent)
            if hasattr(agent, "meta_agents"):
                agent.meta_agents.discard(self)
                # Update backward compatibility attribute deterministically
                if len(agent.meta_agents) > 0:
                    # Optimized O(n) selection for consistency and to prevent TypeError
                    agent.meta_agent = min(agent.meta_agents, key=lambda x: str(x.unique_id))
                else:
                    agent.meta_agent = None

    def step(self):
        """Perform the agent's step."""
        pass