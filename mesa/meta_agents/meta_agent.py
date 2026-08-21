"""Internal meta-agent group object and discovery helpers.

Meta-agents are agents composed of other agents. Memberships are tracked by
the model's membership manager (``model.meta_agents``). This module holds the
internal group ``Agent`` subclass and combination-discovery helpers.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable
from types import MethodType
from typing import Any

from mesa.agent import Agent


def _unique_id_sort_key(agent: Agent) -> tuple[bool, str]:
    """Return a deterministic, type-stable key for ordering agents by ID."""
    unique_id = getattr(agent, "unique_id", None)
    return (unique_id is not None, "" if unique_id is None else str(unique_id))


def _deduplicate_preserving_order(agents: Iterable[Any]) -> list[Any]:
    """Return unique agents while preserving caller order."""
    return list(dict.fromkeys(agents))


def _normalize_agent_bases(
    mesa_agent_type: type[Agent] | tuple[type[Agent], ...] | None,
) -> tuple[type[Agent], ...]:
    """Normalize user-provided Mesa base classes for dynamic meta-agent classes."""
    if mesa_agent_type is None:
        return (Agent,)
    if isinstance(mesa_agent_type, tuple):
        return mesa_agent_type
    return (mesa_agent_type,)


def evaluate_combination(
    candidate_group: tuple[Agent, ...],
    model,
    evaluation_func: Callable[[tuple[Agent, ...]], float] | None,
) -> tuple[tuple[Agent, ...], float] | None:
    """Evaluate a candidate meta-agent group with a user-supplied function."""
    if evaluation_func is None:
        return None
    return candidate_group, evaluation_func(candidate_group)


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
    """Find candidate agent groups and score them with ``evaluation_func``.

    The helper discovers potential meta-agents before creating them. It
    deliberately does not mutate model or membership state.

    Args:
        model: The model instance.
        group: The set of agents to find combinations in.
        size: The size or range of sizes for combinations. Defaults to (2, 5).
        evaluation_func: The function to evaluate combinations. Defaults to None.
        filter_func: Allows the user to specify how agents are filtered to form groups.
          Defaults to None.

    Returns:
        List: The list of valuable combinations, in a tuple first agentset of valuable combination  and then the value of
        the combination.
    """
    if isinstance(size, int):
        size_range = range(size, size + 1)
    else:
        min_size, max_size = size
        size_range = range(min_size, max_size + 1)

    combinations = []
    for candidate_group in itertools.chain.from_iterable(
        itertools.combinations(group, combination_size)
        for combination_size in size_range
    ):
        evaluation_result = evaluate_combination(
            candidate_group, model, evaluation_func
        )
        if evaluation_result is not None:
            _evaluated_group, result = evaluation_result
            if result is not None:
                combinations.append(evaluation_result)

    if combinations and filter_func is not None:
        return filter_func(combinations)
    return combinations


def extract_class(agents_by_type: dict, new_agent_class: object) -> type[Agent] | None:
    """Return the existing model agent class named ``new_agent_class`` if present."""
    agent_type_names = {
        agent_type.__name__: agent_type for agent_type in agents_by_type
    }
    agent_type = agent_type_names.get(new_agent_class)
    if agent_type is None:
        return None
    return type(next(iter(agents_by_type[agent_type])))


def _apply_meta_attributes(
    meta_agent: Any,
    meta_attributes: dict[str, Any] | None,
) -> None:
    """Set resolved meta-agent attributes on an instance."""
    for key, value in (meta_attributes or {}).items():
        setattr(meta_agent, key, value)


def _apply_meta_methods(
    meta_agent: Any,
    meta_methods: dict[str, Callable] | None,
) -> None:
    """Bind resolved meta-agent methods to an instance."""
    for name, method in (meta_methods or {}).items():
        setattr(meta_agent, name, MethodType(method, meta_agent))


def _find_existing_meta_agent(
    agents: Iterable[Any],
    new_agent_class: str,
    membership_api: Any,
) -> Any | None:
    """Find a compatible existing meta-agent among an agent's current groups."""
    existing_meta_agents = []
    for agent in agents:
        for meta_agent in sorted(
            membership_api.groups_of(agent), key=_unique_id_sort_key
        ):
            if (
                meta_agent.__class__.__name__ == new_agent_class
                and meta_agent not in existing_meta_agents
            ):
                existing_meta_agents.append(meta_agent)

    if not existing_meta_agents:
        return None
    return sorted(existing_meta_agents, key=_unique_id_sort_key)[0]


def _build_meta_agent_class(
    new_agent_class: str,
    mesa_agent_type: tuple[type[Agent], ...],
) -> type[Agent]:
    """Create a dynamic meta-agent class with the requested Mesa base types."""
    return type(
        new_agent_class,
        (MetaAgent, *mesa_agent_type),
        {
            "unique_id": None,
        },
    )


def _create_meta_agent_instance(
    model: Any,
    new_agent_class: str,
    agents: Iterable[Any],
    mesa_agent_type: type[Agent] | tuple[type[Agent], ...] | None,
    meta_attributes: dict[str, Any] | None = None,
    meta_methods: dict[str, Callable] | None = None,
    _membership_api: Any | None = None,
) -> Any:
    """Create or reuse a meta-agent instance without recording backend edges."""
    if _membership_api is None:
        raise RuntimeError("Use model.meta_agents.create() to create a meta-agent")

    agents = _deduplicate_preserving_order(agents)
    agent_bases = _normalize_agent_bases(mesa_agent_type)

    meta_agent = _find_existing_meta_agent(agents, new_agent_class, _membership_api)
    if meta_agent is not None:
        existing_api = getattr(meta_agent, "_membership_api", None)
        if existing_api not in (None, _membership_api):
            raise RuntimeError("Meta-agent is bound to a different membership manager")
        meta_agent._membership_api = _membership_api
        _apply_meta_attributes(meta_agent, meta_attributes)
        _apply_meta_methods(meta_agent, meta_methods)
        return meta_agent

    agent_class = extract_class(model.agents_by_type, new_agent_class)
    if agent_class is None:
        agent_class = _build_meta_agent_class(new_agent_class, agent_bases)

    meta_agent = agent_class(
        model,
        name=new_agent_class,
        initial_attributes=meta_attributes,
        _membership_api=_membership_api,
    )
    _apply_meta_attributes(meta_agent, meta_attributes)
    _apply_meta_methods(meta_agent, meta_methods)
    return meta_agent


class MetaAgent(Agent):
    """Internal agent subclass used as a live group object.

    Construct instances through ``model.meta_agents.create``. Membership
    reads and writes go through the membership manager, not this class.
    """

    def __init__(
        self,
        model,
        name: str = "MetaAgent",
        initial_attributes: dict[str, Any] | None = None,
        _membership_api: Any | None = None,
    ):
        """Create a meta-agent group bound to the model's membership manager."""
        if _membership_api is None:
            raise RuntimeError("Use model.meta_agents.create() to create a meta-agent")
        installed_api = getattr(model, "meta_agents", None)
        if installed_api is not _membership_api:
            raise RuntimeError(
                "Meta-agent must be created by its model's membership manager"
            )
        if initial_attributes:
            for key, value in initial_attributes.items():
                object.__setattr__(self, key, value)

        super().__init__(model)
        self._membership_api = _membership_api
        self.name = name

    def _remove_from_model(self) -> None:
        """Deregister this group without going through dissolve again."""
        super().remove()

    def remove(self) -> None:
        """Dissolve this group through the membership manager."""
        self._membership_api.dissolve(self)

    def step(self) -> None:
        """Default meta-agent behavior."""


__all__ = [
    "evaluate_combination",
    "extract_class",
    "find_combinations",
]
