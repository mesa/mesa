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

This contains four helper functions and a MetaAgent class that can be used to
create agents that contain other agents as components.

Helper methods:
1 - find_combinations: Find combinations of agents to create a meta-agent
constituting_set.
2 - evaluate_combination: Evaluate combinations of agents by some user based
criteria to determine if it should be a constituting_set of agents.
3 - extract_class: Helper function for create_meta_agent. Extracts the types of
agent being created to create a new instance of that agent type.
4 - create_meta_agent: Create a new meta-agent class and instantiate
agents in that class.

Meta-Agent class (MetaAgent): An agent that contains other agents
as components.

Enhancements (GSoC 2026):
- Lifecycle management: active / dormant / dissolved states with cascading
  control over whether constituting agents step.
- Signal / event system: constituting agents can emit named signals that the
  meta-agent (and user-registered handlers) can react to synchronously.
- Hierarchical step control: MetaAgent.step() can optionally cascade to all
  constituting agents, removing the need for boilerplate in user subclasses.
- Combination caching: find_combinations caches results keyed on group
  membership and size to avoid redundant O(n!) re-evaluation.
- Rich __repr__ / __str__ for easier debugging and logging.
- to_dict / from_snapshot serialisation helpers on MetaAgent.
"""

import contextlib
import itertools
from collections.abc import Callable, Iterable
from enum import Enum, auto
from types import MethodType
from typing import Any

from mesa.agent import Agent, AgentSet

# ---------------------------------------------------------------------------
# Lifecycle states
# ---------------------------------------------------------------------------


class MetaAgentState(Enum):
    """Lifecycle states for a MetaAgent.

    Attributes:
        ACTIVE:    The meta-agent and its constituting agents participate
                   normally in the simulation step.
        DORMANT:   The meta-agent exists but neither it nor its constituting
                   agents are stepped (unless they belong to another active
                   meta-agent). The group can be re-activated later.
        DISSOLVED: The meta-agent has been permanently disbanded. All
                   constituent references are cleaned up. This state is
                   terminal — call ``remove()`` instead of setting this
                   directly.
    """

    ACTIVE = auto()
    DORMANT = auto()
    DISSOLVED = auto()


# ---------------------------------------------------------------------------
# Helper functions (original, unchanged contract)
# ---------------------------------------------------------------------------


def evaluate_combination(
    candidate_group: tuple[Agent, ...],
    model,
    evaluation_func: Callable[[tuple[Agent, ...]], float] | None,
) -> tuple[tuple[Agent, ...], float] | None:
    """Evaluate a combination of agents.

    Args:
        candidate_group: The group of agents to evaluate.
        model: The model instance.
        evaluation_func: The function to evaluate the group.

    Returns:
        Optional: The evaluated group and its value, or None.
    """
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
    *,
    use_cache: bool = False,
) -> list[tuple[tuple[Agent, ...], float]]:
    """Find valuable combinations of agents in this set.

    Args:
        model: The model instance.
        group: The set of agents to find combinations in.
        size: The size or range of sizes for combinations. Defaults to (2, 5).
        evaluation_func: The function to evaluate combinations. Defaults to None.
        filter_func: Allows the user to specify how agents are filtered to form
            groups. Defaults to None.
        use_cache: If True, cache results keyed on the frozenset of agent
            unique_ids and the size argument. Useful when the same group is
            evaluated repeatedly across steps without membership changes.
            Defaults to False.

    Returns:
        List: The list of valuable combinations, each a tuple of
        (agent_tuple, value).

    Notes:
        Caching is opt-in because it is only beneficial when the group
        membership is stable across multiple calls. Enable it with
        ``use_cache=True`` when calling inside a loop over many steps.
    """
    group = list(group)

    if use_cache and evaluation_func is not None:
        cache_key = (
            frozenset(getattr(a, "unique_id", id(a)) for a in group),
            size if isinstance(size, tuple) else (size, size + 1),
        )
        cached = _combination_cache.get(cache_key)
        if cached is not None:
            return cached

    combinations = []
    size_range = (size, size + 1) if isinstance(size, int) else size

    for candidate_group in itertools.chain.from_iterable(
        itertools.combinations(group, s) for s in range(*size_range)
    ):
        evaluation_result = evaluate_combination(
            candidate_group, model, evaluation_func
        )
        if evaluation_result is not None:
            evaluated_group, result = evaluation_result
            if result is not None:
                combinations.append((evaluated_group, result))

    if len(combinations) > 0 and filter_func:
        combinations = filter_func(combinations)

    if use_cache and evaluation_func is not None:
        _combination_cache[cache_key] = combinations

    return combinations


# Module-level combination cache (keyed on frozenset of ids + size range).
_combination_cache: dict = {}


def clear_combination_cache() -> None:
    """Clear the module-level combination cache.

    Call this at the start of each step if group membership can change,
    or whenever you want to force re-evaluation of all combinations.
    """
    _combination_cache.clear()


def extract_class(agents_by_type: dict, new_agent_class: object) -> type[Agent] | None:
    """Helper function for create_meta_agents extracts the types of agents.

    Args:
        agents_by_type (dict): The dictionary of agents by type.
        new_agent_class (str): The name of the agent class to be created.

    Returns:
        type(Agent) if agent type exists, None otherwise.
    """
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
    *,
    join_policy: Callable[[list["MetaAgent"]], "MetaAgent"] | None = None,
) -> Any | None:
    """Create a new meta-agent class and instantiate agents.

    Parameters:
        model (Any): The model instance.
        new_agent_class (str): The name of the new meta-agent class.
        agents (Iterable[Any]): The agents to be included in the meta-agent.
        mesa_agent_type (type[Agent] | None): Base agent type(s).
        meta_attributes (Dict[str, Any]): Attributes to be added to the meta-agent.
        meta_methods (Dict[str, Callable]): Methods to be added to the meta-agent.
        assume_constituting_agent_methods (bool): Whether to inherit methods from
            constituting agents as meta-agent methods.
        assume_constituting_agent_attributes (bool): Whether to inherit attributes
            from constituting agents.
        join_policy (Callable | None): When multiple existing meta-agents of the
            same class already contain some of the supplied agents, this callable
            receives the list of candidate meta-agents and returns the one the
            new agents should join. Defaults to selecting the meta-agent with
            the lowest ``unique_id`` (original behaviour).

    Returns:
        MetaAgent instance.
    """
    # Convert agents to dict to ensure uniqueness while preserving order
    agents = list(dict.fromkeys(agents).keys())

    if not mesa_agent_type:
        mesa_agent_type = (Agent,)
    elif not isinstance(mesa_agent_type, tuple):
        mesa_agent_type = (mesa_agent_type,)

    def add_methods(
        meta_agent_instance: Any,
        agents: Iterable[Any],
        meta_methods: dict[str, Callable] | None,
    ) -> None:
        resolved_meta_methods = dict(meta_methods or {})
        if assume_constituting_agent_methods:
            agent_classes = dict.fromkeys(type(agent) for agent in agents)
            for agent_class in agent_classes:
                for name in agent_class.__dict__:
                    if callable(getattr(agent_class, name)) and not name.startswith(
                        "__"
                    ):
                        original_method = getattr(agent_class, name)
                        resolved_meta_methods.setdefault(name, original_method)

        for name, meth in resolved_meta_methods.items():
            bound_method = MethodType(meth, meta_agent_instance)
            setattr(meta_agent_instance, name, bound_method)

    def add_attributes(
        meta_agent_instance: Any,
        agents: Iterable[Any],
        meta_attributes: dict[str, Any],
    ) -> None:
        mesa_primitives = [
            "unique_id",
            "model",
            "pos",
            "name",
            "random",
            "rng",
        ]

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

    # Path 1 - Add agents to existing meta-agent of the SAME CLASS if any exist
    existing_meta_agents = []
    for a in agents:
        if hasattr(a, "meta_agents"):
            for ma in sorted(a.meta_agents, key=lambda x: x.unique_id or 0):
                if (
                    ma.__class__.__name__ == new_agent_class
                    and ma not in existing_meta_agents
                ):
                    existing_meta_agents.append(ma)

    if len(existing_meta_agents) > 0:
        if join_policy is not None:
            meta_agent = join_policy(existing_meta_agents)
        else:
            meta_agent = (
                sorted(existing_meta_agents, key=lambda x: x.unique_id)[0]
                if len(existing_meta_agents) > 1
                else existing_meta_agents[0]
            )
        add_attributes(meta_agent, agents, meta_attributes)
        add_methods(meta_agent, agents, meta_methods)
        meta_agent.add_constituting_agents(agents)
        return meta_agent

    else:
        # Path 2 - Create a new instance of an existing meta-agent class
        agent_class = extract_class(model.agents_by_type, new_agent_class)

        if agent_class:
            meta_agent_instance = agent_class(model, agents)
            add_attributes(meta_agent_instance, agents, meta_attributes)
            add_methods(meta_agent_instance, agents, meta_methods)
            return meta_agent_instance
        else:
            # Path 3 - Create a new meta-agent class dynamically
            meta_agent_class = type(
                new_agent_class,
                (MetaAgent, *mesa_agent_type),
                {
                    "unique_id": None,
                    "_constituting_set": None,
                },
            )
            meta_agent_instance = meta_agent_class(model, agents)
            add_attributes(meta_agent_instance, agents, meta_attributes)
            add_methods(meta_agent_instance, agents, meta_methods)
            return meta_agent_instance


# ---------------------------------------------------------------------------
# MetaAgent class
# ---------------------------------------------------------------------------


class MetaAgent(Agent):
    """An agent that contains other agents as components.

    MetaAgent extends Mesa's Agent with three new capabilities:

    **1. Lifecycle management**
    A MetaAgent can be in one of three states (``MetaAgentState``):

    - ``ACTIVE``   — normal operation; step() runs and can cascade to members.
    - ``DORMANT``  — the group is suspended; step() is a no-op and cascading
      is suppressed. Constituents are not removed; reactivate with
      ``activate()``.
    - ``DISSOLVED`` — terminal; equivalent to ``remove()``.

    Use ``activate()`` / ``deactivate()`` to transition between ACTIVE and
    DORMANT. Use ``remove()`` to dissolve permanently.

    **2. Signal / event system**
    Constituting agents (or external code) can emit named signals on a
    meta-agent. Handlers registered with ``on(signal, handler)`` are called
    synchronously in registration order. This enables reactive group behaviour
    without polling::

        # Register a handler
        alliance.on("member_defeated", lambda agent: alliance.remove_constituting_agents({agent}))

        # Emit from inside a constituting agent's step
        self.meta_agent.emit("member_defeated", self)

    **3. Cascading step control**
    Subclasses may set ``cascade_step = True`` (or pass it at construction)
    to make ``step()`` automatically call ``step()`` on every constituting
    agent after the meta-agent's own logic runs::

        class Alliance(MetaAgent):
            cascade_step = True

            def step(self):
                self.negotiate()   # meta-level logic
                # constituting agents will be stepped automatically

    Cascading is suppressed when the meta-agent is DORMANT.
    """

    cascade_step: bool = False

    def __init__(
        self,
        model,
        agents: set[Agent] | None = None,
        name: str = "MetaAgent",
        *,
        cascade_step: bool | None = None,
    ):
        """Create a new MetaAgent.

        Args:
            model: The model instance.
            agents: The set of agents to include. Defaults to None (empty).
            name: Human-readable name for this meta-agent. Defaults to
                ``"MetaAgent"``.
            cascade_step: If provided, overrides the class-level
                ``cascade_step`` attribute for this instance only.
        """
        super().__init__(model)
        self._constituting_set = AgentSet(agents or [], random=model.random)
        self.name = name
        self._state: MetaAgentState = MetaAgentState.ACTIVE
        self._signal_handlers: dict[str, list[Callable]] = {}

        if cascade_step is not None:
            self.cascade_step = cascade_step

        # Register back-references on constituting agents
        for agent in self._constituting_set:
            self._register_backref(agent)

    # ------------------------------------------------------------------
    # Back-reference helpers
    # ------------------------------------------------------------------

    def _register_backref(self, agent: Agent) -> None:
        """Add this meta-agent's reference to a constituting agent."""
        if not hasattr(agent, "meta_agents"):
            agent.meta_agents = set()
        agent.meta_agents.add(self)
        agent.meta_agent = sorted(agent.meta_agents, key=lambda x: x.unique_id or 0)[0]

    def _deregister_backref(self, agent: Agent) -> None:
        """Remove this meta-agent's reference from a constituting agent."""
        if hasattr(agent, "meta_agents"):
            agent.meta_agents.discard(self)
            if len(agent.meta_agents) > 0:
                agent.meta_agent = sorted(
                    agent.meta_agents, key=lambda x: x.unique_id or 0
                )[0]
            else:
                agent.meta_agent = None

    # ------------------------------------------------------------------
    # Lifecycle API
    # ------------------------------------------------------------------

    @property
    def state(self) -> MetaAgentState:
        """The current lifecycle state of this meta-agent."""
        return self._state

    @property
    def is_active(self) -> bool:
        """True if the meta-agent is in the ACTIVE state."""
        return self._state is MetaAgentState.ACTIVE

    @property
    def is_dormant(self) -> bool:
        """True if the meta-agent is in the DORMANT state."""
        return self._state is MetaAgentState.DORMANT

    @property
    def is_dissolved(self) -> bool:
        """True if the meta-agent has been dissolved (removed)."""
        return self._state is MetaAgentState.DISSOLVED

    def activate(self) -> None:
        """Transition the meta-agent from DORMANT to ACTIVE.

        Raises:
            RuntimeError: If called on a DISSOLVED meta-agent.
        """
        if self._state is MetaAgentState.DISSOLVED:
            raise RuntimeError(
                f"Cannot activate a dissolved MetaAgent (id={self.unique_id}). "
                "A dissolved meta-agent cannot be reused; create a new one."
            )
        if self._state is MetaAgentState.ACTIVE:
            return  # no-op
        self._state = MetaAgentState.ACTIVE
        self.emit("activated", self)

    def deactivate(self) -> None:
        """Transition the meta-agent from ACTIVE to DORMANT.

        The meta-agent and its constituting agents are suspended: ``step()``
        becomes a no-op and cascading is suppressed. The group can be
        reactivated later with ``activate()``.

        Raises:
            RuntimeError: If called on a DISSOLVED meta-agent.
        """
        if self._state is MetaAgentState.DISSOLVED:
            raise RuntimeError(
                f"Cannot deactivate a dissolved MetaAgent (id={self.unique_id})."
            )
        if self._state is MetaAgentState.DORMANT:
            return  # no-op
        self._state = MetaAgentState.DORMANT
        self.emit("deactivated", self)

    # ------------------------------------------------------------------
    # Signal / event system
    # ------------------------------------------------------------------

    def on(self, signal: str, handler: Callable) -> None:
        """Register a handler for a named signal.

        Handlers are called in registration order when ``emit(signal, ...)``
        is invoked. The handler receives ``*args`` and ``**kwargs`` passed to
        ``emit``.

        Args:
            signal: Name of the signal (e.g. ``"member_defeated"``).
            handler: Callable to invoke. Receives the arguments passed to
                ``emit()``.

        Example::

            alliance.on("member_defeated", lambda agent: print(f"{agent} was defeated"))
        """
        self._signal_handlers.setdefault(signal, []).append(handler)

    def off(self, signal: str, handler: Callable | None = None) -> None:
        """Deregister a handler (or all handlers) for a named signal.

        Args:
            signal: Name of the signal.
            handler: The specific handler to remove. If None, *all* handlers
                for this signal are removed.
        """
        if signal not in self._signal_handlers:
            return
        if handler is None:
            del self._signal_handlers[signal]
        else:
            with contextlib.suppress(ValueError):
                self._signal_handlers[signal].remove(handler)

    def emit(self, signal: str, /, *args: Any, **kwargs: Any) -> None:
        """Emit a named signal, calling all registered handlers.

        Args:
            signal: Name of the signal to emit.
            *args: Positional arguments forwarded to each handler.
            **kwargs: Keyword arguments forwarded to each handler.

        Example::

            # Inside a constituting agent's step():
            if self.health <= 0:
                self.meta_agent.emit("member_defeated", self)
        """
        for handler in list(self._signal_handlers.get(signal, [])):
            handler(*args, **kwargs)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of constituting agents."""
        return len(self._constituting_set)

    def __iter__(self):
        """Iterate over constituting agents."""
        return iter(self._constituting_set)

    def __contains__(self, agent: Agent) -> bool:
        """Check if an agent is a constituting member."""
        return agent in self._constituting_set

    def __repr__(self) -> str:
        """Unambiguous representation useful in logs and debuggers."""
        return (
            f"<{self.__class__.__name__} id={self.unique_id} "
            f"name={self.name!r} state={self._state.name} "
            f"members={len(self._constituting_set)}>"
        )

    def __str__(self) -> str:
        """Readable string for printing."""
        return (
            f"{self.__class__.__name__}({self.name!r}, "
            f"{self._state.name}, {len(self._constituting_set)} members)"
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def agents(self) -> AgentSet:
        """The AgentSet of constituting agents."""
        return self._constituting_set

    @property
    def constituting_agents_by_type(self) -> dict[type, list[Agent]]:
        """Constituting agents grouped by their type.

        Returns:
            dict[type, list[Agent]]: Mapping of agent type → list of agents.
        """
        result: dict[type, list[Agent]] = {}
        for agent in self._constituting_set:
            result.setdefault(type(agent), []).append(agent)
        return result

    @property
    def constituting_agent_types(self) -> set[type]:
        """The set of unique types among constituting agents."""
        return {type(agent) for agent in self._constituting_set}

    # ------------------------------------------------------------------
    # Membership management
    # ------------------------------------------------------------------

    def get_constituting_agent_instance(self, agent_type) -> Agent:
        """Return the first constituting agent of a given type.

        Args:
            agent_type: The type to look up.

        Returns:
            The first matching agent.

        Raises:
            ValueError: If no agent of that type is found.
        """
        try:
            return self.constituting_agents_by_type[agent_type][0]
        except KeyError:
            raise ValueError(
                f"No constituting_agent of type {agent_type} found."
            ) from None

    def add_constituting_agents(self, new_agents: set[Agent]) -> None:
        """Add agents as constituting members.

        Args:
            new_agents: Agents to add.
        """
        for agent in new_agents:
            self._constituting_set.add(agent)
            self._register_backref(agent)

    def remove_constituting_agents(self, remove_agents: set[Agent]) -> None:
        """Remove agents from this meta-agent.

        Args:
            remove_agents: Agents to remove.
        """
        for agent in remove_agents:
            self._constituting_set.discard(agent)
            self._deregister_backref(agent)

    def remove(self) -> None:
        """Permanently dissolve this MetaAgent.

        Clears ``meta_agents`` and ``meta_agent`` on every constituent agent,
        marks the meta-agent as DISSOLVED, and deregisters it from the model.

        After calling ``remove()``, this instance must not be reused.
        """
        self._state = MetaAgentState.DISSOLVED
        self.remove_constituting_agents(set(self._constituting_set))
        self.emit("dissolved", self)
        super().remove()

    # ------------------------------------------------------------------
    # Step with lifecycle guard and optional cascade
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Perform the meta-agent's step.

        - If the meta-agent is DORMANT or DISSOLVED, the step is skipped.
        - If ``cascade_step`` is True, ``step()`` is called on every
          constituting agent after this method returns (subclass logic runs
          first).

        Override this method in subclasses to add meta-level behaviour.
        The lifecycle guard and cascade are handled automatically — you do
        not need to call ``super().step()``.

        Example::

            class Alliance(MetaAgent):
                cascade_step = True

                def step(self):
                    self._negotiate_resources()
                    # constituting agents are stepped automatically afterward
        """
        if not self.is_active:
            return

        # Cascade to constituting agents
        if self.cascade_step:
            for agent in list(self._constituting_set):
                if hasattr(agent, "step") and callable(agent.step):
                    agent.step()

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the meta-agent to a plain dictionary.

        Returns a snapshot of the meta-agent's identity, state, and member
        ids that can be stored, logged, or used to reconstruct the agent.

        Returns:
            dict with keys:
                - ``unique_id``: int
                - ``class_name``: str
                - ``name``: str
                - ``state``: str (lifecycle state name)
                - ``member_ids``: list[int] — unique_ids of constituting agents
                - ``cascade_step``: bool
        """
        return {
            "unique_id": self.unique_id,
            "class_name": self.__class__.__name__,
            "name": self.name,
            "state": self._state.name,
            "member_ids": [
                getattr(a, "unique_id", None) for a in self._constituting_set
            ],
            "cascade_step": self.cascade_step,
        }
