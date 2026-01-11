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
from collections.abc import Callable, Iterable
from types import MethodType
from typing import Any

from mesa.agent import Agent, AgentSet


def evaluate_combination(
    candidate_group: tuple[Agent, ...],
    model,
    evaluation_func: Callable[[AgentSet], float] | None,
) -> tuple[AgentSet, float] | None:
    """Evaluate a combination of agents.

    Args:
        candidate_group (Tuple[Agent, ...]): The group of agents to evaluate.
        model: The model instance.
        evaluation_func (Optional[Callable[[AgentSet], float]]): The function
        to evaluate the group.

    Returns:
        Optional[Tuple[AgentSet, float]]: The evaluated group and its value,
        or None.
    """
    group_set = AgentSet(candidate_group, random=model.random)
    if evaluation_func:
        value = evaluation_func(group_set)
        return group_set, value
    return None


def find_combinations(
    model,
    group: AgentSet,
    size: int | tuple[int, int] = (2, 5),
    evaluation_func: Callable[[AgentSet], float] | None = None,
    filter_func: Callable[[list[tuple[AgentSet, float]]], list[tuple[AgentSet, float]]]
    | None = None,
) -> list[tuple[AgentSet, float]]:
    """Find valuable combinations of agents in this set.

    Args:
        model: The model instance.
        group (AgentSet): The set of agents to find combinations in.
        size (Union[int, Tuple[int, int]], optional): The size or range of
        sizes for combinations. Defaults to (2, 5).
        evaluation_func (Optional[Callable[[AgentSet], float]], optional): The
          function to evaluate combinations. Defaults to None.
        filter_func (Optional[Callable[[List[Tuple[AgentSet, float]]]): Allows
          the user to specify how agents are filtered to form groups.
          Defaults to None.
        List[Tuple[AgentSet, float]]]], optional): The function to filter
        combinations. Defaults to None.

    Returns:
        List[Tuple[AgentSet, float]]: The list of valuable combinations, in
        a tuple first agentset of valuable combination  and then the value of
        the combination.
    """
    combinations = []
    # Allow one size or range of sizes to be passed
    size_range = (size, size + 1) if isinstance(size, int) else size

    for candidate_group in itertools.chain.from_iterable(
        itertools.combinations(group, size) for size in range(*size_range)
    ):
        evaluation_result = evaluate_combination(
            candidate_group, model, evaluation_func
        )
        if evaluation_result is not None:
            group_set, result = evaluation_result
            if result:
                combinations.append((group_set, result))

    if len(combinations) > 0 and filter_func:
        filtered_combinations = filter_func(combinations)
        return filtered_combinations

    return combinations


def extract_class(agents_by_type: dict, new_agent_class: object) -> type[Agent] | None:
    """Helper function for create_meta_agents extracts the types of agents.

    Args:
        agents_by_type (dict): The dictionary of agents by type.
        new_agent_class (str): The name of the agent class to be created

    Returns:
        type(Agent) if agent type exists
        None otherwise
    """
    agent_type_names = {}
    for agent in agents_by_type:
        agent_type_names[agent.__name__] = agent

    if new_agent_class in agent_type_names:
        return type(agents_by_type[agent_type_names[new_agent_class]][0])
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
    """Create a new meta-agent class and instantiate agents.

    Parameters:
    model (Any): The model instance.
    new_agent_class (str): The name of the new meta-agent class.
    agents (Iterable[Any]): The agents to be included in the meta-agent.
    meta_attributes (Dict[str, Any]): Attributes to be added to the meta-agent.
    meta_methods (Dict[str, Callable]): Methods to be added to the meta-agent.
    assume_constituting_agent_methods (bool): Whether to assume methods from
    constituting_-agents as meta_agent methods.
    assume_constituting_agent_attributes (bool): Whether to retain attributes
    from constituting_-agents.

    Returns:
        - MetaAgent Instance
    """
    # Convert agents to set to ensure uniqueness
    agents = set(agents)

    # Ensure there is at least one agent base class
    if not mesa_agent_type:
        mesa_agent_type = (Agent,)
    elif not isinstance(mesa_agent_type, tuple):
        mesa_agent_type = (mesa_agent_type,)

    def add_methods(
        meta_agent_instance: Any,
        agents: Iterable[Any],
        meta_methods: dict[str, Callable],
    ) -> None:
        """Add methods to the meta-agent instance.

        Parameters:
        meta_agent_instance (Any): The meta-agent instance.
        agents (Iterable[Any]): The agents to derive methods from.
        meta_methods (Dict[str, Callable]): methods to be added to the meta-agent.
        """
        if assume_constituting_agent_methods:
            agent_classes = {type(agent) for agent in agents}
            if meta_methods is None:
                # Initialize meta_methods if not provided
                meta_methods = {}
            for agent_class in agent_classes:
                for name in agent_class.__dict__:
                    if callable(getattr(agent_class, name)) and not name.startswith(
                        "__"
                    ):
                        original_method = getattr(agent_class, name)
                        meta_methods[name] = original_method

        if meta_methods is not None:
            for name, meth in meta_methods.items():
                bound_method = MethodType(meth, meta_agent_instance)
                setattr(meta_agent_instance, name, bound_method)

    def add_attributes(
        meta_agent_instance: Any, agents: Iterable[Any], meta_attributes: dict[str, Any]
    ) -> None:
        """Add attributes to the meta-agent instance.

        Parameters:
        meta_agent_instance (Any): The meta-agent instance.
        agents (Iterable[Any]): The agents to derive attributes from.
        meta_attributes (Dict[str, Any]): Attributes to be added to the
        meta-agent.
        """
        if assume_constituting_agent_attributes:
            if meta_attributes is None:
                # Initialize meta_attributes if not provided
                meta_attributes = {}
            for agent in agents:
                for name, value in agent.__dict__.items():
                    if not callable(value):
                        meta_attributes[name] = value

        if meta_attributes is not None:
            for key, value in meta_attributes.items():
                setattr(meta_agent_instance, key, value)

    # Path 1 - Add agents to existing meta-agent
    constituting_agents = [a for a in agents if hasattr(a, "meta_agent")]
    if len(constituting_agents) > 0:
        if len(constituting_agents) == 1:
            add_attributes(constituting_agents[0].meta_agent, agents, meta_attributes)
            add_methods(constituting_agents[0].meta_agent, agents, meta_methods)
            constituting_agents[0].meta_agent.add_constituting_agents(agents)

            return constituting_agents[0].meta_agent  # Return the existing meta-agent

        else:
            constituting_agent = model.random.choice(constituting_agents)
            agents = set(agents) - set(constituting_agents)
            add_attributes(constituting_agent.meta_agent, agents, meta_attributes)
            add_methods(constituting_agent.meta_agent, agents, meta_methods)
            constituting_agent.meta_agent.add_constituting_agents(agents)
            # TODO: Add way for user to specify how agents join meta-agent
            # instead of random choice
            return constituting_agent.meta_agent

    else:
        # Path 2 - Create a new instance of an existing meta-agent class
        agent_class = extract_class(model.agents_by_type, new_agent_class)

        if agent_class:
            meta_agent_instance = agent_class(model, agents)
            add_attributes(meta_agent_instance, agents, meta_attributes)
            add_methods(meta_agent_instance, agents, meta_methods)
            return meta_agent_instance
        else:
            # Path 3 - Create a new meta-agent class
            meta_agent_class = type(
                new_agent_class,
                (MetaAgent, *mesa_agent_type),  # Inherit Mesa Agent Classes
                {
                    "unique_id": None,
                    "_constituting_set": None,
                },
            )
            meta_agent_instance = meta_agent_class(model, agents)
            add_attributes(meta_agent_instance, agents, meta_attributes)
            add_methods(meta_agent_instance, agents, meta_methods)
            return meta_agent_instance


class MetaAgent(Agent):
    """A MetaAgent is an agent that contains other agents as components."""

    def __init__(
        self, model, agents: set[Agent] | None = None, name: str = "MetaAgent"
    ):
        """Create a new MetaAgent.

        Args:
            model: The model instance.
            agents (Optional[set[Agent]], optional): The set of agents to
            include in the MetaAgent. Defaults to None.
            name (str, optional): The name of the MetaAgent. Defaults to "MetaAgent".
        """
        super().__init__(model)
        self._constituting_set = AgentSet(agents or [], random=model.random)
        self.name = name

        # Add ref to meta_agent in constituting_agents
        for agent in self._constituting_set:
            agent.meta_agent = self  # TODO: Make a set for meta_agents

    def __len__(self) -> int:
        """Return the number of components."""
        return len(self._constituting_set)

    def __iter__(self):
        """Iterate over components."""
        return iter(self._constituting_set)

    def __contains__(self, agent: Agent) -> bool:
        """Check if an agent is a component."""
        return agent in self._constituting_set

    @property
    def agents(self) -> AgentSet:
        """Get list of Meta-Agent constituting_agents."""
        return self._constituting_set

    @property
    def constituting_agents_by_type(self) -> dict[type, list[Agent]]:
        """Get the constituting_agents grouped by type.

        Returns:
            dict[type, list[Agent]]: A dictionary of constituting_agents grouped by type.
        """
        constituting_agents_by_type = {}
        for agent in self._constituting_set:
            agent_type = type(agent)
            if agent_type not in constituting_agents_by_type:
                constituting_agents_by_type[agent_type] = []
            constituting_agents_by_type[agent_type].append(agent)
        return constituting_agents_by_type

    @property
    def constituting_agent_types(self) -> set[type]:
        """Get the types of all constituting_agents.

        Returns:
            set[type]: A set of unique types of the constituting_agents.
        """
        return {type(agent) for agent in self._constituting_set}

    def get_constituting_agent_instance(self, agent_type) -> set[type]:
        """Get the instance of a constituting_agent of the specified type.

        Args:
            agent_type: The type of the constituting_agent to retrieve.

        Returns:
            The first instance of the specified constituting_agent type.

        Raises:
            ValueError: If no constituting_agent of the specified type is found.
        """
        try:
            return self.constituting_agents_by_type[agent_type][0]
        except KeyError:
            raise ValueError(
                f"No constituting_agent of type {agent_type} found."
            ) from None

    def add_constituting_agents(
        self,
        new_agents: set[Agent],
    ):
        """Add agents as components.

        Args:
            new_agents (set[Agent]): The agents to add to MetaAgent constituting_set.
        """
        for agent in new_agents:
            self._constituting_set.add(agent)
            agent.meta_agent = self  # TODO: Make a set for meta_agents
            self.model.register_agent(agent)

    def remove_constituting_agents(self, remove_agents: set[Agent]):
        """Remove agents as components.

        Args:
            remove_agents (set[Agent]): The agents to remove from MetaAgent.
        """
        for agent in remove_agents:
            self._constituting_set.discard(agent)
            agent.meta_agent = None  # TODO: Remove meta_agent from set
            self.model.deregister_agent(agent)

    def step(self):
        """Perform the agent's step.

        Override this method to define the meta agent's behavior.
        By default, does nothing.
        """
