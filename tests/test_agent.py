"""Agent.py related tests."""

import pickle

import numpy as np
import pytest

from mesa.agent import Agent, AgentSet
from mesa.model import Model


class AgentTest(Agent):
    """Agent class for testing."""

    def get_unique_identifier(self):
        """Return unique identifier for this agent."""
        return self.unique_id


class AgentDoTest(Agent):
    """Agent class for testing."""

    def __init__(
        self,
        model,
    ):
        """Initialize an Agent.

        Args:
            model (Model): the model to which the agent belongs

        """
        super().__init__(model)
        self.agent_set = None

    def get_unique_identifier(self):  # noqa: D102
        return self.unique_id

    def do_add(self):  # noqa: D102
        agent = AgentDoTest(self.model)
        self.agent_set.add(agent)

    def do_remove(self):  # noqa: D102
        self.agent_set.remove(self)


def test_agent_removal():
    """Test agent removal."""
    model = Model()
    agent = AgentTest(model)
    # Check if the agent is added
    assert agent in model.agents

    agent.remove()
    # Check if the agent is removed
    assert agent not in model.agents


def test_agentset():
    """Test agentset class."""
    # create agentset
    model = Model()
    agents = [AgentTest(model) for _ in range(10)]

    agentset = AgentSet(agents)

    assert agents[0] in agentset
    assert len(agentset) == len(agents)
    assert all(a1 == a2 for a1, a2 in zip(agentset.to_list()[0:5], agents[0:5]))

    for a1, a2 in zip(agentset, agents):
        assert a1 == a2

    def test_function(agent):
        return agent.unique_id > 5

    assert len(agentset.select(at_most=0.2)) == 2  # Select 20% of agents
    assert len(agentset.select(at_most=0.549)) == 5  # Select 50% of agents
    assert len(agentset.select(at_most=0.09)) == 0  # Select 0% of agents
    assert len(agentset.select(at_most=1.0)) == 10  # Select 100% agents
    assert len(agentset.select(at_most=1)) == 1  # Select 1 agent

    assert len(agentset.select(test_function)) == 5
    assert len(agentset.select(test_function, at_most=2)) == 2
    assert len(agentset.select(test_function, inplace=True)) == 5
    assert agentset.select(inplace=True) == agentset
    assert all(a1 == a2 for a1, a2 in zip(agentset.select(), agentset))
    assert all(
        a1 == a2 for a1, a2 in zip(agentset.select(at_most=5), agentset.to_list()[:5])
    )

    assert len(agentset.shuffle(inplace=False).select(at_most=5)) == 5

    def test_function(agent):
        return agent.unique_id

    assert all(
        a1 == a2
        for a1, a2 in zip(
            agentset.sort(test_function, ascending=False), agentset.to_list()[::-1]
        )
    )
    assert all(
        a1 == a2
        for a1, a2 in zip(
            agentset.sort("unique_id", ascending=False), agentset.to_list()[::-1]
        )
    )

    assert all(
        a1 == a2.unique_id for a1, a2 in zip(agentset.get("unique_id"), agentset)
    )
    assert agentset == agentset.do("get_unique_identifier")

    agentset.discard(agents[0])
    assert agents[0] not in agentset
    agentset.discard(agents[0])  # check if no error is raised on discard

    with pytest.raises(KeyError):
        agentset.remove(agents[0])

    agentset.add(agents[0])
    assert agents[0] in agentset

    # because AgentSet uses weakrefs, we need hard refs as well....
    other_agents, another_set = pickle.loads(  # noqa: S301
        pickle.dumps([agents, AgentSet(agents)])
    )
    assert all(
        a1.unique_id == a2.unique_id for a1, a2 in zip(another_set, other_agents)
    )
    assert len(another_set) == len(other_agents)


def test_agentset_initialization():
    """Test agentset initialization."""
    model = Model()
    empty_agentset = AgentSet([], random=model.random)
    assert len(empty_agentset) == 0
    with pytest.warns(UserWarning):
        empty_agentset2 = AgentSet([])
    assert len(empty_agentset2) == 0

    agents = [AgentTest(model) for _ in range(10)]
    agentset = AgentSet(agents)
    assert len(agentset) == 10


def test_agentset_initialization_w_random():
    """Test agentset initialization."""
    model = Model()
    empty_agentset = AgentSet([], random=model.random)
    assert len(empty_agentset) == 0
    assert empty_agentset.random == model.random

    agents = [AgentTest(model) for _ in range(10)]
    agentset = AgentSet(agents)
    assert len(agentset) == 10
    assert agentset.random == model.random


def test_agentset_serialization():
    """Test pickleability of agentset."""
    model = Model()
    agents = [AgentTest(model) for _ in range(5)]
    agentset = AgentSet(agents)

    serialized = pickle.dumps(agentset)
    deserialized = pickle.loads(serialized)  # noqa: S301

    original_ids = [agent.unique_id for agent in agents]
    deserialized_ids = [agent.unique_id for agent in deserialized]

    assert deserialized_ids == original_ids


def test_agent_membership():
    """Test agent membership in AgentSet."""
    model = Model()
    agents = [AgentTest(model) for _ in range(5)]
    agentset = AgentSet(agents)

    assert agents[0] in agentset
    assert AgentTest(model) not in agentset


def test_agent_rng():
    """Test whether agent.random and agent.rng are equal to model.random and model.rng."""
    model = Model(rng=42)
    agent = Agent(model)
    assert agent.random is model.random
    assert agent.rng is model.rng


def test_agent_create():
    """Test create agent factory method."""
    # Fast Path (No args/kwargs)
    model = Model()
    n = 10
    fast_agents = Agent.create_agents(model, n)

    assert len(fast_agents) == n
    assert all(isinstance(a, Agent) for a in fast_agents)
    assert all(a.model is model for a in fast_agents)

    # Standard Path (With args/kwargs)
    class TestAgent(Agent):
        def __init__(self, model, attr, def_attr, a=0, b=0):
            super().__init__(model)
            self.some_attribute = attr
            self.some_default_value = def_attr
            self.a = a
            self.b = b

    model = Model(rng=42)
    n = 10
    some_attribute = model.rng.random(n)
    a = tuple([model.random.random() for _ in range(n)])
    TestAgent.create_agents(model, n, some_attribute, 5, a=a, b=7)

    for agent, value, a_i in zip(model.agents, some_attribute, a):
        assert agent.some_attribute == value
        assert agent.some_default_value == 5
        assert agent.a == a_i
        assert agent.b == 7


def test_agent_add_remove_discard():
    """Test adding, removing and discarding agents from AgentSet."""
    model = Model()
    agent = AgentTest(model)
    agentset = AgentSet([], random=model.random)

    agentset.add(agent)
    assert agent in agentset

    agentset.remove(agent)
    assert agent not in agentset

    agentset.add(agent)
    agentset.discard(agent)
    assert agent not in agentset

    with pytest.raises(KeyError):
        agentset.remove(agent)


def test_agentset_to_list():
    """Test AgentSet.to_list method."""
    model = Model()
    agents = [AgentTest(model) for _ in range(10)]
    agentset = AgentSet(agents)

    # Test that to_list returns a list
    agent_list = agentset.to_list()
    assert isinstance(agent_list, list)
    assert len(agent_list) == len(agents)

    # Test that the list contains the same agents in the same order
    assert agent_list == agents

    # Test indexing on the returned list
    assert agent_list[0] == agents[0]
    assert agent_list[-1] == agents[-1]
    assert agent_list[1:3] == agents[1:3]

    # Test that modifying the list doesn't affect the AgentSet
    agent_list.pop()
    assert len(agentset) == 10


def test_agentset_get_item():
    """Test integer based access to AgentSet and deprecation warning."""
    model = Model()
    agents = [AgentTest(model) for _ in range(10)]
    agentset = AgentSet(agents)

    # Test that __getitem__ raises PendingDeprecationWarning
    with pytest.warns(
        PendingDeprecationWarning, match="AgentSet.__getitem__ is deprecated"
    ):
        assert agentset[0] == agents[0]
        assert agentset[-1] == agents[-1]
        assert agentset[1:3] == agents[1:3]

    with pytest.warns(PendingDeprecationWarning), pytest.raises(IndexError):
        agentset[20]


def test_agentset_do_str():
    """Test AgentSet.do with str."""
    model = Model()
    agents = [AgentTest(model) for _ in range(10)]
    agentset = AgentSet(agents)

    with pytest.raises(AttributeError):
        agentset.do("non_existing_method")

    # tests for addition and removal in do
    # do iterates, so no error should be raised to change size while iterating
    # related to issue #1595

    # setup
    n = 10
    model = Model()
    agents = [AgentDoTest(model) for _ in range(n)]
    agentset = AgentSet(agents)
    for agent in agents:
        agent.agent_set = agentset

    agentset.do("do_add")
    assert len(agentset) == 2 * n

    # setup
    model = Model()
    agents = [AgentDoTest(model) for _ in range(10)]
    agentset = AgentSet(agents)
    for agent in agents:
        agent.agent_set = agentset

    agentset.do("do_remove")
    assert len(agentset) == 0


def test_agentset_do_callable():
    """Test AgentSet.do with callable."""
    model = Model()
    agents = [AgentTest(model) for _ in range(10)]
    agentset = AgentSet(agents)

    # Test callable with non-existent function
    with pytest.raises(AttributeError):
        agentset.do(lambda agent: agent.non_existing_method())

    # tests for addition and removal in do using callables
    # do iterates, so no error should be raised to change size while iterating
    # related to issue #1595

    # setup for lambda function tests
    n = 10
    model = Model()
    agents = [AgentDoTest(model) for _ in range(n)]
    agentset = AgentSet(agents)
    for agent in agents:
        agent.agent_set = agentset

    # Lambda for addition
    agentset.do(lambda agent: agent.do_add())
    assert len(agentset) == 2 * n

    # setup again for lambda function tests
    model = Model()
    agents = [AgentDoTest(model) for _ in range(10)]
    agentset = AgentSet(agents)
    for agent in agents:
        agent.agent_set = agentset

    # Lambda for removal
    agentset.do(lambda agent: agent.do_remove())
    assert len(agentset) == 0

    # setup for actual function tests
    def add_function(agent):
        agent.do_add()

    def remove_function(agent):
        agent.do_remove()

    # setup again for actual function tests
    model = Model()
    agents = [AgentDoTest(model) for _ in range(n)]
    agentset = AgentSet(agents)
    for agent in agents:
        agent.agent_set = agentset

    # Actual function for addition
    agentset.do(add_function)
    assert len(agentset) == 2 * n

    # setup again for actual function tests
    model = Model()
    agents = [AgentDoTest(model) for _ in range(10)]
    agentset = AgentSet(agents)
    for agent in agents:
        agent.agent_set = agentset

    # Actual function for removal
    agentset.do(remove_function)
    assert len(agentset) == 0


def test_agentset_get():
    """Test AgentSet.get."""
    model = Model()
    [AgentTest(model) for _ in range(10)]

    agentset = model.agents

    agentset.set("a", 5)
    agentset.set("b", 6)

    # Case 1: Normal retrieval of existing attributes
    values = agentset.get(["a", "b"])
    assert all((a == 5) & (b == 6) for a, b in values)

    # Case 2: Raise AttributeError when attribute doesn't exist
    with pytest.raises(AttributeError):
        agentset.get("unknown_attribute")

    # Case 3: Use default value when attribute is missing
    results = agentset.get(
        "unknown_attribute", handle_missing="default", default_value=True
    )
    assert all(results) is True

    # Case 4: Retrieve mixed attributes with default value for missing ones
    values = agentset.get(
        ["a", "unknown_attribute"], handle_missing="default", default_value=True
    )
    assert all((a == 5) & (unknown is True) for a, unknown in values)

    # Case 5: Invalid handle_missing value raises ValueError
    with pytest.raises(ValueError):
        agentset.get("unknown_attribute", handle_missing="some nonsense value")

    # Case 6: Retrieve multiple attributes with mixed existence and 'default' handling
    values = agentset.get(
        ["a", "b", "unknown_attribute"], handle_missing="default", default_value=0
    )
    assert all((a == 5) & (b == 6) & (unknown == 0) for a, b, unknown in values)

    # Case 7: 'default' handling when one attribute is completely missing from some agents
    agentset.select(at_most=0.5).set("c", 8)  # Only some agents have attribute 'c'
    values = agentset.get(["a", "c"], handle_missing="default", default_value=-1)
    assert all((a == 5) & (c in [8, -1]) for a, c in values)


def test_agentset_agg():
    """Test agentset.agg."""
    model = Model()
    agents = [AgentTest(model) for i in range(10)]

    # Assign some values to attributes
    for i, agent in enumerate(agents):
        agent.energy = i + 1
        agent.wealth = 10 * (i + 1)

    agentset = AgentSet(agents)

    # Test min aggregation
    min_energy = agentset.agg("energy", min)
    assert min_energy == 1

    # Test max aggregation
    max_energy = agentset.agg("energy", max)
    assert max_energy == 10

    # Test sum aggregation
    total_energy = agentset.agg("energy", sum)
    assert total_energy == sum(range(1, 11))

    # Test mean aggregation using numpy
    avg_wealth = agentset.agg("wealth", np.mean)
    assert avg_wealth == 55.0

    # Test aggregation with a custom function
    def custom_func(values):
        return sum(values) / len(values)

    custom_avg_energy = agentset.agg("energy", custom_func)
    assert custom_avg_energy == 5.5

    # Test with list of functions
    min_max_energy = agentset.agg("energy", [min, max])
    assert min_max_energy == [1, 10]

    # Test with tuple of functions
    min_energy, max_energy, total_energy = agentset.agg("energy", (min, max, sum))
    assert [min_energy, max_energy, total_energy] == [1, 10, 55]

    # Test with custom functions in a list
    stats = agentset.agg("wealth", [min, max, np.mean, custom_func])
    assert stats == [10, 100, 55.0, 55.0]


def test_agentset_set_method():
    """Test AgentSet.set."""

    # Initialize the model and agents with and without existing attributes
    class TestAgentWithAttribute(Agent):
        def __init__(self, model, age=None):
            super().__init__(model)
            self.age = age

    model = Model()
    agents = [TestAgentWithAttribute(model, age=i) for i in range(5)]
    agentset = AgentSet(agents)

    # Set a new attribute "health" and an existing attribute "age" for all agents
    agentset.set("health", 100).set("age", 50).set("status", "active")

    # Check if all agents have the "health", "age", and "status" attributes correctly set
    for agent in agentset:
        assert hasattr(agent, "health")
        assert agent.health == 100
        assert hasattr(agent, "age")
        assert agent.age == 50
        assert hasattr(agent, "status")
        assert agent.status == "active"


def test_agentset_map_str():
    """Test AgentSet.map with strings."""
    model = Model()
    agents = [AgentTest(model) for _ in range(10)]
    agentset = AgentSet(agents)

    with pytest.raises(AttributeError):
        agentset.do("non_existing_method")

    results = agentset.map("get_unique_identifier")
    assert all(i == entry for i, entry in zip(results, range(1, 11)))


def test_agentset_map_callable():
    """Test AgentSet.map with callable."""
    model = Model()
    agents = [AgentTest(model) for _ in range(10)]
    agentset = AgentSet(agents)

    # Test callable with non-existent function
    with pytest.raises(AttributeError):
        agentset.map(lambda agent: agent.non_existing_method())

    # tests for addition and removal in do using callables
    # do iterates, so no error should be raised to change size while iterating
    # related to issue #1595

    results = agentset.map(lambda agent: agent.unique_id)
    assert all(i == entry for i, entry in zip(results, range(1, 11)))


def test_agentset_shuffle_do():
    """Test AgentSet.shuffle_do method."""
    model = Model()

    class TestAgentShuffleDo(Agent):
        def __init__(self, model):
            super().__init__(model)
            self.called = False

        def test_method(self):
            self.called = True

    agents = [TestAgentShuffleDo(model) for _ in range(100)]
    agentset = AgentSet(agents)

    # Test shuffle_do with a string method name
    agentset.shuffle_do("test_method")
    assert all(agent.called for agent in agents)

    # Reset the called flag
    for agent in agents:
        agent.called = False

    # Test shuffle_do with a callable
    agentset.shuffle_do(lambda agent: setattr(agent, "called", True))
    assert all(agent.called for agent in agents)

    # Verify that the order is indeed shuffled
    original_order = list(agentset)
    shuffled_order = []
    agentset.shuffle_do(lambda agent: shuffled_order.append(agent))
    assert original_order != shuffled_order, (
        "The order should be different after shuffle_do"
    )

    class AgentWithRemove(Agent):
        def __init__(self, model):
            super().__init__(model)
            self.is_alive = True

        def remove(self):
            super().remove()
            self.is_alive = False

        def step(self):
            if not self.is_alive:
                raise Exception

            agent_to_remove = self.random.choice(self.model.agents)

            if agent_to_remove is not self:
                agent_to_remove.remove()

    model = Model(rng=32)
    for _ in range(100):
        AgentWithRemove(model)
    model.agents.shuffle_do("step")


def test_agentset_get_attribute():
    """Test AgentSet.get for attributes."""
    model = Model()
    agents = [AgentTest(model) for _ in range(10)]
    agentset = AgentSet(agents)

    unique_ids = agentset.get("unique_id")
    assert unique_ids == [agent.unique_id for agent in agents]

    with pytest.raises(AttributeError):
        agentset.get("non_existing_attribute")

    model = Model()
    agents = []
    for i in range(10):
        agent = AgentTest(model)
        agent.i = i**2
        agents.append(agent)
    agentset = AgentSet(agents)

    values = agentset.get(["unique_id", "i"])

    for value, agent in zip(values, agents):
        (
            unique_id,
            i,
        ) = value
        assert agent.unique_id == unique_id
        assert agent.i == i


class OtherAgentType(Agent):
    """Another Agent class for testing."""

    def get_unique_identifier(self):
        """Return unique identifier."""
        return self.unique_id


def test_agentset_select_by_type():
    """Test AgentSet.select for agent type."""
    model = Model()
    # Create a mix of agents of two different types
    test_agents = [AgentTest(model) for _ in range(4)]
    other_agents = [OtherAgentType(model) for _ in range(6)]

    # Combine the two types of agents
    mixed_agents = test_agents + other_agents
    agentset = AgentSet(mixed_agents, random=model.random)

    # Test selection by type
    selected_test_agents = agentset.select(agent_type=AgentTest)
    assert len(selected_test_agents) == len(test_agents)
    assert all(isinstance(agent, AgentTest) for agent in selected_test_agents)
    assert len(selected_test_agents) == 4

    selected_other_agents = agentset.select(agent_type=OtherAgentType)
    assert len(selected_other_agents) == len(other_agents)
    assert all(isinstance(agent, OtherAgentType) for agent in selected_other_agents)
    assert len(selected_other_agents) == 6

    # Test with no type specified (should select all agents)
    all_agents = agentset.select()
    assert len(all_agents) == len(mixed_agents)


def test_agentset_shuffle():
    """Test AgentSet.shuffle."""
    model = Model()
    test_agents = [AgentTest(model) for _ in range(12)]

    agentset = AgentSet(test_agents, random=model.random)
    agentset = agentset.shuffle()
    assert not all(a1 == a2 for a1, a2 in zip(test_agents, agentset))

    agentset = AgentSet(test_agents, random=model.random)
    agentset.shuffle(inplace=True)
    assert not all(a1 == a2 for a1, a2 in zip(test_agents, agentset))


def test_agentset_groupby():
    """Test AgentSet.groupby."""

    class TestAgent(Agent):
        def __init__(self, model):
            super().__init__(model)
            self.even = self.unique_id % 2 == 0
            self.value = self.unique_id * 10

        def get_unique_identifier(self):
            return self.unique_id

    model = Model()
    agents = [TestAgent(model) for _ in range(10)]
    agentset = AgentSet(agents)

    groups = agentset.groupby("even")
    assert len(groups.groups[True]) == 5
    assert len(groups.groups[False]) == 5

    groups = agentset.groupby(lambda a: a.unique_id % 2 == 0)
    assert len(groups.groups[True]) == 5
    assert len(groups.groups[False]) == 5
    assert len(groups) == 2

    for group_name, group in groups:
        assert len(group) == 5
        assert group_name in {True, False}

    sizes = agentset.groupby("even", result_type="list").map(len)
    assert sizes == {True: 5, False: 5}

    attributes = agentset.groupby("even", result_type="agentset").map("get", "even")
    for group_name, group in attributes.items():
        assert all(group_name == entry for entry in group)

    groups = agentset.groupby("even", result_type="agentset")
    another_ref_to_groups = groups.do("do", "step")
    assert groups == another_ref_to_groups

    groups = agentset.groupby("even", result_type="agentset")
    another_ref_to_groups = groups.do(lambda x: x.do("step"))
    assert groups == another_ref_to_groups

    # New tests for count() method
    groups = agentset.groupby("even")
    count_result = groups.count()
    assert count_result == {True: 5, False: 5}

    # New tests for agg() method
    groups = agentset.groupby("even")
    sum_result = groups.agg("value", sum)
    assert sum_result[True] == sum(agent.value for agent in agents if agent.even)
    assert sum_result[False] == sum(agent.value for agent in agents if not agent.even)

    max_result = groups.agg("value", max)
    assert max_result[True] == max(agent.value for agent in agents if agent.even)
    assert max_result[False] == max(agent.value for agent in agents if not agent.even)

    min_result = groups.agg("value", min)
    assert min_result[True] == min(agent.value for agent in agents if agent.even)
    assert min_result[False] == min(agent.value for agent in agents if not agent.even)

    # Test with a custom aggregation function
    def custom_agg(values):
        return sum(values) / len(values) if values else 0

    custom_result = groups.agg("value", custom_agg)
    assert custom_result[True] == custom_agg(
        [agent.value for agent in agents if agent.even]
    )
    assert custom_result[False] == custom_agg(
        [agent.value for agent in agents if not agent.even]
    )
