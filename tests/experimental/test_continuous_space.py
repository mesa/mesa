"""Tests for continuous space."""

import numpy as np
import pytest

from mesa import Model
from mesa.experimental.continuous_space import ContinuousSpace, ContinuousSpaceAgent


def test_continuous_space():
    """Test ContinuousSpace class."""
    model = Model(rng=42)

    dimensions = np.asarray([[0, 1], [-1, 0]])
    space = ContinuousSpace(dimensions, torus=False, random=model.random)

    # check some default fields
    assert space.ndims == 2
    assert np.all(space.size == [1, 1])
    assert np.all(space.center == [0.5, -0.5])
    assert len(space.agents) == 0

    # check in_bounds
    assert space.in_bounds([0.5, -0.5])
    assert not space.in_bounds([-0.5, -0.5])
    assert not space.in_bounds([1.5, -0.5])
    assert not space.in_bounds([0.5, 0.5])
    assert not space.in_bounds([0.5, -1.5])

    # check torus correction
    space = ContinuousSpace(dimensions, torus=True, random=model.random)
    assert np.all(space.torus_correct([-0.5, 0.5]) == [0.5, -0.5])
    assert np.all(space.torus_correct([0.5, -0.5]) == [0.5, -0.5])
    assert np.all(space.torus_correct([0.5, -0.5]) == [0.5, -0.5])
    assert np.all(space.torus_correct([1.5, -0.5]) == [0.5, -0.5])
    assert np.all(space.torus_correct([0.5, -1.5]) == [0.5, -0.5])

    # check 3d
    dimensions = np.asarray([[0, 2], [-2, 0], [-2, 2]])
    space = ContinuousSpace(dimensions, torus=False, random=model.random)

    # check some default fields
    assert space.ndims == 3
    assert np.all(space.size == [2, 2, 4])
    assert np.all(space.center == [1, -1, 0])

    # check in_bounds
    assert space.in_bounds([1, -1, 0])
    assert not space.in_bounds([-0.5, -1, 0])
    assert not space.in_bounds([2.5, -1, 0])
    assert not space.in_bounds([1, 0.5, 0])
    assert not space.in_bounds([1, -2.5, 0])
    assert not space.in_bounds([1, -1, -3])
    assert not space.in_bounds([1, -1, 3])
    assert not space.in_bounds([-0.5, -1, 3])
    assert not space.in_bounds([1, 0.5, 3])


def test_continuous_agent():
    """Test ContinuousSpaceAgent class."""
    model = Model(rng=42)

    dimensions = np.asarray([[0, 1], [0, 1]])
    space = ContinuousSpace(dimensions, torus=False, random=model.random)

    for _ in range(10):
        agent = ContinuousSpaceAgent(space, model)
        position = [agent.random.random(), agent.random.random()]
        agent.position = position
        agent.coordinate = position

    assert space.agent_positions.shape == (10, 2)
    for agent in space.agents:
        a = agent.position
        b = space._agent_positions[space._agent_to_index[agent]]
        assert np.all(a == b)
        assert np.all(agent.position == agent.coordinate)

    # add more agents, triggering a resizeing of the array
    for _ in range(100):
        agent = ContinuousSpaceAgent(space, model)
        position = [agent.random.random(), agent.random.random()]
        agent.position = position
        agent.coordinate = position

    assert space.agent_positions.shape == (110, 2)
    for agent in space.agents:
        a = agent.position
        b = space._agent_positions[space._agent_to_index[agent]]
        assert np.all(a == b)
        assert np.all(agent.position == agent.coordinate)

    # remove all agents and check if the view is updated throughout correctly
    for i, agent in enumerate(space.agents):
        assert np.all(
            agent.position == agent.coordinate
        )  ## check if updates of indices is correctly done
        agent.remove()
        assert space.agent_positions.shape == (110 - 1 - i, 2)

    model = Model(rng=42)

    dimensions = np.asarray([[0, 1], [0, 1]])
    space = ContinuousSpace(dimensions, torus=True, random=model.random)
    agent = ContinuousSpaceAgent(space, model)
    agent.position = [1.1, 1.1]
    assert np.allclose(agent.position, [0.1, 0.1])

    dimensions = np.asarray([[0, 1], [0, 1]])
    space = ContinuousSpace(dimensions, torus=False, random=model.random)
    agent = ContinuousSpaceAgent(space, model)
    with pytest.raises(ValueError):
        agent.position = [1.1, 1.1]


def test_continous_space_calculate_distances():
    """Test ContinuousSpace.distance method."""
    # non torus
    model = Model(rng=42)
    dimensions = np.asarray([[0, 1], [0, 1]])
    space = ContinuousSpace(dimensions, torus=False, random=model.random)

    agent = ContinuousSpaceAgent(space, model)
    agent.position = [0.1, 0.1]

    distances, agents = space.calculate_distances([0.1, 0.9])
    assert np.all(
        distances
        == [
            0.8,
        ]
    )
    assert np.all(
        agents
        == [
            agent,
        ]
    )

    distances, agents = space.calculate_distances([0.9, 0.1])
    assert np.all(
        distances
        == [
            0.8,
        ]
    )
    assert np.all(
        agents
        == [
            agent,
        ]
    )

    # torus
    model = Model(rng=42)
    dimensions = np.asarray([[0, 1], [0, 1]])
    space = ContinuousSpace(dimensions, torus=True, random=model.random)

    agent = ContinuousSpaceAgent(space, model)
    agent.position = [0.1, 0.1]

    distances, agents = space.calculate_distances([0.1, 0.9])
    assert np.all(
        np.isclose(
            distances,
            [
                0.2,
            ],
        )
    )
    assert np.all(
        agents
        == [
            agent,
        ]
    )

    distances, agents = space.calculate_distances([0.9, 0.1])
    assert np.all(
        np.isclose(
            distances,
            [
                0.2,
            ],
        )
    )
    assert np.all(
        agents
        == [
            agent,
        ]
    )

    distances, agents = space.calculate_distances([0.9, 0.9])
    assert np.all(
        np.isclose(
            distances,
            [
                0.2 * 2**0.5,
            ],
        )
    )
    assert np.all(
        agents
        == [
            agent,
        ]
    )

    distances, agents = space.calculate_distances(
        [0.9, 0.9],
        agents=[
            agent,
        ],
    )
    assert np.all(
        np.isclose(
            distances,
            [
                0.2 * 2**0.5,
            ],
        )
    )
    assert np.all(
        agents
        == [
            agent,
        ]
    )


def test_continous_space_difference_vector():
    """Test ContinuousSpace.get_difference_vector method."""
    # non torus
    model = Model(rng=42)
    dimensions = np.asarray([[0, 1], [0, 1]])
    space = ContinuousSpace(dimensions, torus=False, random=model.random)

    agent = ContinuousSpaceAgent(space, model)
    agent.position = [0.1, 0.1]

    vector = space.calculate_difference_vector([0.1, 0.9])
    assert np.all(vector == [0, -0.8])

    vector = space.calculate_difference_vector([0.9, 0.1])
    assert np.all(vector == [-0.8, 0])

    # torus
    model = Model(rng=42)
    dimensions = np.asarray([[0, 1], [0, 1]])
    space = ContinuousSpace(dimensions, torus=True, random=model.random)

    agent = ContinuousSpaceAgent(space, model)
    agent.position = [0.1, 0.1]

    vector = space.calculate_difference_vector([0.1, 0.9])
    assert np.allclose(vector, [0, 0.2])

    vector = space.calculate_difference_vector([0.9, 0.1])
    assert np.allclose(vector, [0.2, 0])

    vector = space.calculate_difference_vector([0.9, 0.9])
    assert np.allclose(vector, [0.2, 0.2])


def test_continuous_space_get_k_nearest_agents():  # noqa: D103
    # non torus
    model = Model(rng=42)
    dimensions = np.asarray([[0, 1], [0, 1]])
    space = ContinuousSpace(dimensions, torus=False, random=model.random)

    agent = ContinuousSpaceAgent(space, model)
    agent.position = [0.1, 0.1]

    agent = ContinuousSpaceAgent(space, model)
    agent.position = [0.1, 0.9]

    agent = ContinuousSpaceAgent(space, model)
    agent.position = [0.9, 0.1]

    agent = ContinuousSpaceAgent(space, model)
    agent.position = [0.9, 0.9]

    agent = ContinuousSpaceAgent(space, model)
    agent.position = [0.5, 0.5]

    agents, distances = space.get_k_nearest_agents([0.1, 0.1], k=1)
    assert len(agents) == 1
    assert np.allclose(
        distances,
        [
            0,
        ],
    )

    agents, distances = space.get_k_nearest_agents([0.5, 0.1], k=1)
    assert len(agents) == 1
    assert np.allclose(distances, [0.4, 0.4])

    agents, distances = space.get_k_nearest_agents([0.5, 0.1], k=2)
    assert len(agents) == 2
    assert np.allclose(distances, [0.4, 0.4])

    # torus
    model = Model(rng=42)
    dimensions = np.asarray([[0, 1], [0, 1]])
    space = ContinuousSpace(dimensions, torus=True, random=model.random)

    agent = ContinuousSpaceAgent(space, model)
    agent.position = [0.1, 0.1]

    agent = ContinuousSpaceAgent(space, model)
    agent.position = [0.9, 0.1]

    agent = ContinuousSpaceAgent(space, model)
    agent.position = [0.9, 0.9]

    agents, distances = space.get_k_nearest_agents([0.0, 0.1], k=2)
    assert len(agents) == 2
    assert np.allclose(distances, [0.1, 0.1])


def test_continuous_space_get_agents_in_radius():  # noqa: D103
    # non torus
    model = Model(rng=42)
    dimensions = np.asarray([[0, 1], [0, 1]])
    space = ContinuousSpace(dimensions, torus=False, random=model.random)

    positions = [
        [0.1, 0.1],
        [0.1, 0.9],
        [0.9, 0.1],
        [0.9, 0.9],
        [0.5, 0.5],
    ]

    for position in positions:
        agent = ContinuousSpaceAgent(space, model)
        agent.position = position

    agents, distances = space.get_agents_in_radius([0.1, 0.1], radius=0.1)
    assert len(agents) == 1
    assert np.allclose(
        distances,
        [
            0,
        ],
    )

    agents, distances = space.get_agents_in_radius([0.5, 0.1], radius=0.4)
    assert len(agents) == 3
    assert np.allclose(distances, [0.4, 0.4, 0.4])

    agents, distances = space.get_agents_in_radius([0.5, 0.5], radius=1)
    assert len(agents) == 5

    # torus
    model = Model(rng=42)
    dimensions = np.asarray([[0, 1], [0, 1]])
    space = ContinuousSpace(dimensions, torus=True, random=model.random)

    positions = [
        [0.1, 0.1],
        [0.1, 0.9],
        [0.9, 0.1],
        [0.9, 0.9],
        [0.5, 0.5],
    ]

    for position in positions:
        agent = ContinuousSpaceAgent(space, model)
        agent.position = position

    agents, distances = space.get_agents_in_radius([0.0, 0.1], radius=0.1)
    assert len(agents) == 2
    assert np.allclose(distances, [0.1, 0.1])


def test_get_neighbor_methos():  # noqa: D103
    # non torus
    model = Model(rng=42)
    dimensions = np.asarray([[0, 1], [0, 1]])
    space = ContinuousSpace(dimensions, torus=False, random=model.random)

    positions = [
        [0.1, 0.1],
        [0.1, 0.9],
        [0.9, 0.1],
        [0.9, 0.9],
        [0.5, 0.5],
    ]

    for position in positions:
        agent = ContinuousSpaceAgent(space, model)
        agent.position = position

    agent: ContinuousSpaceAgent = model.agents[-1]  # 0.5, 0.5
    agents, distances = agent.get_neighbors_in_radius(1)
    assert len(agents) == 4

    agents, distances = agent.get_neighbors_in_radius(0.1)
    assert len(agents) == 0

    agent: ContinuousSpaceAgent = model.agents[0]  # 0.1, 0.1
    agents, distances = agent.get_nearest_neighbors(k=2)
    assert len(agents) == 2

    # torus
    model = Model(rng=42)
    dimensions = np.asarray([[0, 1], [0, 1]])
    space = ContinuousSpace(dimensions, torus=True, random=model.random)

    positions = [
        [0.1, 0.1],
        [0.1, 0.9],
        [0.9, 0.1],
        [0.9, 0.9],
        [0.5, 0.5],
    ]

    for position in positions:
        agent = ContinuousSpaceAgent(space, model)
        agent.position = position

    agent: ContinuousSpaceAgent = model.agents[-1]  # 0.5, 0.5
    agents, distances = agent.get_neighbors_in_radius(1)
    assert len(agents) == 4

    agent: ContinuousSpaceAgent = model.agents[0]  # 0.1, 0.1
    agents, distances = agent.get_nearest_neighbors(k=2)
    assert len(agents) == 2
    assert np.allclose(distances, [0.2, 0.2])


def test_agent_removal_no_ghost_entries():
    """Test that removing an agent doesn't leave ghost entries in _index_to_agent.

    This test verifies the fix for the "Ghost Agent" bug where removing an agent
    from ContinuousSpace would leave stale entries in the internal _index_to_agent
    dictionary. The bug occurred because when agents were shifted to fill the gap
    left by a removed agent, the old index entry of the last shifted agent was not
    deleted.

    Regression test for GitHub issue #3029.
    """
    model = Model(rng=42)
    dimensions = np.asarray([[0, 10], [0, 10]])

    # Test 1: Remove middle agent
    space = ContinuousSpace(dimensions, torus=False, random=model.random)
    agents = []
    for i in range(3):
        agent = ContinuousSpaceAgent(space, model)
        agent.position = [float(i), float(i)]
        agents.append(agent)

    space._remove_agent(agents[1])
    assert len(space.active_agents) == 2
    assert len(space._index_to_agent) == 2
    assert len(space._agent_to_index) == 2

    for idx, agent in enumerate(space.active_agents):
        assert space._agent_to_index[agent] == idx
        assert space._index_to_agent[idx] == agent

    # Test 2: Remove last agent
    space = ContinuousSpace(dimensions, torus=False, random=model.random)
    agents = [ContinuousSpaceAgent(space, model) for _ in range(3)]
    for i, agent in enumerate(agents):
        agent.position = [float(i), float(i)]

    space._remove_agent(agents[2])
    assert len(space.active_agents) == 2
    assert len(space._index_to_agent) == 2

    # Test 3: Sequential removals
    space = ContinuousSpace(dimensions, torus=False, random=model.random)
    agents = [ContinuousSpaceAgent(space, model) for _ in range(5)]
    for i, agent in enumerate(agents):
        agent.position = [float(i), float(i)]

    space._remove_agent(agents[2])
    assert len(space.active_agents) == 4
    assert len(space._index_to_agent) == 4

    space._remove_agent(agents[0])
    assert len(space.active_agents) == 3
    assert len(space._index_to_agent) == 3

    for idx, agent in enumerate(space.active_agents):
        assert space._agent_to_index[agent] == idx
        assert space._index_to_agent[idx] == agent
