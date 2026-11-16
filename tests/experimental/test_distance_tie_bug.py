"""Regression tests for insertion-order ranking of equal-distance agents."""

import numpy as np

from mesa import Model
from mesa.experimental.continuous_space import ContinuousSpace, ContinuousSpaceAgent


def test_k_nearest_agents_returns_tied_agents_in_insertion_order():
    """Agents tied in the result are returned in insertion-order rank."""
    model = Model(rng=42)
    space = ContinuousSpace([[0, 1000], [0, 1000]], random=model.random)
    positions = [
        (502, 500),
        (490, 500),
        (480, 500),
        (520, 500),
        (500, 510),
        (500, 490),
        (500, 480),
        (500, 520),
    ]
    agents = []
    for position in positions:
        agent = ContinuousSpaceAgent(space, model)
        agent.position = np.asarray(position)
        agents.append(agent)

    distances, _ = space.calculate_distances((500, 500))
    assert np.allclose(distances, [2, 10, 20, 20, 10, 10, 20, 20])

    nearest, nearest_distances = space.get_k_nearest_agents((500, 500), k=7)

    assert set(nearest) == {
        agents[0],
        agents[1],
        agents[2],
        agents[3],
        agents[4],
        agents[5],
        agents[6],
    }
    assert np.allclose(np.sort(nearest_distances), [2, 10, 10, 10, 20, 20, 20])
    assert [
        agent for agent, distance in zip(nearest, nearest_distances) if distance == 10
    ] == [agents[1], agents[4], agents[5]]
    assert [
        agent for agent, distance in zip(nearest, nearest_distances) if distance == 20
    ] == [agents[2], agents[3], agents[6]]


def test_k_nearest_agents_selects_earliest_agent_at_tied_cutoff():
    """A later agent tied at the cutoff cannot displace an earlier tied agent."""
    model = Model(rng=42)
    space = ContinuousSpace([[0, 1000], [0, 1000]], random=model.random)
    positions = [
        (480, 500),  # A0: distance 20
        (510, 500),  # A1: distance 10
        (502, 500),  # A2: distance 2
        (480, 500),  # A3: distance 20
        (480, 500),  # A4: distance 20
        (500, 510),  # A5: distance 10
        (480, 500),  # A6: distance 20
        (490, 500),  # A7: distance 10
    ]
    agents = []
    for position in positions:
        agent = ContinuousSpaceAgent(space, model)
        agent.position = np.asarray(position)
        agents.append(agent)

    distances, _ = space.calculate_distances((500, 500))
    assert np.allclose(distances, [20, 10, 2, 20, 20, 10, 20, 10])

    nearest, nearest_distances = space.get_k_nearest_agents((500, 500), k=3)

    assert set(nearest) == {agents[1], agents[2], agents[5]}
    assert [
        agent for agent, distance in zip(nearest, nearest_distances) if distance == 10
    ] == [
        agents[1],
        agents[5],
    ]


def test_k_nearest_agents_preserves_tie_order_after_removal():
    """Removing an agent does not change insertion rank among surviving ties."""
    model = Model(rng=42)
    space = ContinuousSpace([[0, 10], [0, 10]], random=model.random)
    agents = []
    for position in ([4, 5], [5, 6], [6, 5]):
        agent = ContinuousSpaceAgent(space, model)
        agent.position = np.asarray(position)
        agents.append(agent)

    # Swap the last agent into the first slot; active order becomes A2, A1.
    agents[0].remove()
    assert space.active_agents == [agents[2], agents[1]]

    nearest_one, _ = space.get_k_nearest_agents((5, 5), k=1)
    nearest_two, _ = space.get_k_nearest_agents((5, 5), k=2)

    assert nearest_one == [agents[1]]
    assert nearest_two == [agents[1], agents[2]]
