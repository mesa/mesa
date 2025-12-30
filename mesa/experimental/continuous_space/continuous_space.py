"""A Continuous Space class."""

import warnings
from collections.abc import Iterable
from itertools import compress
from random import Random

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import cdist

from mesa.agent import Agent, AgentSet


class ContinuousSpace:
    """Continuous space where each agent can have an arbitrary position."""

    @property
    def x_min(self):  # noqa: D102
        # compatibility with solara_viz
        return self.dimensions[0, 0]

    @property
    def x_max(self):  # noqa: D102
        # compatibility with solara_viz
        return self.dimensions[0, 1]

    @property
    def y_min(self):  # noqa: D102
        # compatibility with solara_viz
        return self.dimensions[1, 0]

    @property
    def y_max(self):  # noqa: D102
        # compatibility with solara_viz
        return self.dimensions[1, 1]

    @property
    def width(self):  # noqa: D102
        # compatibility with solara_viz
        return self.size[0]

    @property
    def height(self):  # noqa: D102
        # compatibility with solara_viz
        return self.size[1]

    def __init__(
        self,
        dimensions: ArrayLike,
        torus: bool = False,
        random: Random | None = None,
        n_agents: int = 100,
    ) -> None:
        """Create a new continuous space.

        Args:
            dimensions: a numpy array like object where each row specifies the minimum and maximum value of that dimension.
            torus: boolean for whether the space wraps around or not
            random: a seeded stdlib random.Random instance
            n_agents: the expected number of agents in the space

        Internally, a numpy array is used to store the positions of all agents. This is resized if needed,
        but you can control the initial size explicitly by passing n_agents.


        """
        if random is None:
            warnings.warn(
                "Random number generator not specified, this can make models non-reproducible. Please pass a random number generator explicitly",
                UserWarning,
                stacklevel=2,
            )
            random = Random()
        self.random = random

        self.dimensions: np.array = np.asanyarray(dimensions)
        self.ndims: int = self.dimensions.shape[0]
        self.size: np.array = self.dimensions[:, 1] - self.dimensions[:, 0]
        self.center: np.array = np.sum(self.dimensions, axis=1) / 2

        self.torus: bool = torus

        # self._agent_positions is the array containing all agent positions
        # plus potential extra empty rows
        # agent_positions is a view into _agent_positions containing only the filled rows
        self._agent_positions: np.array = np.empty(
            (n_agents, self.dimensions.shape[0]), dtype=float
        )
        self.agent_positions: (
            np.array
        )  # a view on _agent_positions containing all active positions

        # the list of agents in the space
        self.active_agents = []
        self._n_agents = 0  # the number of active agents in the space

        #  a mapping from agents to index and vice versa
        self._index_to_agent: dict[int, Agent] = {}
        self._agent_to_index: dict[Agent, int | None] = {}

    @property
    def agents(self) -> AgentSet:
        """Return an AgentSet with the agents in the space."""
        return AgentSet(self.active_agents, random=self.random)

    def _add_agent(self, agent: Agent) -> int:
        """Helper method for adding an agent to the space.

        This method manages the numpy array with the agent positions and ensuring it is
        enlarged if and when needed. It is called automatically by ContinousSpaceAgent when created.

        """
        index = self._n_agents
        self._n_agents += 1

        if self._agent_positions.shape[0] <= index:
            # we are out of space
            fraction = 0.2  # we add 20%  Fixme
            n = round(fraction * self._n_agents, None)
            self._agent_positions = np.vstack(
                [
                    self._agent_positions,
                    np.empty(
                        (n, self.dimensions.shape[0]),
                    ),
                ]
            )

        self._agent_to_index[agent] = index
        self._index_to_agent[index] = agent

        # we want to maintain a view rather than a copy on the active agents and positions
        # this is essential for the performance of the rest of this code
        self.active_agents.append(agent)
        self.agent_positions = self._agent_positions[0 : self._n_agents]

        return index

    def _remove_agent(self, agent: Agent) -> None:
        """Remove an agent from the space.

        This method is automatically called by ContinuousSpaceAgent.remove.

        """
        index = self._agent_to_index[agent]
        self._agent_to_index.pop(agent, None)
        self._index_to_agent.pop(index, None)
        del self.active_agents[index]

        # Shift all subsequent agents up by 1
        for agent in self.active_agents[index::]:
            old_index = self._agent_to_index[agent]
            self._agent_to_index[agent] = old_index - 1
            self._index_to_agent[old_index - 1] = agent

        # Clean up the stale entry from the last shifted agent
        if len(self.active_agents) > index:
            self._index_to_agent.pop(len(self.active_agents), None)

        # we move all data below the removed agent one row up
        self._agent_positions[index : self._n_agents - 1] = self._agent_positions[
            index + 1 : self._n_agents
        ]
        self._n_agents -= 1
        self.agent_positions = self._agent_positions[0 : self._n_agents]

    def calculate_difference_vector(self, point: np.ndarray, agents=None) -> np.ndarray:
        """Calculate the difference vector between the point and all agenents.

        Args:
            point: the point to calculate the difference vector for
            agents: the agents to calculate the difference vector of point with. By default,
                    all agents are considered.


        """
        point = np.asanyarray(point)
        positions = (
            self.agent_positions
            if agents is None
            else self._agent_positions[[self._agent_to_index[a] for a in agents]]
        )

        delta = positions - point

        if self.torus:
            inverse_delta = delta - np.sign(delta) * self.size

            # we need to use the lowest absolute value from delta and inverse delta
            logical = np.abs(delta) < np.abs(inverse_delta)

            out = np.zeros(delta.shape)
            out[logical] = delta[logical]
            out[~logical] = inverse_delta[~logical]

            delta = out

        return delta

    def calculate_distances(
        self, point: ArrayLike, agents: Iterable[Agent] | None = None, **kwargs
    ) -> tuple[np.ndarray, list]:
        """Calculate the distance between the point and all agents.

        Args:
            point: the point to calculate the difference vector for
            agents: the agents to calculate the difference vector of point with. By default,
                    all agents are considered.
            kwargs: any additional keyword arguments are passed to scipy's cdist, which is used
                    only if torus is False. This allows for non-Euclidian distance measures.

        """
        point = np.asanyarray(point)

        if agents is None:
            positions = self.agent_positions
            agents = self.active_agents
        else:
            positions = self._agent_positions[[self._agent_to_index[a] for a in agents]]
            agents = np.asarray(agents)

        if self.torus:
            delta = np.abs(point - positions)
            delta = np.minimum(delta, self.size - delta, out=delta)

            # + is much faster than np.sum or array.sum
            dists = delta[:, 0] ** 2
            for i in range(1, self.ndims):
                dists += delta[:, i] ** 2
            dists = np.sqrt(dists)
        else:
            dists = cdist(point[np.newaxis, :], positions, **kwargs)[0, :]
        return dists, agents

    def get_agents_in_radius(
        self, point: ArrayLike, radius: float | int = 1
    ) -> tuple[list, np.ndarray]:
        """Return the agents and their distances within a radius for the point."""
        distances, agents = self.calculate_distances(point)
        logical = distances <= radius
        agents = list(compress(agents, logical))
        return (
            agents,
            distances[logical],
        )

    def get_k_nearest_agents(
        self, point: ArrayLike, k: int = 1
    ) -> tuple[list, np.ndarray]:
        """Return the k nearest agents and their distances to the point.

        Notes:
            This method returns exactly k agents, ignoring ties. In case of ties, the
            earlier an agent is inserted the higher it will rank.

        """
        dists, agents = self.calculate_distances(point)

        indices = np.argpartition(dists, k)[:k]
        agents = [agents[i] for i in indices]
        return agents, dists[indices]

    def in_bounds(self, point: ArrayLike) -> bool:
        """Check if point is inside the bounds of the space."""
        return bool(
            (
                (np.asanyarray(point) >= self.dimensions[:, 0])
                & (point <= self.dimensions[:, 1])
            ).all()
        )

    def torus_correct(self, point: ArrayLike) -> np.ndarray:
        """Apply a torus correction to the point."""
        return self.dimensions[:, 0] + np.mod(
            np.asanyarray(point) - self.dimensions[:, 0], self.size
        )
