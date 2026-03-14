from collections.abc import MutableMapping

import numpy as np

from .continuous_space_agents import ContinuousSpaceAgent


class AgentLocations(MutableMapping):
    """Mapping protocol to handle multi-space routing via dictionary syntax."""

    __slots__ = ["agent"]

    def __init__(self, agent):
        self.agent = agent

    def __getitem__(self, space):
        if space is self.agent.space:
            return self.agent.position
        if space not in self.agent._spaces:
            raise KeyError(f"Agent {self.agent.unique_id} is not in this space.")
        return self.agent._mesa_locations.get(space)

    def __setitem__(self, space, location):
        if space is self.agent.space:
            self.agent.position = location
        elif space in self.agent._spaces:
            space.move_agent(self.agent, location)
        else:
            raise KeyError(f"Agent {self.agent.unique_id} is not in this space.")

    def __delitem__(self, space):
        raise NotImplementedError(
            "Use space.remove_agent() to remove an agent from a space."
        )

    def __iter__(self):
        yield self.agent.space
        yield from self.agent._spaces

    def __len__(self):
        return 1 + len(self.agent._spaces)


class SpatialAgent(ContinuousSpaceAgent):
    """An agent designed for the Stacked Spaces architecture."""

    __slots__ = ["_agent_locations", "_mesa_locations", "_spaces"]

    def __init__(self, space, model):
        super().__init__(space, model)
        self._spaces = set()
        self._mesa_locations = {}
        self._agent_locations = AgentLocations(
            self
        )  # proxy to handle _mesa_locations and routing

    @property
    def position(self) -> np.ndarray:
        return super().position

    @position.setter
    def position(self, value: np.ndarray) -> None:
        ContinuousSpaceAgent.position.fset(self, value)

        for space in self._spaces:
            space.move_agent(self, value)

    @property
    def positions(self):
        return self._agent_locations

    def add_to_space(self, spaces):
        if not isinstance(spaces, (list, tuple, set)):
            spaces = [spaces]

        for space in spaces:
            self._spaces.add(space)
            space.add_agent(self)

    def remove(self) -> None:
        for space in list(self._spaces):
            space.remove_agent(self)

        self._spaces.clear()
        self._mesa_locations.clear()

        super().remove()
