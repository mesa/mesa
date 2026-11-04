"""Locations: declare where an agent is, in as many spaces as it needs.

A location is an attribute declared on an agent class. Assigning to it places the
agent in a space, and an agent can have as many locations as it needs::

    class Citizen(Agent):
        home = CellLocation()  # a cell of any discrete space
        district = CellLocation()
        view = ContinuousLocation("opinion")  # a point in model.opinion

    citizen.home = model.city[3, 4]  # joins the city grid
    citizen.view = [0.2, -0.4]  # joins model.opinion
    citizen.view += drift  # moves within it
    citizen.remove()  # leaves every space

Locations are independent unless declared otherwise. When two locations describe
the same place, such as a bird's position in the air and the ground cell below it,
let one follow the other::

    class Bird(Agent):
        position = ContinuousLocation("air")
        cell = CellLocation("ground", follows="position")

The bird then has a single location: writing ``position`` updates ``cell``, and
writing ``cell`` moves ``position`` to the center of that cell. A move is atomic:
if a cell is full or a point has no cell, the bird stays where it was.
"""

from __future__ import annotations

from operator import attrgetter
from typing import TYPE_CHECKING

import numpy as np

from mesa.discrete_space import Network

if TYPE_CHECKING:
    from mesa.agent import Agent
    from mesa.discrete_space import Cell
    from mesa.experimental.continuous_space import ContinuousSpace

__all__ = ["CellLocation", "ContinuousLocation"]


class _Location:
    """Shared behavior of locations: registration, space lookup, and followers.

    A location installs a plain ``property`` under its name on the agent class, so
    reading it is as fast as reading any property, and registers itself in the
    class's ``_mesa_locations`` so that ``Agent.remove`` can take the agent out of
    its spaces.
    """

    def __init__(self, space: str | None = None, *, follows: str | None = None):
        self.space_name = space
        self.follows = follows
        self.name: str | None = None
        self._key: str | None = None
        # The locations that follow this one, per agent class
        self._followers_cache: dict[type, tuple[CellLocation, ...]] = {}

    def __set_name__(self, owner, name: str) -> None:
        self.name = name
        self._key = f"_mesa_location_{name}"
        inherited = getattr(owner, "_mesa_locations", ())
        owner._mesa_locations = (
            *(loc for loc in inherited if loc.name != name),
            self,
        )
        # Agents without the location read None from the class
        setattr(owner, self._key, None)
        setattr(owner, name, self._property())

    def __repr__(self) -> str:
        args = [repr(self.space_name)] if self.space_name is not None else []
        if self.follows is not None:
            args.append(f"follows={self.follows!r}")
        return f"{type(self).__name__}({', '.join(args)})"

    def _property(self) -> property:  # pragma: no cover
        raise NotImplementedError

    def _set(self, agent: Agent, value) -> None:  # pragma: no cover
        raise NotImplementedError

    def _space(self, agent: Agent):
        try:
            return getattr(agent.model, self.space_name)
        except AttributeError:
            raise AttributeError(
                f"{type(agent).__name__}.{self.name} lives in model.{self.space_name}, "
                f"but the model has no attribute {self.space_name!r}."
            ) from None

    def _followers(self, agent: Agent) -> tuple[CellLocation, ...]:
        cls = type(agent)
        try:
            return self._followers_cache[cls]
        except KeyError:
            followers = _declarations(cls)[1].get(self.name, ())
            self._followers_cache[cls] = followers
            return followers

    def _primary(self, agent: Agent) -> _Location:
        """Return the location this one follows."""
        return _declarations(type(agent))[0][self.follows]

    def _remove(self, agent: Agent) -> None:
        """Leave the space when the agent is removed; followers leave with their primary."""
        if self.follows is None:
            self._set(agent, None)


def _declarations(cls: type) -> tuple[dict, dict]:
    """Return the locations of an agent class by name, and their followers.

    Checks that every followed location exists and is not itself a follower.
    """
    cached = cls.__dict__.get("_mesa_location_declarations")
    if cached is not None:
        return cached

    locations = {loc.name: loc for loc in getattr(cls, "_mesa_locations", ())}
    followers: dict[str, list[CellLocation]] = {}
    for location in locations.values():
        if location.follows is None:
            continue
        primary = locations.get(location.follows)
        if primary is None:
            raise TypeError(
                f"{cls.__name__}.{location.name} follows {location.follows!r}, "
                f"which is not a location of {cls.__name__}."
            )
        if primary.follows is not None:
            raise TypeError(
                f"{cls.__name__}.{location.name} follows {primary.name}, which itself "
                f"follows {primary.follows}; follow {primary.follows} directly."
            )
        followers.setdefault(primary.name, []).append(location)

    declarations = (locations, {k: tuple(v) for k, v in followers.items()})
    cls._mesa_location_declarations = declarations
    return declarations


# ----------------------------------------------------------------------
# moving followers
# ----------------------------------------------------------------------
def _resolve(agent: Agent, followers, point) -> list:
    """Find each follower's cell for a point. Raises if one has none."""
    return [(follower, follower._resolve(agent, point)) for follower in followers]


def _enter(agent: Agent, targets) -> None:
    """Enter all new cells, or none of them if one is full."""
    state = agent.__dict__
    entered = []
    try:
        for location, cell in targets:
            if cell is not state.get(location._key):
                cell.add_agent(agent)
                entered.append(cell)
    except Exception:
        for cell in entered:
            cell.remove_agent(agent)
        raise


def _commit(agent: Agent, targets) -> None:
    """Leave the old cells and record the new ones. Cannot fail."""
    state = agent.__dict__
    for location, cell in targets:
        old = state.get(location._key)
        if cell is not old:
            if old is not None:
                old.remove_agent(agent)
            state[location._key] = cell


def _leave(agent: Agent, followers) -> None:
    state = agent.__dict__
    for follower in followers:
        cell = state.pop(follower._key, None)
        if cell is not None:
            cell.remove_agent(agent)


class CellLocation(_Location):
    """A location that is a cell of a discrete space.

    Assigning a cell places the agent in it: the agent counts toward the cell's
    capacity and shows up in ``cell.agents`` and neighborhood queries. Assigning
    ``None`` takes it out. Reading returns the cell, or ``None``.

    Args:
        space: the name of the model attribute holding the space. Needed when the
            location follows another one, or is followed by one.
        follows: the name of another location of the same agent that this one
            follows. The agent's cell is then always the cell of ``space`` at that
            location, and assigning a cell moves the other location to its center.

    Notes:
        Without ``follows``, this is what ``CellAgent.cell`` does, under a name of
        your choosing, so an agent can have several cells, in one or more spaces.
    """

    def __init__(self, space: str | None = None, *, follows: str | None = None):
        """Create a cell location."""
        if follows is not None and space is None:
            raise TypeError(
                "A CellLocation that follows another location needs the space to "
                "find its cells in, e.g. CellLocation('ground', follows='position')."
            )
        super().__init__(space, follows=follows)

    def _property(self) -> property:
        key = self._key
        followers_cache = self._followers_cache
        set_location = self._set

        def set_cell(agent: Agent, cell: Cell | None) -> None:
            # Fast path for a cell that nothing follows, as in CellAgent
            if cell is not None and followers_cache.get(type(agent)) == ():
                state = agent.__dict__
                old = state.get(key)
                if cell is old:
                    return
                cell.add_agent(agent)
                if old is not None:
                    old.remove_agent(agent)
                state[key] = cell
                return
            set_location(agent, cell)

        return property(
            attrgetter(key),
            set_location if self.follows is not None else set_cell,
            doc=f"The cell of this agent's {self.name!r} location, or None.",
        )

    def _set(self, agent: Agent, cell: Cell | None) -> None:
        if self.follows is not None:
            self._set_follower(agent, cell)
            return

        followers = self._followers(agent)
        state = agent.__dict__
        old = state.get(self._key)
        if cell is None:
            _leave(agent, followers)
            if old is not None:
                old.remove_agent(agent)
                del state[self._key]
            return
        if cell is old:
            return

        if not followers:
            # The single-space path, as in CellAgent
            cell.add_agent(agent)
            if old is not None:
                old.remove_agent(agent)
            state[self._key] = cell
            return

        targets = [(self, cell), *_resolve(agent, followers, cell.position)]
        _enter(agent, targets)
        _commit(agent, targets)

    def _set_follower(self, agent: Agent, cell: Cell | None) -> None:
        """Move the primary location to the center of the cell."""
        primary = self._primary(agent)
        if cell is None:
            raise ValueError(
                f"{type(agent).__name__}.{self.name} follows {self.follows}; set "
                f"{self.follows} to None to take the agent out of both."
            )
        space = self._space(agent)
        if space._cells.get(cell.coordinate) is not cell:
            raise ValueError(
                f"{type(agent).__name__}.{self.name} takes cells of model."
                f"{self.space_name}, got a cell of another space."
            )
        primary._move_to(agent, cell.position)

    def _resolve(self, agent: Agent, point) -> Cell:
        """Return the cell of this location's space at a point."""
        space = self._space(agent)
        if isinstance(space, Network):
            # A network does not divide space into cells: most points are on no node
            raise TypeError(
                f"{type(agent).__name__}.{self.name} cannot follow a location in a "
                "Network, because a point is not on a node."
            )
        return space.find_nearest_cell(point)

    def _move_to(self, agent: Agent, point) -> None:
        """Move to the cell at a point, for a follower that was written."""
        if self.space_name is None:
            raise TypeError(
                f"{type(agent).__name__}.{self.name} needs its space, e.g. "
                f"CellLocation('grid'), to be moved through a location that follows it."
            )
        self._set(agent, self._resolve(agent, point))


class _Handle:
    """Keeps an agent's row in a ContinuousSpace, for one ContinuousLocation."""

    __slots__ = ("_mesa_index", "space")

    def __init__(self, space: ContinuousSpace) -> None:
        self.space = space
        self._mesa_index: int | None = None


class ContinuousLocation(_Location):
    """A location that is a point in a ContinuousSpace.

    Assigning a point places the agent there, adding it to the space on first use;
    assigning ``None`` takes it out. Reading returns the point as a new array, or
    ``None``. Points outside a bounded space raise; a torus wraps them.

    Args:
        space: the name of the model attribute holding the space, e.g. ``"air"``
            for ``model.air``.

    Notes:
        Reading returns a copy, like any other attribute value: a position saved
        before a move keeps its value, and ``agent.view += step`` moves the agent
        through the checks above. Use the space's ``agent_positions`` array for
        vectorized reads.
    """

    def __init__(self, space: str):
        """Create a continuous location."""
        super().__init__(space)

    def _property(self) -> property:
        key = self._key

        def get(agent: Agent) -> np.ndarray | None:
            handle = agent.__dict__.get(key)
            if handle is None:
                return None
            return handle.space._agent_positions[handle._mesa_index].copy()

        return property(
            get,
            self._set,
            doc=f"The point of this agent's {self.name!r} location, or None.",
        )

    def _set(self, agent: Agent, value) -> None:
        state = agent.__dict__
        handle = state.get(self._key)
        followers = self._followers(agent)

        if value is None:
            _leave(agent, followers)
            if handle is not None:
                handle.space._remove_agent(agent)
                del state[self._key]
            return

        space = self._space(agent) if handle is None else handle.space
        point = np.array(value, dtype=float)
        if point.shape != (space.ndims,):
            raise ValueError(
                f"Expected a point with {space.ndims} coordinates, got {value!r}."
            )
        if not space.in_bounds(point):
            if space.torus:
                point = space.torus_correct(point)
            else:
                raise ValueError(f"point {value} is outside the bounds of the space")

        # Followers resolve and enter their cells before anything changes
        targets = _resolve(agent, followers, point) if followers else ()
        if targets:
            _enter(agent, targets)

        if handle is None:
            handle = _Handle(space)
            space._add_agent(agent, holder=handle)
            state[self._key] = handle
        space._agent_positions[handle._mesa_index] = point

        if targets:
            _commit(agent, targets)

    _move_to = _set
