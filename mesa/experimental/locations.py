"""Locations: declare where an agent is, in as many spaces as it needs.

A location is an attribute declared on an agent class. Assigning to it places the
agent in a space, and an agent can have as many locations as it needs::

    class Citizen(Agent):
        home = CellLocation("city")  # a cell of model.city
        district = CellLocation("districts")
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
        cell = CellLocation("ground", follows=position)

The bird then has a single location: writing ``position`` updates ``cell``, and
writing ``cell`` moves ``position`` to the center of that cell. A move is atomic:
if a cell is full or a point has no cell, it raises, and nothing has moved.

Locations find a named space, such as ``model.air``, by looking up that model
attribute when they need it. Don't replace a space while agents are in it: agents
already placed stay in the old space, while followers move into the new one.
"""

from __future__ import annotations

import warnings
from operator import attrgetter
from typing import TYPE_CHECKING

import numpy as np

from mesa.discrete_space import Grid, HexGrid, Network, VoronoiGrid

if TYPE_CHECKING:
    from mesa.agent import Agent
    from mesa.discrete_space import Cell
    from mesa.experimental.continuous_space import ContinuousSpace

__all__ = ["CellLocation", "ContinuousLocation"]


class _Location:
    """Shared behavior of locations: registration, space lookup, and followers.

    A location installs a plain ``property`` under its name on the agent class, so
    reading it is as fast as reading any property, and records itself in the
    class's ``_mesa_declared_locations``. ``Agent`` collects those of a class and
    all its bases into ``_mesa_locations``, so that ``Agent.remove`` can take the
    agent out of its spaces.
    """

    def __init__(self, space: str, *, follows: _Location | property | None = None):
        if isinstance(follows, str):
            raise TypeError(
                f"follows takes the location itself, not its name: write "
                f"follows={follows}, without quotes."
            )
        self.space_name = space
        self.follows = follows
        self._installed: property | None = None
        self.name: str | None = None
        self._key: str | None = None
        # The locations that follow this one, per agent class
        self._followers_cache: dict[type, tuple[CellLocation, ...]] = {}

    def __set_name__(self, owner, name: str) -> None:
        self.name = name
        self._key = f"_mesa_location_{name}"
        # Only the class's own declarations; Agent.__init_subclass__, which runs
        # after this, adds those of the bases
        owner._mesa_declared_locations = (
            *owner.__dict__.get("_mesa_declared_locations", ()),
            self,
        )
        # Agents without the location read None from the class
        setattr(owner, self._key, None)
        # Kept so that a subclass can follow this location as Base.name
        self._installed = self._property()
        setattr(owner, name, self._installed)

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
        if cls not in self._followers_cache:
            # A class whose __init_subclass__ chain skipped Agent's
            _declarations(cls)
        return self._followers_cache[cls]

    def _primary(self, agent: Agent) -> _Location:
        """Return the location this one follows."""
        return _declarations(type(agent))[2][self.name]

    def _check_declaration(self, cls: type) -> None:
        """Check the locations of an agent class when it is defined."""
        _declarations(cls)

    def _remove(self, agent: Agent) -> None:
        """Leave the space when the agent is removed; followers leave with their primary."""
        if self.follows is None:
            self._set(agent, None)


def _declarations(cls: type) -> tuple[dict, dict, dict]:
    """Return the locations of an agent class by name, their followers, and primaries.

    Checks that every followed location is a location of the class, and is not
    itself a follower. Runs when the class is defined, or on first use if
    Agent.__init_subclass__ didn't run for it, and records each
    location's followers for the class.
    """
    cached = cls.__dict__.get("_mesa_location_declarations")
    if cached is not None:
        return cached

    locations = {loc.name: loc for loc in getattr(cls, "_mesa_locations", ())}
    followers: dict[str, list[CellLocation]] = {}
    primaries: dict[str, _Location] = {}
    for location in locations.values():
        target = location.follows
        if target is None:
            continue
        # A location from the class body, or Base.name, which is its property
        primary = next(
            (
                loc
                for loc in locations.values()
                if loc is target or loc._installed is target
            ),
            None,
        )
        where = f"{cls.__name__}.{location.name}"
        if primary is None:
            name = getattr(target, "name", None)
            if name in locations:
                raise TypeError(
                    f"{where} follows {name}, which {cls.__name__} overrides; "
                    f"declare {location.name} again in {cls.__name__} to follow "
                    "the new one."
                )
            raise TypeError(
                f"{where} follows a location that is not one of {cls.__name__}."
            )
        if primary.follows is not None:
            root = getattr(primary.follows, "name", "another location")
            raise TypeError(
                f"{where} follows {primary.name}, which follows {root} itself; "
                f"follow {root} directly."
            )
        if primary.space_name == location.space_name:
            raise TypeError(
                f"{where} follows {primary.name} in the same space, "
                f"model.{location.space_name}; an agent has one cell per space."
            )
        followers.setdefault(primary.name, []).append(location)
        primaries[location.name] = primary

    followers = {k: tuple(v) for k, v in followers.items()}
    for location in locations.values():
        location._followers_cache[cls] = followers.get(location.name, ())
    declarations = (locations, followers, primaries)
    cls._mesa_location_declarations = declarations
    return declarations


# ----------------------------------------------------------------------
# moving followers
# ----------------------------------------------------------------------
def _resolve(agent: Agent, followers, point, pending=()) -> list:
    """Find each follower's cell for a point. Raises if one has none."""
    state = agent.__dict__
    targets = list(pending)
    for follower in followers:
        if state.get(follower._key) is None:
            follower._check_alignment(agent)
            follower._check_unique(agent, follower._space(agent), targets)
        try:
            cell = follower._resolve(agent, point)
        except ValueError as e:
            raise ValueError(
                f"{type(agent).__name__}.{follower.name} follows "
                f"{follower._primary(agent).name}, but model.{follower.space_name} has no "
                f"cell at {np.asarray(point).tolist()}, so the move is refused."
            ) from e
        targets.append((follower, cell))
    return targets


def _enter(agent: Agent, targets) -> None:
    """Enter all new cells, or none of them if one is full."""
    state = agent.__dict__
    entered = []
    try:
        for location, cell in targets:
            old = state.get(location._key)
            if cell is not old:
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


# The kinds of primary and follower space that are known to line up
_SUPPORTED = frozenset(
    {("point", "grid"), ("point", "hex"), ("point", "voronoi"), ("grid", "grid")}
)


def _kind(space) -> str:
    """Return the kind of a discrete space, as used in ``_SUPPORTED``."""
    if isinstance(space, Network):
        return "network"
    if isinstance(space, HexGrid):
        return "hex"
    if isinstance(space, Grid):
        return "grid"
    if isinstance(space, VoronoiGrid):
        return "voronoi"
    return type(space).__name__


def _bounds(space) -> list[tuple[float, float]] | None:
    """Return the lower and upper bound of a space along each axis, if it has them.

    None for spaces that don't place cells by ``cell_size`` and ``origin``, such
    as a HexGrid. Plain floats, since this runs whenever an agent first joins and
    numpy is many times slower for a handful of values.
    """
    if isinstance(space, Grid):
        if isinstance(space, HexGrid):
            return None
        origin, cell_size = space._mapping
        return [
            (o, o + n * s)
            for o, s, n in zip(origin, cell_size, space.dimensions, strict=True)
        ]
    dimensions = getattr(space, "dimensions", None)
    if isinstance(dimensions, np.ndarray) and dimensions.ndim == 2:
        return [(low, high) for low, high in dimensions.tolist()]
    return None


def _to_point(space: ContinuousSpace, value) -> np.ndarray:
    """Return a value as a point in a space, wrapped by a torus; raise outside it."""
    point = np.array(value, dtype=float)
    if point.shape != (space.ndims,):
        raise ValueError(
            f"Expected a point with {space.ndims} coordinates, got {value!r}."
        )
    if not space.in_bounds(point):
        # NaN and inf are never in bounds, and a torus would wrap them to NaN.
        if not np.isfinite(point).all():
            raise ValueError(f"point {value} has a NaN or infinite coordinate")
        if not space.torus:
            raise ValueError(f"point {value} is outside the bounds of the space")
        point = space.torus_correct(point)
    return point


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
    ``None`` takes it out. Reading returns the cell, or ``None``. A cell of another
    space raises ``ValueError``.

    Args:
        space: the name of the model attribute holding the space, such as
            ``"ground"`` for ``model.ground``.
        follows: another location of the same agent, declared above this one,
            or ``Base.name`` for one declared on a base class. The agent's cell is
            then always the cell of ``space`` at that location, and assigning a
            cell moves the other location to its center, or raises
            ``ValueError`` if that center leads to another cell. Grid, hex and
            Voronoi cells can follow a point, and a grid cell a grid cell no
            larger than itself. A Network can't take part (``TypeError``), other
            pairs warn, and spaces that wrap in different places raise
            ``ValueError`` on the first move.

    Notes:
        Without ``follows``, this is what ``CellAgent.cell`` does, under a name of
        your choosing, so an agent can have cells in several spaces. An agent has
        at most one cell per space: placing it in a space where it already has a
        cell, through another location or as a ``CellAgent``, raises
        ``ValueError``, since it would count twice there.
    """

    def _property(self) -> property:
        key = self._key
        followers_cache = self._followers_cache
        set_location = self._set
        space_name = self.space_name
        check_space = self._check_space
        check_unique = self._check_unique

        def set_cell(agent: Agent, cell: Cell | None) -> None:
            # A cell that nothing follows, as in CellAgent
            if cell is not None and followers_cache.get(type(agent)) == ():
                state = agent.__dict__
                old = state.get(key)
                if cell is old:
                    return
                space = getattr(agent.model, space_name, None)
                if space is None or space._cells.get(cell.coordinate) is not cell:
                    check_space(agent, cell)  # raises
                if old is None:
                    check_unique(agent, space)
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
        self._check_space(agent, cell)
        if old is None:
            self._check_unique(agent, self._space(agent))
        targets = _resolve(agent, followers, cell.position, [(self, cell)])
        _enter(agent, targets)
        _commit(agent, targets)

    def _set_follower(self, agent: Agent, cell: Cell | None) -> None:
        """Move the primary location to the center of the cell."""
        primary = self._primary(agent)
        placed = agent.__dict__.get(self._key) is not None
        if cell is None:
            if not placed:
                return  # already out, for instance after Agent.remove
            raise ValueError(
                f"{type(agent).__name__}.{self.name} follows {primary.name}; set "
                f"{primary.name} to None to take the agent out of both."
            )
        self._check_space(agent, cell)
        if not placed:
            # Before the round trip below, so a misdeclared pair gets its own error
            self._check_alignment(agent)
            self._check_unique(agent, self._space(agent))
        # The primary moves to the point it would store for this cell's center: a
        # cell there, or the point wrapped by a torus. This location then follows
        # that point, which must lead back to this cell.
        if isinstance(primary, CellLocation):
            target = primary._resolve(agent, cell.position)
            point = target.position
        else:
            target = point = _to_point(primary._space(agent), cell.position)
        if self._resolve(agent, point) is not cell:
            raise ValueError(
                f"{type(agent).__name__}.{self.name} follows {primary.name} in "
                f"model.{primary.space_name}, where cell {cell.coordinate} of "
                f"model.{self.space_name} can't be reached: its center leads to "
                "another cell."
            )
        primary._set(agent, target)

    def _check_unique(self, agent: Agent, space, pending=()) -> None:
        """Refuse a second cell in a space: the agent would count twice there."""
        state = agent.__dict__
        others = [
            (other.name, state.get(other._key))
            for other in agent._mesa_locations
            if other is not self and isinstance(other, CellLocation)
        ]
        others.extend(
            (other.name, cell) for other, cell in pending if other is not self
        )
        if "cell" not in {location.name for location in agent._mesa_locations}:
            others.append(("cell", getattr(agent, "cell", None)))  # as a CellAgent
        for name, cell in others:
            if cell is not None and space._cells.get(cell.coordinate) is cell:
                raise ValueError(
                    f"{type(agent).__name__}.{self.name} would put the agent in "
                    f"model.{self.space_name}, where it already has {name}: an "
                    "agent has at most one cell in a discrete space."
                )

    def _check_space(self, agent: Agent, cell: Cell) -> None:
        """Refuse a cell that is not in this location's space."""
        if self._space(agent)._cells.get(cell.coordinate) is not cell:
            raise ValueError(
                f"{type(agent).__name__}.{self.name} takes cells of model."
                f"{self.space_name}, got a cell of another space."
            )

    def _check_alignment(self, agent: Agent) -> None:
        """Check once per agent, before it first gets a cell here, that the spaces fit.

        A Network has no shared coordinates, so it raises. Pairs of spaces not in
        ``_SUPPORTED`` warn, since nothing below checks them. The rest would give
        wrong cells without raising by themselves: a follower finer than its
        primary cell can only reach the cells at the centers of the primary's
        cells, a torus grid that doesn't match its primary's space wraps points
        into cells on the other side, and a torus primary wraps a cell outside it
        back into its space. Partial grids in a bounded space fail loudly on their
        own and stay allowed.
        """
        primary = self._primary(agent)
        space = self._space(agent)
        outer = primary._space(agent)
        pair = (
            "point" if isinstance(primary, ContinuousLocation) else _kind(outer),
            _kind(space),
        )
        if "network" in pair:
            raise TypeError(
                f"{type(agent).__name__}.{self.name} follows {primary.name}, but a "
                "Network can't follow or be followed: a point is not on a node, and "
                "a node's position is only where it is drawn."
            )
        if pair not in _SUPPORTED:
            warnings.warn(
                f"{type(agent).__name__}.{self.name}: a {type(space).__name__} "
                f"following a {type(outer).__name__} isn't a supported combination, "
                "so nothing checks that their cells line up.",
                UserWarning,
                skip_file_prefixes=(__file__,),
            )
            return
        if pair[1] == "hex" and getattr(space, "torus", False):
            raise ValueError(
                f"{type(agent).__name__}.{self.name} follows {primary.name}, but "
                f"model.{self.space_name} is a torus HexGrid. A HexGrid has no cell "
                "size or origin to check where it wraps, so it can only follow when "
                "it isn't a torus."
            )
        if pair == ("grid", "grid") and any(
            size < outer_size * (1 - 1e-9)
            for size, outer_size in zip(
                space._mapping[1], outer._mapping[1], strict=True
            )
        ):
            raise ValueError(
                f"{type(agent).__name__}.{self.name} follows {primary.name}, "
                f"but model.{primary.space_name} has larger cells than model."
                f"{self.space_name}. The primary must be the most detailed "
                "location: let the finer grid be followed instead."
            )
        torus = getattr(space, "torus", False)
        outer_torus = getattr(outer, "torus", False)
        if not (torus or outer_torus):
            return
        bounds, outer_bounds = _bounds(space), _bounds(outer)
        if bounds is None or outer_bounds is None:
            return
        axes = list(zip(bounds, outer_bounds, strict=True))
        # bounds is not None, so space is a grid with a cell size
        tolerance = 1e-9 * min(space._mapping[1])
        where = (
            f"{type(agent).__name__}.{self.name} follows {primary.name}, but "
            f"model.{self.space_name}"
        )
        if torus:
            matches = all(
                abs(low - outer_low) <= tolerance
                and abs(high - outer_high) <= tolerance
                for (low, high), (outer_low, outer_high) in axes
            )
            if not (outer_torus and matches):
                raise ValueError(
                    f"{where} is a torus that doesn't match model."
                    f"{primary.space_name}: a torus grid must wrap where its "
                    "primary's space wraps, with the same bounds and torus=True."
                )
        elif any(
            low < outer_low - tolerance or high > outer_high + tolerance
            for (low, high), (outer_low, outer_high) in axes
        ):
            raise ValueError(
                f"{where} extends beyond the torus model.{primary.space_name}, "
                "which would wrap a cell outside it to another place."
            )

    def _resolve(self, agent: Agent, point) -> Cell:
        """Return the cell of this location's space at a point."""
        return self._space(agent).find_nearest_cell(point)


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

        An agent has at most one point in a space: placing it in a space it is
        already in, through another location or as a ``ContinuousSpaceAgent``,
        raises.
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

        if handle is None:
            space = self._space(agent)
            # The space keeps one row per agent: a second point would corrupt it
            if space._index_of(agent) is not None:
                raise ValueError(
                    f"{type(agent).__name__} is already in model.{self.space_name}, "
                    f"so {self.name} cannot place it there too: an agent has at most "
                    "one point in a continuous space."
                )
        else:
            space = handle.space
        point = _to_point(space, value)

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
