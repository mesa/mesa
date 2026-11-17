"""Tests for locations declared on agents."""

import pickle
import random

import networkx as nx
import numpy as np
import pytest

from mesa import Agent, Model
from mesa.discrete_space import CellAgent, HexGrid, Network, OrthogonalMooreGrid
from mesa.exceptions import CellFullException
from mesa.experimental.continuous_space import ContinuousSpace, ContinuousSpaceAgent
from mesa.experimental.locations import CellLocation, ContinuousLocation


class Citizen(Agent):
    """An agent with independent locations in several spaces."""

    home = CellLocation("ground")
    work = CellLocation("districts")
    view = ContinuousLocation("opinion")
    lifestyle = ContinuousLocation("traits")


class Bird(Agent):
    """An agent whose ground cell follows its position in the air."""

    position = ContinuousLocation("air")
    cell = CellLocation("ground", follows=position)


class Surveyor(Agent):
    """An agent whose cells in two grids follow its position."""

    position = ContinuousLocation("air")
    cell = CellLocation("ground", follows=position)
    district = CellLocation("districts", follows=position)


class Walker(Agent):
    """An agent on a grid whose district follows its cell."""

    cell = CellLocation("ground")
    district = CellLocation("districts", follows=cell)


@pytest.fixture
def model():
    """A model with stacked physical spaces and two abstract spaces."""
    model = Model(rng=42)
    model.air = ContinuousSpace([[0, 100], [0, 100]], random=model.random)
    model.ground = OrthogonalMooreGrid((50, 50), cell_size=2.0, random=model.random)
    model.districts = OrthogonalMooreGrid((5, 5), cell_size=20.0, random=model.random)
    model.opinion = ContinuousSpace([[-1, 1], [-1, 1]], random=model.random)
    model.traits = ContinuousSpace([[0, 1], [0, 1]], torus=True, random=model.random)
    return model


# ----------------------------------------------------------------------
# CellLocation
# ----------------------------------------------------------------------
def test_cell_location_moves_and_leaves(model):
    """Assigning a cell joins it, a new cell moves, None leaves."""
    citizen = Citizen(model)
    assert citizen.home is None

    citizen.home = model.ground[1, 1]
    assert citizen.home is model.ground[1, 1]
    assert citizen in model.ground[1, 1].agents

    citizen.home = model.ground[2, 2]
    assert citizen not in model.ground[1, 1].agents
    assert citizen in model.ground[2, 2].agents

    citizen.home = None
    assert citizen.home is None
    assert citizen not in model.ground[2, 2].agents


def test_cell_location_full_cell_keeps_agent_in_place(model):
    """A move into a full cell raises and leaves the agent where it was."""
    grid = model.ground = OrthogonalMooreGrid((3, 3), capacity=1, random=model.random)
    blocker, citizen = Citizen(model), Citizen(model)
    blocker.home = grid[1, 1]
    citizen.home = grid[0, 0]

    with pytest.raises(CellFullException):
        citizen.home = grid[1, 1]
    assert citizen.home is grid[0, 0]
    assert citizen in grid[0, 0].agents


def test_one_cell_per_agent_in_a_discrete_space(model):
    """A second cell in a space raises, so an agent never counts twice there."""

    class Resident(Agent):
        home = CellLocation("ground")
        work = CellLocation("ground")

    resident = Resident(model)
    resident.home = model.ground[1, 1]
    for target in (model.ground[1, 1], model.ground[9, 9]):
        with pytest.raises(ValueError, match="at most one cell"):
            resident.work = target
        assert resident.work is None
        assert model.ground[1, 1].agents == [resident]
        assert resident not in model.ground[9, 9].agents

    # the same for a CellAgent's own cell
    class Settler(CellAgent):
        home = CellLocation("ground")

    settler = Settler(model)
    settler.cell = model.ground[2, 2]
    with pytest.raises(ValueError, match="already has cell"):
        settler.home = model.ground[2, 2]
    assert model.ground[2, 2].agents == [settler]


@pytest.mark.parametrize("second_space", ["ground", "alias"])
def test_followers_cannot_enter_the_same_space_together(model, second_space):
    """Pending followers are checked before any memberships change, even via aliases."""
    model.alias = model.ground

    class Twice(Agent):
        position = ContinuousLocation("air")
        first = CellLocation("ground", follows=position)
        second = CellLocation(second_space, follows=position)

    blockers = [Citizen(model) for _ in range(2)]
    for blocker in blockers:
        blocker.home = model.ground[0, 0]
    agent = Twice(model)

    with pytest.raises(ValueError, match="already has first"):
        agent.position = [1.0, 1.0]

    assert agent.position is agent.first is agent.second is None
    assert model.air.active_agents == []
    assert model.ground[0, 0].agents == blockers


def test_cell_primary_and_follower_cannot_enter_the_same_space(model):
    """An aliased grid cannot hold both a pending primary and its follower."""
    model.alias = model.ground

    class Twice(Agent):
        primary = CellLocation("ground")
        follower = CellLocation("alias", follows=primary)

    agent = Twice(model)
    with pytest.raises(ValueError, match="already has primary"):
        agent.primary = model.ground[0, 0]

    assert agent.primary is agent.follower is None
    assert model.ground[0, 0].agents == []


def test_cell_location_refuses_cells_of_another_space(model):
    """A cell of another space raises and leaves the agent where it was."""
    citizen = Citizen(model)
    citizen.home = model.ground[1, 1]

    with pytest.raises(ValueError, match="takes cells of model"):
        citizen.home = model.districts[0, 0]
    assert citizen.home is model.ground[1, 1]
    assert citizen not in model.districts[0, 0].agents


# ----------------------------------------------------------------------
# ContinuousLocation
# ----------------------------------------------------------------------
def test_continuous_location_joins_moves_and_leaves(model):
    """The first point joins the space, reads are copies, += moves, None leaves."""
    citizen = Citizen(model)
    assert citizen.view is None
    citizen.view = [0.0, 0.0]
    assert citizen in model.opinion.active_agents

    saved = citizen.view
    citizen.view += np.array([0.5, 0.25])
    np.testing.assert_array_equal(saved, [0.0, 0.0])
    np.testing.assert_array_equal(citizen.view, [0.5, 0.25])
    citizen.view[0] = -0.9  # changes a copy; the agent doesn't move
    np.testing.assert_array_equal(citizen.view, [0.5, 0.25])

    citizen.view = None
    assert citizen.view is None
    assert citizen not in model.opinion.active_agents


def test_continuous_location_bounds_and_torus(model):
    """Points outside a bounded space raise; a torus wraps them."""
    citizen = Citizen(model)
    citizen.view = [0.0, 0.0]
    with pytest.raises(ValueError):
        citizen.view = [2.0, 0.0]
    np.testing.assert_array_equal(citizen.view, [0.0, 0.0])
    with pytest.raises(ValueError):
        citizen.view = [0.0, 0.0, 0.0]

    citizen.lifestyle = [1.25, -0.25]
    np.testing.assert_allclose(citizen.lifestyle, [0.25, 0.75])


def test_continuous_locations_in_several_spaces(model):
    """One agent can be in several continuous spaces, next to other agents."""
    citizens = [Citizen(model) for _ in range(5)]
    for i, citizen in enumerate(citizens):
        citizen.view = [i / 10, 0.0]
        citizen.lifestyle = [0.5, i / 10]

    # removing one agent swaps rows; the others must keep their positions
    citizens[1].view = None
    for i, citizen in enumerate(citizens):
        if i != 1:
            np.testing.assert_allclose(citizen.view, [i / 10, 0.0])
        np.testing.assert_allclose(citizen.lifestyle, [0.5, i / 10])


def test_continuous_location_next_to_continuous_space_agent(model):
    """A ContinuousSpaceAgent can also hold a location in another space."""

    class Boid(ContinuousSpaceAgent):
        view = ContinuousLocation("opinion")

    others = [ContinuousSpaceAgent(model.opinion, model) for _ in range(3)]
    for other in others:
        other.position = [0.9, 0.9]
    boid = Boid(model.air, model)
    boid.position = [50.0, 50.0]
    boid.view = [0.1, 0.1]

    np.testing.assert_array_equal(boid.position, [50.0, 50.0])
    np.testing.assert_array_equal(boid.view, [0.1, 0.1])
    distances, _ = model.opinion.calculate_distances(
        np.array([0.0, 0.0]), agents=[boid]
    )
    np.testing.assert_allclose(distances, [np.sqrt(0.02)])

    boid.remove()
    assert boid not in model.opinion.active_agents
    assert boid not in model.air.active_agents
    for other in others:
        np.testing.assert_array_equal(other.position, [0.9, 0.9])


def test_one_point_per_agent_in_a_continuous_space(model):
    """A second location in the same space raises and leaves the first intact."""

    class Twice(Agent):
        first = ContinuousLocation("opinion")
        second = ContinuousLocation("opinion")

    agent = Twice(model)
    agent.first = [0.1, 0.1]
    with pytest.raises(ValueError, match=r"already in model.opinion"):
        agent.second = [0.2, 0.2]
    assert agent.second is None
    assert model.opinion.active_agents == [agent]

    class Viewer(ContinuousSpaceAgent):
        view = ContinuousLocation("opinion")

    viewer = Viewer(model.opinion, model)
    viewer.position = [0.5, 0.5]
    with pytest.raises(ValueError, match=r"already in model.opinion"):
        viewer.view = [0.5, 0.5]


def test_missing_space_attribute_names_it(model):
    """A location naming a space the model lacks says which location."""

    class Lost(Agent):
        position = ContinuousLocation("nowhere")

    with pytest.raises(AttributeError, match=r"Lost.position lives in model.nowhere"):
        Lost(model).position = [0.0, 0.0]


# ----------------------------------------------------------------------
# queries and removal
# ----------------------------------------------------------------------
def test_queries_exclude_the_asking_agent(model):
    """exclude= leaves an agent out of radius and nearest-neighbor queries."""
    citizens = [Citizen(model) for _ in range(3)]
    for i, citizen in enumerate(citizens):
        citizen.view = [i / 10, 0.0]
    me = citizens[0]

    agents, _ = model.opinion.get_agents_in_radius(me.view, 0.5, exclude=me)
    assert me not in agents
    assert len(agents) == 2

    agents, _ = model.opinion.get_k_nearest_agents(me.view, k=1, exclude=me)
    assert agents == [citizens[1]]

    agents, _ = model.opinion.get_agents_in_radius(me.view, np.inf, exclude=me)
    assert me not in agents

    # an agent that is not in the space is ignored
    stranger = Citizen(model)
    agents, _ = model.opinion.get_agents_in_radius(me.view, 0.5, exclude=stranger)
    assert len(agents) == 3

    with pytest.warns(UserWarning):
        agents, _ = model.opinion.get_k_nearest_agents(me.view, k=3, exclude=me)
    assert len(agents) == 2
    assert me not in agents


def test_remove_leaves_every_location(model):
    """Removing an agent takes it out of every space it was placed in."""
    citizen = Citizen(model)
    citizen.home = model.ground[1, 1]
    citizen.work = model.districts[2, 2]
    citizen.view = [0.0, 0.0]

    citizen.remove()
    assert citizen not in model.ground[1, 1].agents
    assert citizen not in model.districts[2, 2].agents
    assert citizen not in model.opinion.active_agents


class Lefty(Agent):
    """An agent with one location, to combine with Righty."""

    left = ContinuousLocation("opinion")


class Righty(Agent):
    """An agent with one location, to combine with Lefty."""

    right = ContinuousLocation("traits")


@pytest.mark.parametrize("bases", [(Lefty, Righty), (Righty, Lefty)])
def test_multiple_inheritance_collects_every_base(model, bases):
    """A class without locations of its own has those of all its bases."""
    combined = type("Combined", bases, {})
    agent = combined(model)
    agent.left = [0.5, 0.5]
    agent.right = [0.5, 0.5]
    assert sorted(loc.name for loc in combined._mesa_locations) == ["left", "right"]

    agent.remove()
    assert agent.left is None
    assert agent.right is None
    assert len(model.opinion.active_agents) == 0
    assert len(model.traits.active_agents) == 0


def test_locations_from_a_diamond_and_a_mixin(model):
    """Each location counts once, also when declared on a class that isn't an Agent."""

    class Sees:
        sight = CellLocation("ground")

    class Both(Lefty, Righty, Sees):
        pass

    class Twice(Both, Lefty):
        pass

    names = [location.name for location in Twice._mesa_locations]
    assert sorted(names) == ["left", "right", "sight"]

    agent = Twice(model)
    agent.sight = model.ground[1, 1]
    agent.remove()
    assert agent not in model.ground[1, 1].agents


def test_mixin_that_skips_init_subclass(model):
    """A class whose __init_subclass__ chain skips Agent's still moves its followers."""

    class Quiet:
        def __init_subclass__(cls, **kwargs):
            pass

    class Robin(Quiet, Bird):
        pass

    class Hermit(Quiet, Citizen):
        pass

    robin = Robin(model)
    robin.position = [11.0, 21.0]
    assert robin.cell is model.ground[5, 10]
    hermit = Hermit(model)
    hermit.home = model.ground[0, 0]
    assert hermit in model.ground[0, 0].agents


# ----------------------------------------------------------------------
# follows
# ----------------------------------------------------------------------
def test_follower_tracks_position(model):
    """Writing the position updates the cell; writing the cell moves the position."""
    bird = Bird(model)
    bird.position = [11.0, 21.0]
    assert bird.cell is model.ground[5, 10]

    bird.position += np.array([2.0, 0.0])
    assert bird.cell is model.ground[6, 10]
    assert bird not in model.ground[5, 10].agents

    bird.cell = model.ground[0, 0]
    np.testing.assert_array_equal(bird.position, [1.0, 1.0])
    assert bird in model.ground[0, 0].agents

    # a follower written first places the agent at the cell center
    lander = Bird(model)
    lander.cell = model.ground[2, 3]
    np.testing.assert_array_equal(lander.position, [5.0, 7.0])


def test_full_cell_rejects_the_whole_move(model):
    """If the follower's cell is full, neither location changes."""
    model.ground = OrthogonalMooreGrid(
        (50, 50), cell_size=2.0, capacity=1, random=model.random
    )
    blocker = Bird(model)
    blocker.position = [21.0, 21.0]
    bird = Bird(model)
    bird.position = [1.0, 1.0]

    with pytest.raises(CellFullException):
        bird.position = [21.5, 21.5]
    np.testing.assert_array_equal(bird.position, [1.0, 1.0])
    assert bird.cell is model.ground[0, 0]

    with pytest.raises(CellFullException):
        bird.cell = model.ground[10, 10]
    np.testing.assert_array_equal(bird.position, [1.0, 1.0])

    # a first placement that fails leaves the agent out of both spaces
    late = Bird(model)
    with pytest.raises(CellFullException):
        late.position = [21.0, 21.0]
    assert late.position is None
    assert late not in model.air.active_agents


def test_point_without_cell_rejects_the_move(model):
    """A grid that covers part of the air rejects points outside it."""
    model.districts = OrthogonalMooreGrid((2, 2), cell_size=20.0, random=model.random)
    surveyor = Surveyor(model)
    surveyor.position = [10.0, 10.0]
    with pytest.raises(ValueError, match="district follows position"):
        surveyor.position = [90.0, 90.0]
    np.testing.assert_array_equal(surveyor.position, [10.0, 10.0])
    assert surveyor.cell is model.ground[5, 5]


def test_follower_of_a_cell(model):
    """A district can follow a grid cell."""
    walker = Walker(model)
    walker.cell = model.ground[12, 3]
    assert walker.district is model.districts[1, 0]

    walker.district = model.districts[4, 4]
    assert walker.cell is model.ground[45, 45]


class Zoomer(Agent):
    """An agent on a coarse grid with a cell in a fine grid that follows it."""

    district = CellLocation("districts")
    cell = CellLocation("ground", follows=district)


def test_follower_finer_than_its_cell_primary_is_refused(model):
    """A primary must be at least as detailed as its follower; nothing changes."""
    zoomer = Zoomer(model)
    with pytest.raises(ValueError, match="larger cells"):
        zoomer.district = model.districts[1, 1]
    assert zoomer.district is None
    assert zoomer.cell is None
    assert zoomer not in model.districts[1, 1].agents

    # also through the follower, whichever cell is written
    for target in (model.ground[15, 15], model.ground[2, 2]):
        with pytest.raises(ValueError, match="larger cells"):
            zoomer.cell = target
        assert zoomer.district is None
        assert zoomer not in target.agents


def test_follower_of_a_cell_with_shifted_origins(model):
    """A cell that its primary maps back to another cell raises, changing nothing."""
    # Same cell size, offset by half a cell: each offset cell center is on a
    # ground cell boundary, so most offset cells can't be reached
    model.offset = OrthogonalMooreGrid(
        (49, 49), cell_size=2.0, origin=(1.0, 1.0), random=model.random
    )

    class Shifted(Agent):
        cell = CellLocation("ground")
        shifted = CellLocation("offset", follows=cell)

    agent = Shifted(model)
    agent.cell = model.ground[5, 5]
    before = agent.shifted

    with pytest.raises(ValueError, match="can't be reached"):
        agent.shifted = model.offset[20, 20]
    assert agent.cell is model.ground[5, 5]
    assert agent.shifted is before
    assert agent not in model.offset[20, 20].agents


def torus_pair(air_torus, shape, grid_torus, origin=(0, 0)):
    """Return a model with a 100 by 100 air space and a ground grid of 2 by 2 cells."""
    model = Model(rng=42)
    model.air = ContinuousSpace(
        [[0, 100], [0, 100]], torus=air_torus, random=model.random
    )
    model.ground = OrthogonalMooreGrid(
        shape, cell_size=2.0, origin=origin, torus=grid_torus, random=model.random
    )
    return model


@pytest.mark.parametrize(
    ("air_torus", "shape", "grid_torus", "origin", "message"),
    [
        (True, (40, 40), True, (0, 0), "doesn't match"),  # wraps too early
        (True, (60, 60), True, (0, 0), "doesn't match"),  # wraps too late
        (True, (50, 50), True, (10, 0), "doesn't match"),  # wraps elsewhere
        (False, (50, 50), True, (0, 0), "doesn't match"),  # air doesn't wrap
        (True, (60, 60), False, (0, 0), "extends beyond"),  # cells outside the air
    ],
)
def test_torus_mismatch_raises_on_first_move(
    air_torus, shape, grid_torus, origin, message
):
    """A torus mismatch would silently give a wrong cell, so it raises instead."""
    model = torus_pair(air_torus, shape, grid_torus, origin)
    bird = Bird(model)
    with pytest.raises(ValueError, match=message):
        bird.position = [90.0, 90.0]
    assert bird.position is None
    assert bird.cell is None
    assert len(model.air.active_agents) == 0
    assert all(cell.is_empty for cell in model.ground.all_cells)

    with pytest.raises(ValueError, match=message):
        bird.cell = model.ground[1, 1]
    assert bird.position is None


@pytest.mark.parametrize(
    ("air_torus", "shape", "grid_torus", "origin"),
    [
        (True, (50, 50), True, (0, 0)),  # the same torus
        (True, (50, 50), False, (0, 0)),  # a bounded grid covering a torus
        (True, (10, 10), False, (40, 40)),  # an island in a torus
        (False, (40, 40), False, (0, 0)),  # a partial grid, as before
    ],
)
def test_aligned_torus_setups_are_allowed(air_torus, shape, grid_torus, origin):
    """Spaces that wrap in the same place, or not at all, can follow each other."""
    model = torus_pair(air_torus, shape, grid_torus, origin)
    bird = Bird(model)
    bird.position = [45.0, 45.0]
    assert bird.cell is model.ground.find_nearest_cell([45.0, 45.0])


def test_torus_check_between_cell_locations(model):
    """A cell that follows a cell is checked the same way."""
    model.ground = OrthogonalMooreGrid(
        (50, 50), cell_size=2.0, torus=True, random=model.random
    )
    model.districts = OrthogonalMooreGrid(
        (5, 5), cell_size=20.0, torus=True, random=model.random
    )
    walker = Walker(model)
    walker.cell = model.ground[0, 0]
    assert walker.district is model.districts[0, 0]

    model.districts = OrthogonalMooreGrid(
        (4, 4), cell_size=20.0, torus=True, random=model.random
    )
    walker = Walker(model)
    with pytest.raises(ValueError, match="doesn't match"):
        walker.cell = model.ground[0, 0]
    assert walker.cell is None


def test_follower_declaration_errors():
    """A follower takes a location of its own class, checked when it is defined."""
    with pytest.raises(TypeError, match="without quotes"):
        CellLocation("ground", follows="position")

    with pytest.raises(TypeError, match="not one of Orphan"):

        class Orphan(Agent):
            cell = CellLocation("ground", follows=Bird.position)

    with pytest.raises(TypeError, match="follow position directly"):

        class Chain(Agent):
            position = ContinuousLocation("air")
            cell = CellLocation("ground", follows=position)
            district = CellLocation("districts", follows=cell)

    with pytest.raises(TypeError, match="overrides"):

        class Duck(Bird):
            position = ContinuousLocation("water")

    with pytest.raises(TypeError, match="same space"):

        class Shadow(Agent):
            home = CellLocation("ground")
            shadow = CellLocation("ground", follows=home)


def test_follower_of_a_location_on_a_base_class(model):
    """Base.name follows a location declared on a base class."""

    class Flyer(Agent):
        position = ContinuousLocation("air")

    class Lander(Flyer):
        cell = CellLocation("ground", follows=Flyer.position)

    lander = Lander(model)
    lander.position = [21.3, 40.2]
    assert lander.cell is model.ground[10, 20]


def test_follower_rules(model):
    """A follower cannot be cleared on its own, or take a cell of another space."""
    bird = Bird(model)
    bird.position = [1.0, 1.0]
    with pytest.raises(ValueError, match="set position to None"):
        bird.cell = None
    with pytest.raises(ValueError, match="another space"):
        bird.cell = model.districts[0, 0]

    bird.position = None
    assert bird.cell is None
    assert bird not in model.ground[0, 0].agents
    bird.cell = None  # already out: nothing to do


def test_cell_agent_with_a_following_cell_can_be_removed(model):
    """CellAgent.remove clears cell after the agent left; that is fine."""

    class Hybrid(CellAgent):
        position = ContinuousLocation("air")
        cell = CellLocation("ground", follows=position)

    hybrid = Hybrid(model)
    hybrid.position = [2.5, 2.5]
    hybrid.remove()
    assert hybrid not in model.air.active_agents
    assert hybrid not in model.ground[1, 1].agents


def test_hex_followers_keep_the_assigned_cell_or_raise(model):
    """A torus HexGrid can't follow; a bounded one under a torus raises if unreachable."""

    class Bee(Agent):
        position = ContinuousLocation("air")
        comb = CellLocation("hexes", follows=position)

    model.air = ContinuousSpace([[0, 10], [0, 10]], torus=True, random=model.random)
    model.hexes = HexGrid((6, 6), torus=True, random=model.random)
    with pytest.raises(ValueError, match="torus HexGrid"):
        Bee(model).position = [9.5, 9.5]

    # this hex's center lies outside the air, which wraps it to another hex
    model.hexes = HexGrid((10, 10), random=model.random)
    bee = Bee(model)
    with pytest.raises(ValueError, match="can't be reached"):
        bee.comb = model.hexes[9, 9]
    assert bee.position is None
    assert bee not in model.hexes[9, 9].agents


def test_network_cannot_follow_or_be_followed(model):
    """A network has no shared coordinates, on either side; nothing changes."""
    model.roads = Network(nx.path_graph(3), random=model.random)

    class Driver(Agent):
        position = ContinuousLocation("air")
        node = CellLocation("roads", follows=position)

    class Commuter(Agent):
        node = CellLocation("roads")
        cell = CellLocation("ground", follows=node)

    with pytest.raises(TypeError, match="Network"):
        Driver(model).position = [1.0, 1.0]

    commuter = Commuter(model)
    with pytest.raises(TypeError, match="Network"):
        commuter.node = model.roads[0]
    assert commuter.node is None
    assert commuter not in model.roads[0].agents


def test_unsupported_pair_of_spaces_warns_and_still_follows(model):
    """A pair not known to line up warns, and still follows."""
    model.hexes = HexGrid((10, 10), random=model.random)

    class Hexer(Agent):
        hex = CellLocation("hexes")
        cell = CellLocation("ground", follows=hex)

    hexer = Hexer(model)
    with pytest.warns(UserWarning, match="supported combination"):
        hexer.hex = model.hexes[2, 3]
    assert hexer.cell is model.ground.find_nearest_cell(model.hexes[2, 3].position)


def test_followers_stay_consistent_under_random_moves():
    """Random moves, some rejected, never leave a follower out of step."""
    rng = random.Random(7)
    model = Model(rng=7)
    model.air = ContinuousSpace([[0, 100], [0, 100]], random=model.random)
    model.ground = OrthogonalMooreGrid(
        (25, 25), cell_size=4.0, capacity=2, random=model.random
    )
    # covers only the lower left of the air, so some moves have no district
    model.districts = OrthogonalMooreGrid(
        (3, 3), cell_size=20.0, capacity=4, random=model.random
    )
    surveyors = [Surveyor(model) for _ in range(20)]

    cells = [*model.ground.all_cells, *model.districts.all_cells]

    def snapshot():
        locations = [
            (
                None if s.position is None else tuple(s.position),
                s.cell,
                s.district,
            )
            for s in surveyors
        ]
        # the order of occupants too, since seeded choices depend on it
        return locations, [list(cell.agents) for cell in cells]

    def check():
        for s in surveyors:
            if s.position is None:
                assert s.cell is None
                assert s.district is None
                continue
            assert s.cell is model.ground.find_nearest_cell(s.position)
            assert s.district is model.districts.find_nearest_cell(s.position)
            assert s in s.cell.agents
            assert s in s.district.agents
        occupants = [a for cell in model.ground.all_cells for a in cell.agents]
        assert len(occupants) == len(set(occupants))
        assert sorted(a.unique_id for a in occupants) == sorted(
            s.unique_id for s in surveyors if s.position is not None
        )

    for _ in range(2000):
        surveyor = rng.choice(surveyors)
        before = snapshot()
        action = rng.random()
        try:
            if action < 0.5:
                surveyor.position = [rng.uniform(0, 70), rng.uniform(0, 70)]
            elif action < 0.7 and surveyor.position is not None:
                surveyor.position += np.array([rng.uniform(-5, 5), rng.uniform(-5, 5)])
            elif action < 0.8:
                surveyor.cell = rng.choice(model.ground.all_cells.cells)
            elif action < 0.9:
                surveyor.district = rng.choice(model.districts.all_cells.cells)
            else:
                surveyor.position = None
        except (CellFullException, ValueError):
            assert snapshot() == before
        check()


def test_pickle_keeps_locations(model):
    """Locations survive pickling, and moves after unpickling stay consistent."""
    bird = Bird(model)
    bird.position = [11.0, 21.0]
    citizen = Citizen(model)
    citizen.view = [0.5, 0.5]

    restored = pickle.loads(pickle.dumps(model))  # noqa: S301
    bird, citizen = sorted(restored.agents, key=lambda a: a.unique_id)
    np.testing.assert_array_equal(bird.position, [11.0, 21.0])
    assert bird.cell is restored.ground[5, 10]

    bird.position = [31.0, 41.0]
    assert bird.cell is restored.ground[15, 20]
    assert restored.ground.find_nearest_cell(bird.position) is bird.cell
    np.testing.assert_array_equal(citizen.view, [0.5, 0.5])


def test_agents_with_named_locations_can_be_drawn(model):
    """Visualization finds an agent in a space whatever its location is called."""
    from mesa.visualization.backends import MatplotlibBackend  # noqa: PLC0415
    from mesa.visualization.components import AgentPortrayalStyle  # noqa: PLC0415
    from mesa.visualization.mpl_space_drawing import (  # noqa: PLC0415
        collect_agent_data,
    )

    citizen = Citizen(model)
    citizen.home = model.ground[3, 4]
    citizen.view = [0.25, -0.5]

    data = collect_agent_data(model.ground, lambda _: AgentPortrayalStyle())
    np.testing.assert_array_equal(data["loc"], [[3, 4]])
    data = collect_agent_data(model.opinion, lambda _: AgentPortrayalStyle())
    np.testing.assert_array_equal(data["loc"], [[0.25, -0.5]])

    backend = MatplotlibBackend(space_drawer=None)
    assert tuple(backend._get_agent_pos(citizen, model.ground)) == (3, 4)
    np.testing.assert_array_equal(
        backend._get_agent_pos(citizen, model.opinion), (0.25, -0.5)
    )
