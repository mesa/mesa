"""Tests for locations declared on agents."""

import pickle
import random

import networkx as nx
import numpy as np
import pytest

from mesa import Agent, Model
from mesa.discrete_space import Network, OrthogonalMooreGrid
from mesa.exceptions import CellFullException
from mesa.experimental.continuous_space import ContinuousSpace, ContinuousSpaceAgent
from mesa.experimental.locations import CellLocation, ContinuousLocation


class Citizen(Agent):
    """An agent with independent locations in several spaces."""

    home = CellLocation()
    work = CellLocation()
    view = ContinuousLocation("opinion")
    lifestyle = ContinuousLocation("traits")


class Bird(Agent):
    """An agent whose ground cell follows its position in the air."""

    position = ContinuousLocation("air")
    cell = CellLocation("ground", follows="position")


class Surveyor(Agent):
    """An agent whose cells in two grids follow its position."""

    position = ContinuousLocation("air")
    cell = CellLocation("ground", follows="position")
    district = CellLocation("districts", follows="position")


class Walker(Agent):
    """An agent on a grid whose district follows its cell."""

    cell = CellLocation("ground")
    district = CellLocation("districts", follows="cell")


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
    grid = OrthogonalMooreGrid((3, 3), capacity=1, random=model.random)
    blocker, citizen = Citizen(model), Citizen(model)
    blocker.home = grid[1, 1]
    citizen.home = grid[0, 0]

    with pytest.raises(CellFullException):
        citizen.home = grid[1, 1]
    assert citizen.home is grid[0, 0]
    assert citizen in grid[0, 0].agents


def test_several_cells_per_agent(model):
    """One agent can hold several cells, in the same space and in others."""
    citizen = Citizen(model)
    citizen.home = model.ground[1, 1]
    citizen.work = model.ground[9, 9]

    assert citizen in model.ground[1, 1].agents
    assert citizen in model.ground[9, 9].agents

    citizen.work = model.districts[0, 0]
    assert citizen not in model.ground[9, 9].agents
    assert citizen in model.districts[0, 0].agents
    assert citizen.home is model.ground[1, 1]


# ----------------------------------------------------------------------
# ContinuousLocation
# ----------------------------------------------------------------------
def test_continuous_location_joins_on_first_write(model):
    """The first point joins the named space; None leaves it."""
    citizen = Citizen(model)
    assert citizen.view is None

    citizen.view = [0.2, -0.4]
    assert citizen in model.opinion.active_agents
    np.testing.assert_array_equal(citizen.view, [0.2, -0.4])

    citizen.view = None
    assert citizen.view is None
    assert citizen not in model.opinion.active_agents


def test_continuous_location_reads_are_values(model):
    """A saved position keeps its value, and += moves through the setter."""
    citizen = Citizen(model)
    citizen.view = [0.0, 0.0]

    saved = citizen.view
    citizen.view += np.array([0.5, 0.25])
    np.testing.assert_array_equal(saved, [0.0, 0.0])
    np.testing.assert_array_equal(citizen.view, [0.5, 0.25])

    # changing a read value does not move the agent
    citizen.view[0] = -0.9
    np.testing.assert_array_equal(citizen.view, [0.5, 0.25])


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

    distances, _ = model.opinion.calculate_distances(
        np.array([0.0, 0.0]), agents=[citizens[3]]
    )
    np.testing.assert_allclose(distances, [0.3])


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


def test_missing_space_attribute_names_it(model):
    """A location naming a space the model lacks says which one."""

    class Lost(Agent):
        position = ContinuousLocation("nowhere")

    with pytest.raises(AttributeError, match="nowhere"):
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
    citizen.work = model.districts[0, 0]
    citizen.view = [0.0, 0.0]

    citizen.remove()
    assert citizen not in model.ground[1, 1].agents
    assert citizen not in model.districts[0, 0].agents
    assert citizen not in model.opinion.active_agents


def test_subclasses_inherit_and_override_locations(model):
    """A subclass keeps its parent's locations and can redefine one."""

    class Commuter(Citizen):
        work = CellLocation()
        car = CellLocation()

    names = [location.name for location in Commuter._mesa_locations]
    assert sorted(names) == ["car", "home", "lifestyle", "view", "work"]
    assert Commuter.work is not Citizen.work


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


def test_follower_written_first_places_the_agent(model):
    """Assigning a follower before the primary places the agent at the cell center."""
    bird = Bird(model)
    bird.cell = model.ground[2, 3]
    np.testing.assert_array_equal(bird.position, [5.0, 7.0])
    assert bird in model.air.active_agents


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
    with pytest.raises(ValueError):
        surveyor.position = [90.0, 90.0]
    np.testing.assert_array_equal(surveyor.position, [10.0, 10.0])
    assert surveyor.cell is model.ground[5, 5]


def test_several_followers(model):
    """All followers move with the primary, and a cell write moves them all."""
    surveyor = Surveyor(model)
    surveyor.position = [45.0, 85.0]
    assert surveyor.cell is model.ground[22, 42]
    assert surveyor.district is model.districts[2, 4]

    surveyor.district = model.districts[0, 0]
    np.testing.assert_array_equal(surveyor.position, [10.0, 10.0])
    assert surveyor.cell is model.ground[5, 5]


def test_follower_of_a_cell(model):
    """A district can follow a grid cell."""
    walker = Walker(model)
    walker.cell = model.ground[12, 3]
    assert walker.district is model.districts[1, 0]

    walker.district = model.districts[4, 4]
    assert walker.cell is model.ground[45, 45]


def test_follower_rules(model):
    """Followers need a space, a real primary, and cannot be cleared on their own."""
    with pytest.raises(TypeError):
        CellLocation(follows="position")

    class Orphan(Agent):
        cell = CellLocation("ground", follows="nothing")

    with pytest.raises(TypeError, match="nothing"):
        Orphan(model).cell = model.ground[0, 0]

    class Chain(Agent):
        position = ContinuousLocation("air")
        cell = CellLocation("ground", follows="position")
        district = CellLocation("districts", follows="cell")

    with pytest.raises(TypeError, match="follow position directly"):
        Chain(model).position = [1.0, 1.0]

    bird = Bird(model)
    bird.position = [1.0, 1.0]
    with pytest.raises(ValueError, match="set position to None"):
        bird.cell = None
    with pytest.raises(ValueError, match="another space"):
        bird.cell = model.districts[0, 0]

    bird.position = None
    assert bird.cell is None
    assert bird not in model.ground[0, 0].agents


def test_follower_in_network_raises(model):
    """A network does not divide space into cells, so it cannot follow a point."""
    model.roads = Network(nx.path_graph(3), random=model.random)

    class Driver(Agent):
        position = ContinuousLocation("air")
        node = CellLocation("roads", follows="position")

    with pytest.raises(TypeError, match="Network"):
        Driver(model).position = [1.0, 1.0]


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

    def snapshot():
        return [
            (
                None if s.position is None else tuple(s.position),
                s.cell,
                s.district,
            )
            for s in surveyors
        ]

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
