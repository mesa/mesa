"""Test the Grid objects."""

import random
import unittest
from unittest.mock import Mock, patch

from mesa.space import HexSingleGrid, MultiGrid, SingleGrid

# Initial agent positions for testing
#
# --- visual aid ----
#   0 0 0
#   1 1 0
#   0 1 0
#   1 0 1
#   0 0 1
# -------------------
TEST_GRID = [[0, 1, 0, 1, 0, 0], [0, 0, 1, 1, 0, 1], [1, 1, 0, 0, 0, 1]]


class MockAgent:
    """Minimalistic agent for testing purposes."""

    def __init__(self, unique_id):  # noqa: D107
        self.random = random.Random(0)
        self.unique_id = unique_id
        self.pos = None


class TestSingleGrid(unittest.TestCase):
    """Testing a non-toroidal singlegrid."""

    torus = False

    def setUp(self):
        """Create a test non-toroidal grid and populate it with Mock Agents."""
        # The height needs to be even to test the edge case described in PR #1517
        height = 6  # height of grid
        width = 3  # width of grid
        self.grid = SingleGrid(width, height, self.torus)
        self.agents = []
        counter = 0
        for x in range(width):
            for y in range(height):
                if TEST_GRID[x][y] == 0:
                    continue
                counter += 1
                # Create and place the mock agent
                a = MockAgent(counter)
                self.agents.append(a)
                self.grid.place_agent(a, (x, y))

    def test_agent_positions(self):
        """Ensure that the agents are all placed properly."""
        for agent in self.agents:
            x, y = agent.pos
            assert self.grid[x][y] == agent

    def test_cell_agent_reporting(self):
        """Ensure that if an agent is in a cell, get_cell_list_contents accurately reports that fact."""
        for agent in self.agents:
            x, y = agent.pos
            assert agent in self.grid.get_cell_list_contents([(x, y)])

    def test_listfree_cell_agent_reporting(self):
        """Test if agent is correctly tracked in cell.

        Ensure that if an agent is in a cell, get_cell_list_contents accurately
        reports that fact, even when single position is not wrapped in a list.
        """
        for agent in self.agents:
            x, y = agent.pos
            assert agent in self.grid.get_cell_list_contents((x, y))

    def test_iter_cell_agent_reporting(self):
        """Ensure that if an agent is in a cell, iter_cell_list_contents accurately reports that fact."""
        for agent in self.agents:
            x, y = agent.pos
            assert agent in self.grid.iter_cell_list_contents([(x, y)])

    def test_listfree_iter_cell_agent_reporting(self):
        """Test if agent is correctly tracked in cell in iterator.

        Ensure that if an agent is in a cell, iter_cell_list_contents
        accurately reports that fact, even when single position is not
        wrapped in a list.
        """
        for agent in self.agents:
            x, y = agent.pos
            assert agent in self.grid.iter_cell_list_contents((x, y))

    def test_neighbors(self):
        """Test the base neighborhood methods on the non-toroid."""
        neighborhood = self.grid.get_neighborhood((1, 1), moore=True)
        assert len(neighborhood) == 8

        neighborhood = self.grid.get_neighborhood((1, 4), moore=False)
        assert len(neighborhood) == 4

        neighborhood = self.grid.get_neighborhood((1, 4), moore=True)
        assert len(neighborhood) == 8

        neighborhood = self.grid.get_neighborhood((0, 0), moore=False)
        assert len(neighborhood) == 2

        with self.assertRaises(Exception):
            neighbors = self.grid.get_neighbors((4, 1), moore=False)

        neighbors = self.grid.get_neighbors((1, 1), moore=False, include_center=True)
        assert len(neighbors) == 3

        neighbors = self.grid.get_neighbors((1, 3), moore=False, radius=2)
        assert len(neighbors) == 3

    def test_coord_iter(self):  # noqa: D102
        ci = self.grid.coord_iter()

        # no agent in first space
        first = next(ci)
        assert first[0] is None
        assert first[1] == (0, 0)

        # first agent in the second space
        second = next(ci)
        assert second[0].unique_id == 1
        assert second[0].pos == (0, 1)
        assert second[1] == (0, 1)

    def test_agent_move(self):  # noqa: D102
        # get the agent at [0, 1]
        agent = self.agents[0]
        self.grid.move_agent(agent, (1, 0))
        assert agent.pos == (1, 0)
        # move it off the torus and check for the exception
        if not self.grid.torus:
            with self.assertRaises(Exception):
                self.grid.move_agent(agent, [-1, 1])
            with self.assertRaises(Exception):
                self.grid.move_agent(agent, [1, self.grid.height + 1])
        else:
            self.grid.move_agent(agent, [0, -1])
            assert agent.pos == (0, self.grid.height - 1)
            self.grid.move_agent(agent, [1, self.grid.height])
            assert agent.pos == (1, 0)

    def test_agent_remove(self):  # noqa: D102
        agent = self.agents[0]
        x, y = agent.pos
        self.grid.remove_agent(agent)
        assert agent.pos is None
        assert self.grid[x][y] is None

    def test_swap_pos(self):  # noqa: D102
        # Swap agents positions
        agent_a, agent_b = list(filter(None, self.grid))[:2]
        pos_a = agent_a.pos
        pos_b = agent_b.pos

        self.grid.swap_pos(agent_a, agent_b)

        assert agent_a.pos == pos_b
        assert agent_b.pos == pos_a
        assert self.grid[pos_a] == agent_b
        assert self.grid[pos_b] == agent_a

        # Swap the same agents
        self.grid.swap_pos(agent_a, agent_a)

        assert agent_a.pos == pos_b
        assert self.grid[pos_b] == agent_a

        # Raise for agents not on the grid
        self.grid.remove_agent(agent_a)
        self.grid.remove_agent(agent_b)

        id_a = agent_a.unique_id
        id_b = agent_b.unique_id
        e_message = f"<Agent id: {id_a}>, <Agent id: {id_b}> - not on the grid"
        with self.assertRaisesRegex(Exception, e_message):
            self.grid.swap_pos(agent_a, agent_b)


class TestSingleGridTorus(TestSingleGrid):
    """Testing the toroidal singlegrid."""

    torus = True

    def test_neighbors(self):
        """Test the toroidal neighborhood methods."""
        neighborhood = self.grid.get_neighborhood((1, 1), moore=True)
        assert len(neighborhood) == 8

        neighborhood = self.grid.get_neighborhood((1, 4), moore=True)
        assert len(neighborhood) == 8

        neighborhood = self.grid.get_neighborhood((0, 0), moore=False)
        assert len(neighborhood) == 4

        # here we test the edge case described in PR #1517 using a radius
        # measuring half of the grid height
        neighborhood = self.grid.get_neighborhood((0, 0), moore=True, radius=3)
        assert len(neighborhood) == 17

        neighborhood = self.grid.get_neighborhood((1, 1), moore=False, radius=3)
        assert len(neighborhood) == 15

        neighbors = self.grid.get_neighbors((1, 4), moore=False)
        assert len(neighbors) == 2

        neighbors = self.grid.get_neighbors((1, 4), moore=True)
        assert len(neighbors) == 4

        neighbors = self.grid.get_neighbors((1, 1), moore=False, include_center=True)
        assert len(neighbors) == 3

        neighbors = self.grid.get_neighbors((1, 3), moore=False, radius=2)
        assert len(neighbors) == 3


class TestSingleGridEnforcement(unittest.TestCase):
    """Test the enforcement in SingleGrid."""

    def setUp(self):
        """Create a test non-toroidal grid and populate it with Mock Agents."""
        width = 3
        height = 5
        self.grid = SingleGrid(width, height, True)
        self.agents = []
        counter = 0
        for x in range(width):
            for y in range(height):
                if TEST_GRID[x][y] == 0:
                    continue
                counter += 1
                # Create and place the mock agent
                a = MockAgent(counter)
                self.agents.append(a)
                self.grid.place_agent(a, (x, y))
        self.num_agents = len(self.agents)

    @patch.object(MockAgent, "model", create=True)
    def test_enforcement(self, mock_model):
        """Test the SingleGrid empty count and enforcement."""
        assert len(self.grid.empties) == 9
        a = MockAgent(100)
        with self.assertRaises(Exception):
            self.grid.place_agent(a, (0, 1))

        # Place the agent in an empty cell
        mock_model.schedule.get_agent_count = Mock(side_effect=lambda: len(self.agents))
        self.grid.move_to_empty(a)
        self.num_agents += 1
        # Test whether after placing, the empty cells are reduced by 1
        assert a.pos not in self.grid.empties
        assert len(self.grid.empties) == 8
        for _i in range(10):
            self.grid.move_to_empty(a)
        assert len(self.grid.empties) == 8

        # Place agents until the grid is full
        empty_cells = len(self.grid.empties)
        for i in range(empty_cells):
            a = MockAgent(101 + i)
            self.grid.move_to_empty(a)
            self.num_agents += 1
        assert len(self.grid.empties) == 0

        a = MockAgent(110)
        with self.assertRaises(Exception):
            self.grid.move_to_empty(a)
        with self.assertRaises(Exception):
            self.move_to_empty(self.agents[0])


# Number of agents at each position for testing
# Initial agent positions for testing
#
# --- visual aid ----
#   0 0 0
#   2 0 3
#   0 5 0
#   1 1 0
#   0 0 0
# -------------------
TEST_MULTIGRID = [[0, 1, 0, 2, 0], [0, 1, 5, 0, 0], [0, 0, 0, 3, 0]]


class TestMultiGrid(unittest.TestCase):
    """Testing a toroidal MultiGrid."""

    torus = True

    def setUp(self):
        """Create a test non-toroidal grid and populate it with Mock Agents."""
        width = 3
        height = 5
        self.grid = MultiGrid(width, height, self.torus)
        self.agents = []
        counter = 0
        for x in range(width):
            for y in range(height):
                for _i in range(TEST_MULTIGRID[x][y]):
                    counter += 1
                    # Create and place the mock agent
                    a = MockAgent(counter)
                    self.agents.append(a)
                    self.grid.place_agent(a, (x, y))

    def test_agent_positions(self):
        """Ensure that the agents are all placed properly on the MultiGrid."""
        for agent in self.agents:
            x, y = agent.pos
            assert agent in self.grid[x][y]

    def test_neighbors(self):
        """Test the toroidal MultiGrid neighborhood methods."""
        neighborhood = self.grid.get_neighborhood((1, 1), moore=True)
        assert len(neighborhood) == 8

        neighborhood = self.grid.get_neighborhood((1, 4), moore=True)
        assert len(neighborhood) == 8

        neighborhood = self.grid.get_neighborhood((0, 0), moore=False)
        assert len(neighborhood) == 4

        neighbors = self.grid.get_neighbors((1, 4), moore=False)
        assert len(neighbors) == 0

        neighbors = self.grid.get_neighbors((1, 4), moore=True)
        assert len(neighbors) == 5

        neighbors = self.grid.get_neighbors((1, 1), moore=False, include_center=True)
        assert len(neighbors) == 7

        neighbors = self.grid.get_neighbors((1, 3), moore=False, radius=2)
        assert len(neighbors) == 11


class TestHexSingleGrid(unittest.TestCase):
    """Testing a hexagonal singlegrid."""

    def setUp(self):
        """Create a test non-toroidal grid and populate it with Mock Agents."""
        width = 3
        height = 5
        self.grid = HexSingleGrid(width, height, torus=False)
        self.agents = []
        counter = 0
        for x in range(width):
            for y in range(height):
                if TEST_GRID[x][y] == 0:
                    continue
                counter += 1
                # Create and place the mock agent
                a = MockAgent(counter)
                self.agents.append(a)
                self.grid.place_agent(a, (x, y))

    def test_neighbors(self):
        """Test the hexagonal neighborhood methods on the non-toroid."""
        neighborhood = self.grid.get_neighborhood((1, 1))
        assert len(neighborhood) == 6

        neighborhood = self.grid.get_neighborhood((0, 2))
        assert len(neighborhood) == 4

        neighborhood = self.grid.get_neighborhood((1, 0))
        assert len(neighborhood) == 3

        neighborhood = self.grid.get_neighborhood((1, 4))
        assert len(neighborhood) == 5

        neighborhood = self.grid.get_neighborhood((0, 4))
        assert len(neighborhood) == 2

        neighborhood = self.grid.get_neighborhood((0, 0))
        assert len(neighborhood) == 3

        neighborhood = self.grid.get_neighborhood((1, 1), include_center=True)
        assert len(neighborhood) == 7

        neighborhood = self.grid.get_neighborhood((0, 0), radius=4)
        assert len(neighborhood) == 13
        assert sum(x + y for x, y in neighborhood) == 39


class TestHexSingleGridTorus(TestSingleGrid):
    """Testing a hexagonal toroidal singlegrid."""

    def setUp(self):
        """Create a test non-toroidal grid and populate it with Mock Agents."""
        width = 3
        height = 5
        self.grid = HexSingleGrid(width, height, torus=True)
        self.agents = []
        counter = 0
        for x in range(width):
            for y in range(height):
                if TEST_GRID[x][y] == 0:
                    continue
                counter += 1
                # Create and place the mock agent
                a = MockAgent(counter)
                self.agents.append(a)
                self.grid.place_agent(a, (x, y))

    def test_neighbors(self):
        """Test the hexagonal neighborhood methods on the toroid."""
        neighborhood = self.grid.get_neighborhood((1, 1))
        assert len(neighborhood) == 6

        neighborhood = self.grid.get_neighborhood((1, 1), include_center=True)
        assert len(neighborhood) == 7

        neighborhood = self.grid.get_neighborhood((0, 0))
        assert len(neighborhood) == 6

        neighborhood = self.grid.get_neighborhood((2, 4))
        assert len(neighborhood) == 6

        neighborhood = self.grid.get_neighborhood((1, 1), include_center=True, radius=2)
        assert len(neighborhood) == 13

        neighborhood = self.grid.get_neighborhood((0, 0), radius=4)
        assert len(neighborhood) == 14
        assert sum(x + y for x, y in neighborhood) == 45


class TestIndexing:  # noqa: D101
    # Create a grid where the content of each coordinate is a tuple of its coordinates
    grid = SingleGrid(3, 5, True)
    for _, pos in grid.coord_iter():
        x, y = pos
        grid._grid[x][y] = pos

    def test_int(self):  # noqa: D102
        assert self.grid[0][0] == (0, 0)

    def test_tuple(self):  # noqa: D102
        assert self.grid[1, 1] == (1, 1)

    def test_list(self):  # noqa: D102
        assert self.grid[(0, 0), (1, 1)] == [(0, 0), (1, 1)]
        assert self.grid[(0, 0), (5, 3)] == [(0, 0), (2, 3)]

    def test_torus(self):  # noqa: D102
        assert self.grid[3, 5] == (0, 0)

    def test_slice(self):  # noqa: D102
        assert self.grid[:, 0] == [(0, 0), (1, 0), (2, 0)]
        assert self.grid[::-1, 0] == [(2, 0), (1, 0), (0, 0)]
        assert self.grid[1, :] == [(1, 0), (1, 1), (1, 2), (1, 3), (1, 4)]
        assert self.grid[:, :] == [(x, y) for x in range(3) for y in range(5)]


if __name__ == "__main__":
    unittest.main()
