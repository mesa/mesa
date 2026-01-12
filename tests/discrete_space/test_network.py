"""Tests for the Network class in discrete_space module."""

import random

import networkx as nx

from mesa.discrete_space.cell import Cell
from mesa.discrete_space.cell_agent import CellAgent
from mesa.discrete_space.network import Network
from mesa.model import Model


class TestNetwork:
    """Tests for the Network class."""

    def test_network_initialization(self):
        """Test Network initializes from a NetworkX graph."""
        graph = nx.complete_graph(5)
        rng = random.Random(42)
        network = Network(graph, random=rng)

        assert len(network._cells) == 5
        assert network.G is graph

    def test_network_cell_creation(self):
        """Test Network creates cells for each node."""
        graph = nx.path_graph(3)  # 0 -- 1 -- 2
        rng = random.Random(42)
        network = Network(graph, random=rng)

        for node_id in graph.nodes:
            assert node_id in network._cells
            assert isinstance(network._cells[node_id], Cell)

    def test_network_cell_connections(self):
        """Test Network cells are connected based on graph edges."""
        graph = nx.path_graph(3)  # 0 -- 1 -- 2
        rng = random.Random(42)
        network = Network(graph, random=rng)

        # Node 0 should be connected to node 1
        cell0 = network._cells[0]
        assert 1 in cell0.connections

        # Node 1 should be connected to both 0 and 2
        cell1 = network._cells[1]
        assert 0 in cell1.connections
        assert 2 in cell1.connections

        # Node 2 should be connected to node 1
        cell2 = network._cells[2]
        assert 1 in cell2.connections

    def test_network_with_capacity(self):
        """Test Network cells respect capacity."""
        graph = nx.complete_graph(3)
        rng = random.Random(42)
        network = Network(graph, capacity=2, random=rng)

        for cell in network.all_cells:
            assert cell.capacity == 2

    def test_network_add_cell(self):
        """Test adding a cell to the network."""
        graph = nx.Graph()
        graph.add_node(0)
        rng = random.Random(42)
        network = Network(graph, random=rng)

        new_cell = Cell(coordinate=1, random=rng)
        network.add_cell(new_cell)

        assert 1 in network._cells
        assert 1 in network.G.nodes

    def test_network_remove_cell(self):
        """Test removing a cell from the network."""
        graph = nx.complete_graph(3)
        rng = random.Random(42)
        network = Network(graph, random=rng)

        cell_to_remove = network._cells[0]
        network.remove_cell(cell_to_remove)

        assert 0 not in network._cells
        assert 0 not in network.G.nodes

    def test_network_add_connection(self):
        """Test adding a connection between cells."""
        graph = nx.Graph()
        graph.add_nodes_from([0, 1])
        rng = random.Random(42)
        network = Network(graph, random=rng)

        cell0 = network._cells[0]
        cell1 = network._cells[1]
        network.add_connection(cell0, cell1)

        assert 1 in cell0.connections
        assert 0 in cell1.connections
        assert graph.has_edge(0, 1)

    def test_network_remove_connection(self):
        """Test removing a connection between cells."""
        graph = nx.complete_graph(3)
        rng = random.Random(42)
        network = Network(graph, random=rng)

        cell0 = network._cells[0]
        cell1 = network._cells[1]
        network.remove_connection(cell0, cell1)

        assert 1 not in cell0.connections
        assert 0 not in cell1.connections
        assert not graph.has_edge(0, 1)

    def test_network_with_custom_cell_class(self):
        """Test Network with custom cell class."""

        class CustomCell(Cell):
            pass

        graph = nx.complete_graph(3)
        rng = random.Random(42)
        network = Network(graph, random=rng, cell_klass=CustomCell)

        for cell in network.all_cells:
            assert isinstance(cell, CustomCell)

    def test_network_with_agents(self):
        """Test placing agents in network cells."""
        graph = nx.complete_graph(3)
        rng = random.Random(42)
        network = Network(graph, random=rng)

        model = Model()
        agent = CellAgent(model)
        agent.cell = network._cells[0]

        assert agent in network._cells[0]._agents

    def test_network_all_cells(self):
        """Test all_cells property returns all cells."""
        graph = nx.complete_graph(5)
        rng = random.Random(42)
        network = Network(graph, random=rng)

        all_cells = list(network.all_cells)
        assert len(all_cells) == 5

    def test_network_complete_graph_connections(self):
        """Test complete graph has proper connections."""
        graph = nx.complete_graph(4)
        rng = random.Random(42)
        network = Network(graph, random=rng)

        # Each cell should be connected to all other cells
        for cell in network.all_cells:
            assert len(cell.connections) == 3  # n-1 connections

    def test_network_star_graph_connections(self):
        """Test star graph has proper connections."""
        graph = nx.star_graph(4)  # Center node 0, leaf nodes 1,2,3,4
        rng = random.Random(42)
        network = Network(graph, random=rng)

        # Center node should have 4 connections
        center_cell = network._cells[0]
        assert len(center_cell.connections) == 4

        # Leaf nodes should have 1 connection each
        for i in range(1, 5):
            leaf_cell = network._cells[i]
            assert len(leaf_cell.connections) == 1
