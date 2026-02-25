"""tests for matplotlib components."""

import networkx as nx
from matplotlib.figure import Figure

from mesa import Model
from mesa.discrete_space import (
    CellAgent,
    HexGrid,
    Network,
    OrthogonalMooreGrid,
    VoronoiGrid,
)
from mesa.visualization.components import AgentPortrayalStyle, PropertyLayerStyle
from mesa.visualization.mpl_space_drawing import (
    draw_hex_grid,
    draw_network,
    draw_orthogonal_grid,
    draw_property_layers,
    draw_space,
    draw_voronoi_grid,
)


def agent_portrayal(agent):
    """Simple portrayal of an agent.

    Args:
        agent (Agent): The agent to portray

    """
    return AgentPortrayalStyle(
        size=10,
        color="tab:blue",
        marker="s" if (agent.unique_id % 2) == 0 else "o",
    )


def test_draw_space():
    """Test draw_space helper method."""

    def my_portrayal(agent):
        """Simple portrayal of an agent.

        Args:
            agent (Agent): The agent to portray

        """
        return AgentPortrayalStyle(
            size=10,
            color="tab:blue",
            marker="s" if (agent.unique_id % 2) == 0 else "o",
            alpha=0.5,
            linewidths=1,
            edgecolors="tab:orange",
        )

    # draw space for voroinoi
    model = Model(rng=42)
    coordinates = model.rng.random((100, 2)) * 10
    grid = VoronoiGrid(coordinates.tolist(), random=model.random, capacity=1)
    for _ in range(10):
        agent = CellAgent(model)
        agent.cell = grid.select_random_empty_cell()

    fig = Figure()
    ax = fig.add_subplot()
    draw_space(grid, my_portrayal, ax=ax)

    # draw orthogonal grid
    model = Model(rng=42)
    grid = OrthogonalMooreGrid((10, 10), torus=True, random=model.random, capacity=1)
    for _ in range(10):
        agent = CellAgent(model)
        agent.cell = grid.select_random_empty_cell()
    fig = Figure()
    ax = fig.add_subplot()
    draw_space(grid, my_portrayal, ax=ax)


def test_draw_hex_grid():
    """Test drawing hexgrids."""
    model = Model(rng=42)
    grid = HexGrid((10, 10), torus=True, random=model.random, capacity=1)
    for _ in range(10):
        agent = CellAgent(model)
        agent.cell = grid.select_random_empty_cell()

    fig = Figure()
    ax = fig.add_subplot()
    draw_hex_grid(grid, agent_portrayal, ax)


def test_draw_voronoi_grid():
    """Test drawing voronoi grids."""
    model = Model(rng=42)

    coordinates = model.rng.random((100, 2)) * 10

    grid = VoronoiGrid(coordinates.tolist(), random=model.random, capacity=1)
    for _ in range(10):
        agent = CellAgent(model)
        agent.cell = grid.select_random_empty_cell()

    fig = Figure()
    ax = fig.add_subplot()
    draw_voronoi_grid(grid, agent_portrayal, ax)


def test_draw_orthogonal_grid():
    """Test drawing orthogonal grids."""
    model = Model(rng=42)
    grid = OrthogonalMooreGrid((10, 10), torus=True, random=model.random, capacity=1)
    for _ in range(10):
        agent = CellAgent(model)
        agent.cell = grid.select_random_empty_cell()

    fig = Figure()
    ax = fig.add_subplot()
    draw_orthogonal_grid(grid, agent_portrayal, ax)


def test_draw_network():
    """Test drawing network."""
    n = 10
    m = 20
    rng = 42
    graph = nx.gnm_random_graph(n, m, seed=rng)

    model = Model(rng=42)
    grid = Network(graph, random=model.random, capacity=1, layout=nx.spring_layout)
    for _ in range(10):
        agent = CellAgent(model)
        agent.cell = grid.select_random_empty_cell()

    fig = Figure()
    ax = fig.add_subplot()
    draw_network(grid, agent_portrayal, ax)


def test_draw_property_layers():
    """Test drawing property layers."""

    def propertylayer_portrayal(_):
        return PropertyLayerStyle(colormap="viridis", colorbar=True)

    model = Model(rng=42)
    grid = OrthogonalMooreGrid((10, 10), torus=True, random=model.random, capacity=1)
    grid.create_property_layer("test", 0.0)

    fig = Figure()
    ax = fig.add_subplot()
    draw_property_layers(grid, propertylayer_portrayal, ax)
