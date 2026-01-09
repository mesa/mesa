"""Tests for the space_drawers module in visualization."""

import random
import unittest.mock as mock

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest

from mesa.discrete_space.cell_agent import CellAgent
from mesa.discrete_space.grid import OrthogonalMooreGrid, OrthogonalVonNeumannGrid
from mesa.discrete_space.network import Network
from mesa.discrete_space.voronoi import VoronoiGrid
from mesa.model import Model
from mesa.space import ContinuousSpace, MultiGrid, NetworkGrid, SingleGrid
from mesa.visualization.space_drawers import (
    BaseSpaceDrawer,
    HexSpaceDrawer,
    OrthogonalSpaceDrawer,
)


class TestBaseSpaceDrawer:
    """Tests for the BaseSpaceDrawer class."""

    def test_base_space_drawer_initialization(self):
        """Test BaseSpaceDrawer initializes with space."""
        rng = random.Random(42)
        space = OrthogonalMooreGrid((5, 5), random=rng)
        drawer = BaseSpaceDrawer(space)
        
        assert drawer.space is space
        assert drawer.viz_xmin is None
        assert drawer.viz_xmax is None
        assert drawer.viz_ymin is None
        assert drawer.viz_ymax is None

    def test_base_space_drawer_get_viz_limits(self):
        """Test get_viz_limits returns tuple of limits."""
        rng = random.Random(42)
        space = OrthogonalMooreGrid((5, 5), random=rng)
        drawer = BaseSpaceDrawer(space)
        
        limits = drawer.get_viz_limits()
        
        assert isinstance(limits, tuple)
        assert len(limits) == 4


class TestOrthogonalSpaceDrawer:
    """Tests for the OrthogonalSpaceDrawer class."""

    def test_orthogonal_drawer_initialization_moore(self):
        """Test OrthogonalSpaceDrawer with Moore grid."""
        rng = random.Random(42)
        space = OrthogonalMooreGrid((10, 10), random=rng)
        drawer = OrthogonalSpaceDrawer(space)
        
        assert drawer.space is space
        assert drawer.viz_xmin == -0.5
        assert drawer.viz_xmax == 9.5
        assert drawer.viz_ymin == -0.5
        assert drawer.viz_ymax == 9.5

    def test_orthogonal_drawer_initialization_von_neumann(self):
        """Test OrthogonalSpaceDrawer with VonNeumann grid."""
        rng = random.Random(42)
        space = OrthogonalVonNeumannGrid((5, 8), random=rng)
        drawer = OrthogonalSpaceDrawer(space)
        
        assert drawer.viz_xmax == 4.5
        assert drawer.viz_ymax == 7.5

    def test_orthogonal_drawer_default_size(self):
        """Test OrthogonalSpaceDrawer calculates default marker size."""
        rng = random.Random(42)
        space = OrthogonalMooreGrid((10, 10), random=rng)
        drawer = OrthogonalSpaceDrawer(space)
        
        expected_size = (180 / 10) ** 2
        assert drawer.s_default == expected_size

    def test_orthogonal_drawer_get_viz_limits(self):
        """Test OrthogonalSpaceDrawer visualization limits."""
        rng = random.Random(42)
        space = OrthogonalMooreGrid((5, 5), random=rng)
        drawer = OrthogonalSpaceDrawer(space)
        
        limits = drawer.get_viz_limits()
        
        assert limits == (-0.5, 4.5, -0.5, 4.5)


class TestHexSpaceDrawer:
    """Tests for the HexSpaceDrawer class."""

    def test_hex_drawer_initialization(self):
        """Test HexSpaceDrawer initializes properly."""
        from mesa.discrete_space import HexGrid
        
        rng = random.Random(42)
        space = HexGrid((5, 5), random=rng)
        drawer = HexSpaceDrawer(space)
        
        assert drawer.space is space
        # Check that hex-specific visualization limits are set
        assert drawer.viz_xmin is not None
        assert drawer.viz_xmax is not None

    def test_hex_drawer_get_viz_limits(self):
        """Test HexSpaceDrawer visualization limits."""
        from mesa.discrete_space import HexGrid
        
        rng = random.Random(42)
        space = HexGrid((5, 5), random=rng)
        drawer = HexSpaceDrawer(space)
        
        limits = drawer.get_viz_limits()
        
        assert limits is not None
        assert len(limits) == 4


class TestSpaceDrawerMatplotlib:
    """Tests for matplotlib drawing functionality."""

    @pytest.fixture
    def orthogonal_grid(self):
        """Create a simple orthogonal grid for testing."""
        rng = random.Random(42)
        return OrthogonalMooreGrid((5, 5), random=rng)

    def test_orthogonal_drawer_draw_matplotlib_creates_figure(self, orthogonal_grid):
        """Test draw_matplotlib creates a figure."""
        drawer = OrthogonalSpaceDrawer(orthogonal_grid)
        
        # Mock to avoid actual drawing
        with mock.patch.object(plt, 'subplots') as mock_subplots:
            fig = mock.MagicMock()
            ax = mock.MagicMock()
            mock_subplots.return_value = (fig, ax)
            
            result = drawer.draw_matplotlib()
            
            # Should have called subplots
            mock_subplots.assert_called_once()
        
        plt.close('all')

    def test_orthogonal_drawer_draw_matplotlib_with_existing_ax(self, orthogonal_grid):
        """Test draw_matplotlib works with existing axes."""
        drawer = OrthogonalSpaceDrawer(orthogonal_grid)
        
        fig, ax = plt.subplots()
        result = drawer.draw_matplotlib(ax=ax)
        
        assert result is ax
        plt.close('all')

    def test_orthogonal_drawer_draw_matplotlib_with_kwargs(self, orthogonal_grid):
        """Test draw_matplotlib accepts additional kwargs."""
        drawer = OrthogonalSpaceDrawer(orthogonal_grid)
        
        # Should not raise
        fig, ax = plt.subplots()
        drawer.draw_matplotlib(ax=ax, linewidth=2, color="blue")
        
        plt.close('all')


class TestSpaceDrawerAltair:
    """Tests for Altair drawing functionality."""

    @pytest.fixture
    def orthogonal_grid(self):
        """Create a simple orthogonal grid for testing."""
        rng = random.Random(42)
        return OrthogonalMooreGrid((5, 5), random=rng)

    def test_orthogonal_drawer_draw_altair(self, orthogonal_grid):
        """Test draw_altair method."""
        drawer = OrthogonalSpaceDrawer(orthogonal_grid)
        
        result = drawer.draw_altair()
        
        # Should return an Altair chart or LayeredChart
        import altair as alt
        assert isinstance(result, alt.LayerChart | alt.Chart)


class TestSpaceDrawerIntegration:
    """Integration tests for space drawers."""

    def test_different_grid_sizes(self):
        """Test drawers work with different grid sizes."""
        rng = random.Random(42)
        
        sizes = [(3, 3), (10, 10), (5, 15), (20, 5)]
        for width, height in sizes:
            space = OrthogonalMooreGrid((width, height), random=rng)
            drawer = OrthogonalSpaceDrawer(space)
            
            assert drawer.viz_xmax == width - 0.5
            assert drawer.viz_ymax == height - 0.5

    def test_drawer_with_agents(self):
        """Test drawer works with agents in the space."""
        rng = random.Random(42)
        space = OrthogonalMooreGrid((5, 5), random=rng)
        model = Model()
        
        # Add some agents
        for _ in range(3):
            agent = CellAgent(model)
            cell = space.select_random_empty_cell()
            agent.cell = cell
        
        drawer = OrthogonalSpaceDrawer(space)
        
        # Drawing should still work
        fig, ax = plt.subplots()
        result = drawer.draw_matplotlib(ax=ax)
        
        assert result is not None
        plt.close('all')
