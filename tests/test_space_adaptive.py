"""Tests for the AdaptiveVoronoiSpace class."""

import numpy as np
import pytest

from mesa.model import Model
from mesa.space_adaptive import AdaptiveVoronoiSpace
from mesa.visualization.space_drawers import ContinuousSpaceDrawer
from mesa.visualization.space_renderer import SpaceRenderer
from tests.test_grid import MockAgent


@pytest.fixture
def adaptive_space():
    """Fixture for a standard non-toroidal adaptive space."""
    return AdaptiveVoronoiSpace(x_max=100, y_max=100, torus=False, n_clusters=2)


@pytest.fixture
def agents_in_groups(adaptive_space):
    """Fixture to place agents in two distinct geographical clusters."""
    agents = []

    # Group 0: Bottom-Left
    for i in range(5):
        pos = (1 + np.random.random(), 1 + np.random.random())
        a = MockAgent(i)
        adaptive_space.place_agent(a, pos)
        agents.append(a)

    # Group 1: Top-Right
    for i in range(5, 10):
        pos = (98 + np.random.random(), 98 + np.random.random())
        a = MockAgent(i)
        adaptive_space.place_agent(a, pos)
        agents.append(a)

    return agents


class TestAdaptiveVoronoiSpace:
    """Test cases for AdaptiveVoronoiSpace following Mesa conventions."""

    def test_initialization(self, adaptive_space):
        """Ensure parameters are set correctly upon init."""
        assert adaptive_space.x_max == 100
        assert adaptive_space.n_clusters == 2
        assert adaptive_space.centroids is None

    def test_rebuild_logic(self, adaptive_space, agents_in_groups):
        """Test the clustering execution and centroid generation."""
        adaptive_space.rebuild_voronoi()
        assert len(adaptive_space.centroids) == 2

        c1, c2 = adaptive_space.centroids
        assert any(np.allclose(c, [1.5, 1.5], atol=2.0) for c in [c1, c2])
        assert any(np.allclose(c, [98.5, 98.5], atol=2.0) for c in [c1, c2])

    def test_get_voronoi_neighbors(self, adaptive_space, agents_in_groups):
        """Test that neighbors are restricted to the same Voronoi cell."""
        adaptive_space.rebuild_voronoi()
        agent_bl = agents_in_groups[0]
        neighbors = adaptive_space.get_voronoi_neighbors(agent_bl, include_center=False)

        neighbor_ids = [a.unique_id for a in neighbors]
        for i in range(1, 5):
            assert i in neighbor_ids
        for i in range(5, 10):
            assert i not in neighbor_ids

    def test_get_region_id_mapping(self, adaptive_space, agents_in_groups):
        """Test that arbitrary coordinates map to the correct dynamic region."""
        adaptive_space.rebuild_voronoi()
        id_bl = adaptive_space.get_region_id((10, 10))
        id_tr = adaptive_space.get_region_id((90, 90))

        assert id_bl != id_tr
        assert adaptive_space.get_region_id((11, 11)) == id_bl
        assert adaptive_space.get_region_id((89, 89)) == id_tr

    def test_low_agent_warning(self, adaptive_space):
        """Ensure it warns rather than crashes when agents < n_clusters."""
        a = MockAgent(1)
        adaptive_space.place_agent(a, (10, 10))
        with pytest.warns(RuntimeWarning, match="Not enough agents"):
            adaptive_space.rebuild_voronoi()


class TestAdaptiveSpaceRendererIntegration:
    """Tests ensuring integration with the visualization system."""

    def test_renderer_selection(self):
        """Verify SpaceRenderer identifies it as a ContinuousSpace descendant."""

        class AdaptiveModel(Model):
            def __init__(self):
                super().__init__()
                self.grid = AdaptiveVoronoiSpace(10, 10, False)

        model = AdaptiveModel()
        sr = SpaceRenderer(model)
        assert isinstance(sr.space_drawer, ContinuousSpaceDrawer)
