"""Adaptive Voronoi Space Module.

This module provides the AdaptiveVoronoiSpace class, which extends
ContinuousSpace to support dynamic region clustering based on agent density.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

from mesa.space import ContinuousSpace


class AdaptiveVoronoiSpace(ContinuousSpace):
    """A continuous space that dynamically clusters agents into Voronoi regions.

    This space allows for 're-tessellation' where agents are grouped into
    clusters (cells) based on their spatial density (using K-Means).

    Attributes:
        n_clusters (int): The target number of Voronoi cells.
        centroids (np.ndarray): (K, 2) array of current cell centers.
        cell_tree (cKDTree): Spatial index for determining cell membership.
        agent_labels (np.ndarray): (N,) array mapping each agent to a cluster ID.
    """

    def __init__(
        self,
        x_max: float,
        y_max: float,
        torus: bool,
        x_min: float = 0,
        y_min: float = 0,
        n_clusters: int = 5,
        clustering_kwargs: dict | None = None,
    ) -> None:
        """Create a new adaptive Voronoi space.

        Args:
            x_max: Maximum x-coordinate.
            y_max: Maximum y-coordinate.
            torus: Boolean for whether the edges loop around.
            x_min: Minimum x-coordinate.
            y_min: Minimum y-coordinate.
            n_clusters: Number of dynamic regions to generate.
            clustering_kwargs: Arguments passed to sklearn.cluster.KMeans.
        """
        super().__init__(x_max, y_max, torus, x_min, y_min)
        self.n_clusters = n_clusters
        self.centroids = None
        self.cell_tree = None
        self.agent_labels = None

        self.clustering_kwargs = clustering_kwargs if clustering_kwargs else {}
        if "n_init" not in self.clustering_kwargs:
            self.clustering_kwargs["n_init"] = 10

    def rebuild_voronoi(self) -> None:
        """Re-calculates the Voronoi tessellation based on agent positions."""
        if self._agent_points is None:
            self._build_agent_cache()

        n_agents = len(self._agent_points)

        if n_agents < self.n_clusters:
            warnings.warn(
                f"Not enough agents ({n_agents}) for {self.n_clusters} clusters. "
                "Skipping Voronoi rebuild.",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        kmeans = KMeans(n_clusters=self.n_clusters, **self.clustering_kwargs)
        self.agent_labels = kmeans.fit_predict(self._agent_points)
        self.centroids = kmeans.cluster_centers_
        self.cell_tree = cKDTree(self.centroids)

    def get_region_id(self, pos: tuple[float, float] | np.ndarray) -> int:
        """Return the cluster ID for a specific spatial position."""
        if self.cell_tree is None:
            raise RuntimeError(
                "Voronoi regions not initialized. Call rebuild_voronoi() first."
            )

        _, region_id = self.cell_tree.query(pos)
        return int(region_id)

    def get_voronoi_neighbors(self, agent, include_center: bool = False) -> list:
        """Return all agents belonging to the same Voronoi cell."""
        if self.agent_labels is None:
            self.rebuild_voronoi()

        if agent not in self._agent_to_index:
            raise ValueError("Agent not found in space")

        idx = self._agent_to_index[agent]
        target_label = self.agent_labels[idx]

        same_cluster_indices = np.where(self.agent_labels == target_label)[0]

        neighbors = []
        for i in same_cluster_indices:
            if not include_center and i == idx:
                continue
            neighbors.append(self._index_to_agent[i])

        return neighbors
