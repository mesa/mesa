"""Experimental spatial world module for Mesa.

This module provides the foundation for geometric operations on coordinates,
supporting continuous spatial operations that other spatial layers can build upon.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from mesa import Agent


class CoordinateSystem:
    """Defines a coordinate system with bounds and wrapping behavior.

    Attributes:
        bounds: List of (min, max) tuples for each dimension
        torus: Whether coordinates wrap around at boundaries
        dimensions: Number of dimensions (2D, 3D, etc.)
    """

    def __init__(
        self,
        bounds: list[tuple[float, float]] | None = None,
        torus: bool = False,
        **kwargs: tuple[float, float],
    ):
        """Initialize a coordinate system.

        Args:
            bounds: List of (min, max) tuples for each dimension
            torus: Whether coordinates wrap around at boundaries
            **kwargs: Alternative specification using x=(min, max), y=(min, max), etc.

        Examples:
            # 2D coordinate system
            coords = CoordinateSystem(bounds=[(0, 100), (0, 100)])

            # Alternative syntax
            coords = CoordinateSystem(x=(0, 100), y=(0, 100))

            # 3D with wrapping
            coords = CoordinateSystem(x=(0, 50), y=(0, 50), z=(0, 50), torus=True)
        """
        if bounds is not None and kwargs:
            raise ValueError("Specify either bounds or keyword arguments, not both")

        if kwargs:
            # Sort by key to ensure consistent ordering (x, y, z, ...)
            sorted_keys = sorted(kwargs.keys())
            bounds = [kwargs[k] for k in sorted_keys]

        if bounds is None:
            raise ValueError("Must specify coordinate bounds")

        self.bounds = bounds
        self.torus = torus
        self.dimensions = len(bounds)

        # Pre-compute sizes for each dimension
        self._sizes = np.array([b[1] - b[0] for b in bounds])
        self._mins = np.array([b[0] for b in bounds])

    def validate(self, position: np.ndarray | list[float]) -> bool:
        """Check if position is within bounds (for non-torus spaces).

        Args:
            position: Coordinate to validate

        Returns:
            True if position is valid, False otherwise
        """
        pos = np.asarray(position)
        if pos.shape[0] != self.dimensions:
            return False

        for i, (min_val, max_val) in enumerate(self.bounds):
            if pos[i] < min_val or pos[i] > max_val:
                return False
        return True

    def wrap(self, position: np.ndarray | list[float]) -> np.ndarray:
        """Wrap coordinates if torus is enabled.

        Args:
            position: Coordinate to wrap

        Returns:
            Wrapped coordinate (or original if torus is False)
        """
        pos = np.asarray(position, dtype=float)

        if not self.torus:
            return pos

        # Wrap each dimension
        wrapped = (pos - self._mins) % self._sizes + self._mins
        return wrapped

    def random_position(self, rng: np.random.Generator) -> np.ndarray:
        """Generate a random position within the coordinate system.

        Args:
            rng: NumPy random number generator

        Returns:
            Random position within bounds
        """
        return np.array(
            [rng.uniform(min_val, max_val) for min_val, max_val in self.bounds]
        )


class World:
    """Container for spatial operations and layers.

    All spatial functionality that shares a coordinate system lives here.
    Provides continuous spatial operations (distance, direction, movement)
    that spatial layers can build upon.

    Attributes:
        coords: The coordinate system defining bounds and wrapping
    """

    def __init__(
        self,
        coords: CoordinateSystem | None = None,
        bounds: list[tuple[float, float]] | None = None,
        torus: bool = False,
        rng=None,
        **kwargs: tuple[float, float],
    ):
        """Initialize a spatial world.

        Args:
            coords: Pre-configured coordinate system
            bounds: List of (min, max) tuples for each dimension (if coords not provided)
            torus: Whether coordinates wrap around at boundaries (if coords not provided)
            **kwargs: Alternative bound specification (x=(min, max), y=(min, max), etc.)

        Examples:
            # Using CoordinateSystem
            world = World(coords=CoordinateSystem(x=(0, 100), y=(0, 100)))

            # Direct specification
            world = World(x=(0, 100), y=(0, 100), torus=False)

            # With bounds list
            world = World(bounds=[(0, 100), (0, 100)], torus=True)
        """
        if coords is not None:
            if bounds is not None or kwargs:
                raise ValueError("Specify either coords or bounds/kwargs, not both")
            self.coords = coords
        else:
            self.coords = CoordinateSystem(bounds=bounds, torus=torus, **kwargs)

        self._rng = rng  # Set by model when world is created

        # Storage for spatial layers (for future extension)
        self._layers: dict[str, Any] = {}

    def add_layer(self, name: str, layer: Any) -> None:
        """Add a spatial layer to the world.

        Args:
            name: Name for the layer
            layer: Spatial layer object (Grid, VoronoiGrid, Network, etc.)

        Notes:
            This is a placeholder for future spatial layer integration.
        """
        self._layers[name] = layer

    def get_layer(self, name: str) -> Any:
        """Get a spatial layer by name.

        Args:
            name: Name of the layer

        Returns:
            The spatial layer object

        Raises:
            KeyError: If layer doesn't exist
        """
        return self._layers[name]

    def __getattr__(self, name: str) -> Any:
        """Allow access to layers as attributes.

        Args:
            name: Layer name

        Returns:
            The layer object

        Raises:
            AttributeError: If layer doesn't exist
        """
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        try:
            return self._layers[name]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

    def random_position(self):
        if self._rng is None:
            raise ValueError("World not associated with a model")
        return self.coords.random_position(self._rng)

    def distance(
        self,
        pos1: np.ndarray | list[float] | Agent,
        pos2: np.ndarray | list[float] | Agent,
    ) -> float:
        """Calculate distance between two positions or agents.

        Args:
            pos1: First position or agent
            pos2: Second position or agent

        Returns:
            Euclidean distance between positions
        """
        p1 = self._extract_position(pos1)
        p2 = self._extract_position(pos2)

        if self.coords.torus:
            # Calculate distance considering wrapping
            delta = np.abs(p1 - p2)
            # For each dimension, take the shorter path (direct or wrapped)
            for i in range(self.coords.dimensions):
                size = self.coords._sizes[i]
                if delta[i] > size / 2:
                    delta[i] = size - delta[i]
            return float(np.linalg.norm(delta))
        else:
            return float(np.linalg.norm(p1 - p2))

    def direction(
        self,
        pos1: np.ndarray | list[float] | Agent,
        pos2: np.ndarray | list[float] | Agent,
    ) -> np.ndarray:
        """Calculate direction vector from pos1 to pos2.

        Args:
            pos1: Source position or agent
            pos2: Target position or agent

        Returns:
            Unit vector pointing from pos1 to pos2
        """
        p1 = self._extract_position(pos1)
        p2 = self._extract_position(pos2)

        if self.coords.torus:
            # Calculate direction considering wrapping
            delta = p2 - p1
            for i in range(self.coords.dimensions):
                size = self.coords._sizes[i]
                if delta[i] > size / 2:
                    delta[i] -= size
                elif delta[i] < -size / 2:
                    delta[i] += size
        else:
            delta = p2 - p1

        norm = np.linalg.norm(delta)
        if norm == 0:
            return np.zeros(self.coords.dimensions)
        return delta / norm

    def move_toward(
        self,
        agent: Agent,
        target: np.ndarray | list[float] | Agent,
        speed: float,
    ) -> None:
        """Move an agent toward a target position.

        Args:
            agent: Agent to move
            target: Target position or agent
            speed: Distance to move (can be larger than distance to target)
        """
        direction = self.direction(agent.position, target)
        new_position = agent.position + direction * speed
        agent.position = self.coords.wrap(new_position)

    def move_by(
        self,
        agent: Agent,
        delta: np.ndarray | list[float],
    ) -> None:
        """Move an agent by a relative offset.

        Args:
            agent: Agent to move
            delta: Offset to add to current position
        """
        new_position = agent.position + np.asarray(delta)
        agent.position = self.coords.wrap(new_position)

    def get_neighbors_in_radius(
        self,
        agent: Agent,
        radius: float,
        agents: list[Agent] | None = None,
    ) -> list[Agent]:
        """Get all agents within a radius of the given agent.

        Args:
            agent: Center agent
            radius: Search radius
            agents: List of agents to search (defaults to all model agents)

        Returns:
            List of agents within radius (excluding the center agent)
        """
        if agents is None:
            agents = list(agent.model.agents)

        center_pos = agent.position
        neighbors = []

        for other in agents:
            if other is agent:
                continue
            if not hasattr(other, "position") or other.position is None:
                continue

            dist = self.distance(center_pos, other.position)
            if dist <= radius:
                neighbors.append(other)

        return neighbors

    def _extract_position(
        self,
        pos_or_agent: np.ndarray | list[float] | Agent,
    ) -> np.ndarray:
        """Extract position from an agent or position array.

        Args:
            pos_or_agent: Position array or agent with position attribute

        Returns:
            Position as numpy array
        """
        if isinstance(pos_or_agent, Agent):
            if not hasattr(pos_or_agent, "position") or pos_or_agent.position is None:
                raise ValueError(f"Agent {pos_or_agent} has no position")
            return np.asarray(pos_or_agent.position)
        return np.asarray(pos_or_agent)
