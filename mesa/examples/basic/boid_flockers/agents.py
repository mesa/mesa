"""A Boid (bird-oid) agent for implementing Craig Reynolds's Boids flocking model.

This implementation uses numpy arrays to represent vectors for efficient computation
of flocking behavior, and the experimental World system for spatial operations.
"""

import numpy as np

from mesa import Agent


class Boid(Agent):
    """A Boid-style flocker agent.

    The agent follows three behaviors to flock:
        - Cohesion: steering towards neighboring agents
        - Separation: avoiding getting too close to any other agent
        - Alignment: trying to fly in the same direction as neighbors

    Boids have a vision that defines the radius in which they look for their
    neighbors to flock with. Their speed (a scalar) and direction (a vector)
    define their movement. Separation is their desired minimum distance from
    any other Boid.
    """

    def __init__(
        self,
        model,
        position=(0, 0),
        speed=1,
        direction=(1, 1),
        vision=1,
        separation=1,
        cohere=0.03,
        separate=0.015,
        match=0.05,
    ):
        """Create a new Boid flocker agent.

        Args:
            model: Model instance the agent belongs to
            position: Initial position as (x, y) tuple or array
            speed: Distance to move per step
            direction: numpy vector for the Boid's direction of movement
            vision: Radius to look around for nearby Boids
            separation: Minimum distance to maintain from other Boids
            cohere: Relative importance of matching neighbors' positions (default: 0.03)
            separate: Relative importance of avoiding close neighbors (default: 0.015)
            match: Relative importance of matching neighbors' directions (default: 0.05)
        """
        super().__init__(model)
        self.position = np.asarray(position, dtype=float)
        self.speed = speed
        self.direction = np.asarray(direction, dtype=float)
        # Normalize initial direction
        self.direction /= np.linalg.norm(self.direction)

        self.vision = vision
        self.separation = separation
        self.cohere_factor = cohere
        self.separate_factor = separate
        self.match_factor = match
        self.neighbors = []
        self.angle = 0.0  # represents the angle at which the boid is moving

    def step(self):
        """Get the Boid's neighbors, compute the new vector, and move accordingly."""
        # Get neighbors within vision radius using world
        neighbors = self.world.get_neighbors_in_radius(self, radius=self.vision)
        self.neighbors = neighbors

        # If no neighbors, maintain current direction
        if not neighbors:
            self.world.move_by(self, self.direction * self.speed)
            return

        # Calculate vectors to neighbors (cohesion component)
        neighbor_positions = np.array([n.position for n in neighbors])
        deltas = neighbor_positions - self.position

        # Handle torus wrapping for deltas
        if self.world.coords.torus:
            for i in range(self.world.coords.dimensions):
                size = self.world.coords._sizes[i]
                deltas[:, i] = np.where(
                    deltas[:, i] > size / 2, deltas[:, i] - size, deltas[:, i]
                )
                deltas[:, i] = np.where(
                    deltas[:, i] < -size / 2, deltas[:, i] + size, deltas[:, i]
                )

        # Calculate distances for separation
        distances = np.linalg.norm(deltas, axis=1)

        # Cohesion: steer towards center of neighbors
        cohere_vector = deltas.sum(axis=0) * self.cohere_factor

        # Separation: avoid crowding neighbors
        close_neighbors = distances < self.separation
        if close_neighbors.any():
            separation_vector = (
                -deltas[close_neighbors].sum(axis=0) * self.separate_factor
            )
        else:
            separation_vector = np.zeros(self.world.coords.dimensions)

        # Alignment: match direction of neighbors
        neighbor_directions = np.array([n.direction for n in neighbors])
        match_vector = neighbor_directions.sum(axis=0) * self.match_factor

        # Update direction based on the three behaviors
        self.direction += (cohere_vector + separation_vector + match_vector) / len(
            neighbors
        )

        # Normalize direction vector
        norm = np.linalg.norm(self.direction)
        if norm > 0:
            self.direction /= norm

        # Move boid
        self.world.move_by(self, self.direction * self.speed)
