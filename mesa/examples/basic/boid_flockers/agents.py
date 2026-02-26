"""A Boid (bird-oid) agent for implementing Craig Reynolds's Boids flocking model.

This implementation uses numpy arrays to represent vectors for efficient computation
of flocking behavior.
"""

import numpy as np

from mesa.experimental.continuous_space import ContinuousSpaceAgent


class Boid(ContinuousSpaceAgent):
    """A Boid-style flocker agent.

    The agent follows three behaviors to flock:
        - Cohesion: steering towards the average position of neighboring agents
        - Separation: avoiding getting too close to any other agent
        - Alignment: trying to fly in the same direction as neighbors

    Boids have a vision that defines the radius in which they look for their
    neighbors to flock with. Their speed (a scalar) and direction (a vector)
    define their movement. Separation is their desired minimum distance from
    any other Boid.

    Cohesion and alignment use distance-weighted averaging (Option 4):
    closer neighbors have stronger influence, controlled by the decay exponent.
    Setting decay=0.0 falls back to canonical equal-weight Reynolds (1986) behavior.
    """

    def __init__(
        self,
        model,
        space,
        position=(0, 0),
        speed=1,
        direction=(1, 1),
        vision=1,
        separation=1,
        cohere=0.03,
        separate=0.015,
        match=0.05,
        decay=2.0,
    ):
        """Create a new Boid flocker agent.

        Args:
            model: Model instance the agent belongs to
            speed: Distance to move per step
            direction: numpy vector for the Boid's direction of movement
            vision: Radius to look around for nearby Boids
            separation: Minimum distance to maintain from other Boids
            cohere: Relative importance of matching neighbors' positions (default: 0.03)
            separate: Relative importance of avoiding close neighbors (default: 0.015)
            match: Relative importance of matching neighbors' directions (default: 0.05)
            decay: Exponent for distance-weighted decay (default: 2.0).
                Higher values mean closer neighbors dominate more strongly.
                Set to 0.0 to use canonical equal-weight Reynolds (1986) behavior.
        """
        super().__init__(space, model)
        self.position = position
        self.speed = speed
        # Always store direction as a normalized numpy float array
        self.direction = np.array(direction, dtype=float)
        norm = np.linalg.norm(self.direction)
        if norm > 0:
            self.direction = self.direction / norm
        self.vision = vision
        self.separation = separation
        self.cohere_factor = cohere
        self.separate_factor = separate
        self.match_factor = match
        self.decay = decay
        self.neighbors = []
        self.angle = 0.0  # represents the angle at which the boid is moving

    def step(self):
        """Get the Boid's neighbors, compute the new vector, and move accordingly."""
        neighbors, distances = self.get_neighbors_in_radius(radius=self.vision)
        self.neighbors = [n for n in neighbors if n is not self]

        # If no neighbors, maintain current direction
        if not self.neighbors:
            # Always normalize before moving to prevent drift
            self.direction = np.array(self.direction, dtype=float)
            norm = np.linalg.norm(self.direction)
            if norm > 0:
                self.direction = self.direction / norm
            self.position += self.direction * self.speed
            return

        # Compute delta vectors only for actual neighbors (exclude self)
        delta = self.space.calculate_difference_vector(
            self.position, agents=self.neighbors
        )

        # Build neighbor distances matching self.neighbors exactly
        neighbor_distances = np.array(
            [d for boid, d in zip(neighbors, distances) if boid is not self]
        )
        # Clamp to avoid near-zero distances causing weight explosion (decay=2 → 1/d²)
        neighbor_distances = np.clip(neighbor_distances, a_min=1e-2, a_max=None)

        # Distance-based weights: closer neighbors have stronger influence
        # decay=2.0 → weight = 1/d² (inverse square law, natural decay)
        # decay=0.0 → weight = 1 for all (falls back to canonical Reynolds)
        weights = 1.0 / (neighbor_distances**self.decay + 1e-6)
        weights /= weights.sum()  # normalize so weights sum to 1

        # Rule 1 — Cohesion: weighted average position of local flockmates
        # Reynolds (1986): steer toward average position of neighbors
        cohere_vector = (delta * weights[:, None]).sum(axis=0) * self.cohere_factor

        # Rule 2 — Separation: avoid crowding neighbors
        # Uses mean (not sum) so force stays bounded regardless of neighbor count
        too_close = neighbor_distances < self.separation
        if too_close.any():
            separation_vector = (
                -1 * delta[too_close].mean(axis=0) * self.separate_factor
            )
        else:
            separation_vector = np.zeros(2)

        # Rule 3 — Alignment: weighted average heading of local flockmates
        # Reynolds (1986): steer toward average heading of neighbors
        directions = np.asarray([n.direction for n in self.neighbors])
        match_vector = (directions * weights[:, None]).sum(axis=0) * self.match_factor

        # Update direction based on the three behaviors
        # No division by len(neighbors) — weights are already normalized
        self.direction += cohere_vector + separation_vector + match_vector

        # Normalize direction vector to maintain constant speed
        self.direction = np.array(self.direction, dtype=float)
        norm = np.linalg.norm(self.direction)
        if norm > 0:
            self.direction = self.direction / norm

        # Move boid
        self.position += self.direction * self.speed
