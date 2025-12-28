"""
Boids Flocking Model
===================
A Mesa implementation of Craig Reynolds's Boids flocker model.
Uses the experimental World system for spatial operations.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../../.."))


import numpy as np

from mesa import Model
from mesa.examples.basic.boid_flockers.agents import Boid
from mesa.experimental.world import World


class BoidFlockers(Model):
    """Flocker model class. Handles agent creation, placement and scheduling."""

    def __init__(
        self,
        population_size=100,
        width=100,
        height=100,
        speed=1,
        vision=10,
        separation=2,
        cohere=0.03,
        separate=0.015,
        match=0.05,
        seed=None,
    ):
        """Create a new Boids Flocking model.

        Args:
            population_size: Number of Boids in the simulation (default: 100)
            width: Width of the space (default: 100)
            height: Height of the space (default: 100)
            speed: How fast the Boids move (default: 1)
            vision: How far each Boid can see (default: 10)
            separation: Minimum distance between Boids (default: 2)
            cohere: Weight of cohesion behavior (default: 0.03)
            separate: Weight of separation behavior (default: 0.015)
            match: Weight of alignment behavior (default: 0.05)
            seed: Random seed for reproducibility (default: None)
        """
        super().__init__(seed=seed)

        # Holds the angle representing the direction of all agents at a given step
        self.agent_angles = np.zeros(population_size)

        # Set up the spatial world
        self.world = World(self, x=(0, width), y=(0, height), torus=True)

        print(f"Number of agents in world before creation: {len(list(self.world.agents))}")

        # Create and place the Boid agents
        positions = self.rng.random(size=(population_size, 2)) * [width, height]
        directions = self.rng.uniform(-1, 1, size=(population_size, 2))

        Boid.create_agents(
            self,
            population_size,
            position=positions,
            direction=directions,
            cohere=cohere,
            separate=separate,
            match=match,
            speed=speed,
            vision=vision,
            separation=separation,
        )

        print(f"Number of agents in world after creation: {len(list(self.world.agents))}")

        # For tracking statistics
        self.average_heading = None
        self.update_average_heading()

    def calculate_angles(self):
        """Vectorized calculation of angles for all agents."""
        d1 = np.array([agent.direction[0] for agent in self.agents])
        d2 = np.array([agent.direction[1] for agent in self.agents])
        self.agent_angles = np.degrees(np.arctan2(d1, d2))
        for agent, angle in zip(self.agents, self.agent_angles):
            agent.angle = angle

    def update_average_heading(self):
        """Calculate the average heading (direction) of all Boids."""
        if not self.agents:
            self.average_heading = 0
            return

        headings = np.array([agent.direction for agent in self.agents])
        mean_heading = np.mean(headings, axis=0)
        self.average_heading = np.arctan2(mean_heading[1], mean_heading[0])

    def step(self):
        """Run one step of the model.

        All agents are activated in random order using the AgentSet shuffle_do method.
        """
        self.agents.shuffle_do("step")
        self.update_average_heading()
        self.calculate_angles()
