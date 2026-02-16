"""
Boids Flocking Model
===================
A Mesa implementation of Craig Reynolds's Boids flocker model.
Uses numpy arrays to represent vectors.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../../.."))


import numpy as np

from mesa import Model
from mesa.examples.basic.boid_flockers.agents import Boid
from mesa.experimental.continuous_space import ContinuousSpace
from mesa.experimental.scenarios import Scenario


class BoidsScenario(Scenario):
    """Scenario parameters for the Boids Flocking model.

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
        rng: Random rng for reproducibility (default: None)
    """

    population_size: int = 100
    width: int = 100
    height: int = 100
    speed: float = 1.0
    vision: float = 10.0
    separation: float = 2.0
    cohere: float = 0.03
    separate: float = 0.015
    match: float = 0.05


class BoidFlockers(Model):
    """Flocker model class. Handles agent creation, placement and scheduling."""

    def __init__(self, scenario=None):
        """Create a new Boids Flocking model.

        Args:
            scenario: BoidsScenario object containing model parameters.
        """
        if scenario is None:
            scenario = BoidsScenario()

        super().__init__(scenario=scenario)

        self.agent_angles = np.zeros(
            scenario.population_size
        )  # holds the angle representing the direction of all agents at a given step

        # Set up the space
        self.space = ContinuousSpace(
            [[0, scenario.width], [0, scenario.height]],
            torus=True,
            random=self.random,
            n_agents=scenario.population_size,
        )

        # Create and place the Boid agents
        positions = (
            self.rng.random(size=(scenario.population_size, 2)) * self.space.size
        )
        directions = self.rng.uniform(-1, 1, size=(scenario.population_size, 2))
        Boid.create_agents(
            self,
            scenario.population_size,
            self.space,
            position=positions,
            direction=directions,
            cohere=scenario.cohere,
            separate=scenario.separate,
            match=scenario.match,
            speed=scenario.speed,
            vision=scenario.vision,
            separation=scenario.separation,
        )

        # For tracking statistics
        self.average_heading = None
        self.update_average_heading()

    # vectorizing the calculation of angles for all agents
    def calculate_angles(self):
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
