"""Experimental spatial world module for Mesa.

This module provides continuous spatial operations that serve as the foundation
for agent-based models with geometric relationships. All spatial functionality
that shares a coordinate system lives under the World container.

Key concepts:
    - agent.position is the single source of truth for agent locations
    - World provides continuous spatial operations (distance, direction, movement)
    - Spatial layers (future) will build upon these operations while sharing coordinates
"""

from mesa.experimental.world.world import CoordinateSystem, World

__all__ = ["CoordinateSystem", "World"]
