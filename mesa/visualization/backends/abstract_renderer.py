"""Abstract base class for visualization backends in Mesa.

This module provides the foundational interface for implementing various
visualization backends for Mesa agent-based models.
"""

from abc import ABC, abstractmethod

import mesa
from mesa.discrete_space import (
    DiscreteSpace,
    OrthogonalMooreGrid,
    OrthogonalVonNeumannGrid,
)

OrthogonalGrid = OrthogonalMooreGrid | OrthogonalVonNeumannGrid
HexGrid = mesa.discrete_space.HexGrid
Network = mesa.discrete_space.Network


class AbstractRenderer(ABC):
    """Abstract base class for visualization backends.

    This class defines the interface for rendering Mesa spaces and agents.
    For details on the methods checkout specific backend implementations.
    """

    def __init__(self, space_drawer):
        """Initialize the renderer.

        Args:
            space_drawer: Object responsible for drawing space elements. Checkout `SpaceDrawer`
            for more details on the detailed implementations of the drawing functions.
        """
        self.space_drawer = space_drawer
        self._canvas = None

    def _get_agent_pos(self, agent, space):
        """Get agent position based on space type."""
        if isinstance(space, DiscreteSpace):
            return agent.cell.position
        else:
            return agent.position

    @abstractmethod
    def initialize_canvas(self):
        """Set up the drawing canvas."""

    @abstractmethod
    def draw_structure(self, **kwargs):
        """Draw the space structure.

        Args:
            **kwargs: Structure drawing configuration options.
        """

    @abstractmethod
    def collect_agent_data(self, space, agent_portrayal, default_size=None):
        """Collect plotting data for all agents in the space.

        Args:
            space: The Mesa space containing agents.
            agent_portrayal (Callable): Function that returns AgentPortrayalStyle for each agent.
            default_size (float, optional): Default marker size if not specified in portrayal.

        Returns:
            dict: Dictionary containing agent plotting data arrays with keys:
        """

    @abstractmethod
    def draw_agents(self, arguments, **kwargs):
        """Drawing agents on space.

        Args:
            arguments (dict): Dictionary containing agent data.
            **kwargs: Additional drawing configuration options.
        """

    @abstractmethod
    def draw_propertylayer(self, space, property_layers, propertylayer_portrayal):
        """Draw property layers on the visualization.

        Args:
            space: The model's space object.
            property_layers (dict): Dictionary of property layers to visualize.
            propertylayer_portrayal (Callable): Function that returns PropertyLayerStyle.
        """
