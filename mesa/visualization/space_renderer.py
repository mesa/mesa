"""Space rendering module for Mesa visualizations.

This module provides functionality to render Mesa model spaces with different
backends, supporting various space types and visualization components.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from mesa.visualization.components import PropertyLayerStyle

import altair as alt
import pandas as pd

import mesa
from mesa.discrete_space import (
    OrthogonalMooreGrid,
    OrthogonalVonNeumannGrid,
    VoronoiGrid,
)
from mesa.experimental.continuous_space import ContinuousSpace
from mesa.visualization.backends import AltairBackend, MatplotlibBackend
from mesa.visualization.space_drawers import (
    ContinuousSpaceDrawer,
    HexSpaceDrawer,
    NetworkSpaceDrawer,
    OrthogonalSpaceDrawer,
    VoronoiSpaceDrawer,
)

OrthogonalGrid = OrthogonalMooreGrid | OrthogonalVonNeumannGrid
HexGrid = mesa.discrete_space.HexGrid
Network = mesa.discrete_space.Network


class SpaceRenderer:
    """Renders Mesa spaces using different visualization backends.

    Supports multiple space types and backends for flexible visualization
    of agent-based models.
    """

    def __init__(
        self,
        model: mesa.Model,
        backend: Literal["matplotlib", "altair"] | None = "matplotlib",
    ):
        """Initialize the space renderer.

        Args:
            model (mesa.Model): The Mesa model to render.
            backend (Literal["matplotlib", "altair"] | None): The visualization backend to use.
        """
        self.space = getattr(model, "grid", getattr(model, "space", None))

        self.space_drawer = self._get_space_drawer()

        self.space_mesh = None
        self.agent_mesh = None
        self.propertylayer_mesh = None

        self.draw_agent_kwargs = {}
        self.draw_space_kwargs = {}

        self.agent_portrayal = None
        self.propertylayer_portrayal = None

        self.post_process_func = None
        # Keep track of whether post-processing has been applied
        # to avoid multiple applications on the same axis.
        self._post_process_applied = False

        self.backend = backend

        if backend == "matplotlib":
            self.backend_renderer = MatplotlibBackend(
                self.space_drawer,
            )
        elif backend == "altair":
            self.backend_renderer = AltairBackend(
                self.space_drawer,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        self.backend_renderer.initialize_canvas()

    def _get_space_drawer(self):
        """Get appropriate space drawer based on space type.

        Returns:
            Space drawer instance for the model's space type.

        Raises:
            ValueError: If the space type is not supported.
        """
        if isinstance(self.space, HexGrid):
            return HexSpaceDrawer(self.space)
        elif isinstance(self.space, OrthogonalGrid):
            return OrthogonalSpaceDrawer(self.space)
        elif isinstance(
            self.space,
            mesa.experimental.continuous_space.ContinuousSpace,
        ):
            return ContinuousSpaceDrawer(self.space)
        elif isinstance(self.space, VoronoiGrid):
            return VoronoiSpaceDrawer(self.space)
        elif isinstance(self.space, Network):
            return NetworkSpaceDrawer(self.space)
        raise ValueError(
            f"Unsupported space type: {type(self.space).__name__}. "
            "Supported types are OrthogonalGrid, HexGrid, ContinuousSpace, VoronoiGrid, and Network."
        )

    def _map_coordinates(self, arguments):
        """Map agent coordinates to appropriate space coordinates.

        Args:
            arguments (dict): Dictionary containing agent data with coordinates.

        Returns:
            dict: Dictionary with mapped coordinates appropriate for the space type.
        """
        mapped_arguments = arguments.copy()

        if isinstance(
            self.space, OrthogonalGrid | VoronoiGrid | ContinuousSpace | Network
        ):
            # Use the coordinates directly for Orthogonal grids, Voronoi grids and Continuous spaces
            mapped_arguments["loc"] = arguments["loc"].astype(float)

        elif isinstance(self.space, HexGrid):
            # Map rectangular coordinates to hexagonal grid coordinates
            loc = arguments["loc"].astype(float)
            if loc.size > 0:
                # Calculate hexagon centers
                loc[:, 0] = loc[:, 0] * self.space_drawer.x_spacing + (
                    (loc[:, 1] - 1) % 2
                ) * (self.space_drawer.x_spacing / 2)
                loc[:, 1] = loc[:, 1] * self.space_drawer.y_spacing
            mapped_arguments["loc"] = loc

        return mapped_arguments

    def setup_structure(self, **kwargs) -> SpaceRenderer:
        """Setup the space structure without drawing.

        Args:
            **kwargs: Additional keyword arguments for the setup function.
            Checkout respective `SpaceDrawer` class on details how to pass **kwargs.

        Returns:
            SpaceRenderer: The current instance for method chaining.
        """
        self.draw_space_kwargs = kwargs
        self.space_mesh = None

        return self

    def setup_agents(self, agent_portrayal: Callable, **kwargs) -> SpaceRenderer:
        """Setup agents on the space without drawing.

        Args:
            agent_portrayal (Callable): Function that takes an agent and returns AgentPortrayalStyle.
            **kwargs: Additional keyword arguments for the setup function.
            Checkout respective `SpaceDrawer` class on details how to pass **kwargs.

        Returns:
            SpaceRenderer: The current instance for method chaining.
        """
        self.agent_portrayal = agent_portrayal
        self.draw_agent_kwargs = kwargs
        self.agent_mesh = None

        return self

    def setup_propertylayer(
        self, propertylayer_portrayal: Callable | dict | PropertyLayerStyle
    ) -> SpaceRenderer:
        """Setup property layers on the space without drawing.

        Args:
            propertylayer_portrayal (Callable | dict | PropertyLayerStyle): A PropertyLayerStyle,
                a function that produces a PropertyLayerStyle instance, or a dictionary specifying portrayal parameters.

        Returns:
            SpaceRenderer: The current instance for method chaining.
        """
        self.propertylayer_portrayal = propertylayer_portrayal
        self.propertylayer_mesh = None

        return self

    def draw_structure(self, **kwargs):
        """Draw the space structure.

        Args:
            **kwargs: (Deprecated) Additional keyword arguments for drawing.
                    Use setup_structure() instead.

        Returns:
            The visual representation of the space structure.
        """
        if kwargs:
            warnings.warn(
                "Passing kwargs to draw_structure() is deprecated. "
                "Use setup_structure(**kwargs) before calling draw_structure().",
                PendingDeprecationWarning,
                stacklevel=2,
            )
            self.draw_space_kwargs.update(kwargs)

        # Network-specific: the space instance is replaced on every model reset.
        # If the drawer still references the old space its layout positions belong
        # to the previous (now stale) graph.  Rebuild before drawing edges so that
        # the structure is always consistent with the current space.
        if (
            isinstance(self.space, Network)
            and self.space_drawer.space is not self.space
        ):
            self.space_drawer = self._get_space_drawer()
            self.backend_renderer.space_drawer = self.space_drawer

        self.space_mesh = self.backend_renderer.draw_structure(**self.draw_space_kwargs)
        return self.space_mesh

    def draw_agents(self, agent_portrayal=None, **kwargs):
        """Draw agents on the space.

        Args:
            agent_portrayal: (Deprecated) Function that takes an agent and returns AgentPortrayalStyle.
                            Use setup_agents() instead.
            **kwargs: (Deprecated) Additional keyword arguments for drawing.

        Returns:
            The visual representation of the agents.
        """
        if agent_portrayal is not None:
            warnings.warn(
                "Passing agent_portrayal to draw_agents() is deprecated and will be removed in Mesa 4.0. "
                "Use setup_agents(agent_portrayal, **kwargs) before calling draw_agents()."
                "See https://mesa.readthedocs.io/latest/migration_guide.html#passing-portrayal-arguments-to-draw-methods",
                FutureWarning,
                stacklevel=2,
            )
            self.agent_portrayal = agent_portrayal
        if kwargs:
            warnings.warn(
                "Passing kwargs to draw_agents() is deprecated. "
                "Use setup_agents(**kwargs) before calling draw_agents().",
                PendingDeprecationWarning,
                stacklevel=2,
            )
            self.draw_agent_kwargs.update(kwargs)

        # Prepare data for agent plotting
        arguments = self.backend_renderer.collect_agent_data(
            self.space, self.agent_portrayal, default_size=self.space_drawer.s_default
        )
        arguments = self._map_coordinates(arguments)

        self.agent_mesh = self.backend_renderer.draw_agents(
            arguments, **self.draw_agent_kwargs
        )
        return self.agent_mesh

    def draw_propertylayer(self, propertylayer_portrayal=None):
        """Draw property layers on the space.

        Args:
            propertylayer_portrayal: (Deprecated) A PropertyLayerStyle, a function that produces
            a PropertyLayerStyle instance, or a dictionary specifying portrayal parameters.
            Use setup_propertylayer() instead.

        Returns:
            The visual representation of the property layers.

        Raises:
            Exception: If no property layers are found on the space.
        """
        if propertylayer_portrayal is not None:
            warnings.warn(
                "Passing propertylayer_portrayal to draw_propertylayer() is deprecated and will be removed in Mesa 4.0. "
                "Use setup_propertylayer(propertylayer_portrayal) before calling draw_propertylayer()."
                "See https://mesa.readthedocs.io/latest/migration_guide.html#passing-portrayal-arguments-to-draw-methods",
                FutureWarning,
                stacklevel=2,
            )
            self.propertylayer_portrayal = propertylayer_portrayal

        # Import here to avoid circular imports
        from mesa.visualization.components import PropertyLayerStyle  # noqa: PLC0415

        def _dict_to_callable(portrayal_dict):
            """Convert legacy dict portrayal to callable.

            Args:
                portrayal_dict (dict): Dictionary with portrayal parameters.

            Returns:
                Callable: Function that returns PropertyLayerStyle.
            """

            def style_callable(layer_object):
                layer_name = layer_object.name
                params = portrayal_dict.get(layer_name)

                warnings.warn(
                    (
                        "The propertylayer_portrayal dict is deprecated and will be removed in Mesa 4.0. "
                        "Please use a callable that returns a PropertyLayerStyle instance instead. "
                        "For more information, refer to the migration guide: "
                        "https://mesa.readthedocs.io/latest/migration_guide.html#defining-portrayal-components"
                    ),
                    FutureWarning,
                    stacklevel=2,
                )

                if params is None:
                    return None

                return PropertyLayerStyle(
                    color=params.get("color"),
                    colormap=params.get("colormap"),
                    alpha=params.get("alpha", PropertyLayerStyle.alpha),
                    vmin=params.get("vmin"),
                    vmax=params.get("vmax"),
                    colorbar=params.get("colorbar", PropertyLayerStyle.colorbar),
                )

            return style_callable

        # Get property layers
        try:
            # old style spaces
            property_layers = self.space.properties
        except AttributeError:
            # new style spaces
            property_layers = self.space._mesa_property_layers

        # Convert portrayal to callable if needed
        if isinstance(self.propertylayer_portrayal, dict):
            self.propertylayer_portrayal = _dict_to_callable(
                self.propertylayer_portrayal
            )
        elif isinstance(self.propertylayer_portrayal, PropertyLayerStyle):
            # Capture the style instance to avoid circular reference
            style = self.propertylayer_portrayal
            self.propertylayer_portrayal = lambda _: style
        # else: already a callable, use as-is

        number_of_propertylayers = sum(
            [1 for layer in property_layers if layer != "empty"]
        )
        if number_of_propertylayers < 1:
            raise Exception("No property layers were found on the space.")

        self.propertylayer_mesh = self.backend_renderer.draw_propertylayer(
            self.space, property_layers, self.propertylayer_portrayal
        )
        return self.propertylayer_mesh

    def render(self, agent_portrayal=None, propertylayer_portrayal=None, **kwargs):
        """Render the complete space with structure, agents, and property layers.

        Args:
            agent_portrayal: (Deprecated) Function for agent portrayal. Use setup_agents() instead.
            propertylayer_portrayal: (Deprecated) Function for property layer portrayal. Use setup_propertylayer() instead.
            **kwargs: (Deprecated) Additional keyword arguments.
        """
        if agent_portrayal is not None or propertylayer_portrayal is not None or kwargs:
            warnings.warn(
                "Passing parameters to render() is deprecated. "
                "Use setup_structure(), setup_agents(), and setup_propertylayer() before calling render().",
                PendingDeprecationWarning,
                stacklevel=2,
            )
            if agent_portrayal is not None:
                self.agent_portrayal = agent_portrayal
            if propertylayer_portrayal is not None:
                self.propertylayer_portrayal = propertylayer_portrayal

            deprecated_kwargs_map = {
                "space_kwargs": self.draw_space_kwargs,
                "agent_kwargs": self.draw_agent_kwargs,
            }
            for key, target_dict in deprecated_kwargs_map.items():
                if key in kwargs:
                    value = kwargs.pop(key)
                    if isinstance(value, dict):
                        target_dict.update(value)

            # Update with any remaining kwargs (now that the dangerous ones are removed)
            self.draw_space_kwargs.update(kwargs)

        if self.space_mesh is None:
            self.draw_structure()
        if self.agent_mesh is None and self.agent_portrayal is not None:
            self.draw_agents()
        if self.propertylayer_mesh is None and self.propertylayer_portrayal is not None:
            self.draw_propertylayer()

        return self

    @property
    def canvas(self):
        """Get the current canvas object.

        Returns:
            The backend-specific canvas object.
        """
        if self.backend == "matplotlib":
            ax = self.backend_renderer.ax
            if ax is None:
                self.backend_renderer.initialize_canvas()
            return ax

        elif self.backend == "altair":
            structure = self.space_mesh if self.space_mesh else None
            agents = self.agent_mesh if self.agent_mesh else None
            prop_base, prop_cbar = self.propertylayer_mesh or (None, None)

            if self.space_mesh:
                structure = self.draw_structure()
            if self.agent_mesh:
                agents = self.draw_agents()
            if self.propertylayer_mesh:
                prop_base, prop_cbar = self.draw_propertylayer()

            spatial_charts_list = [
                chart for chart in [structure, prop_base, agents] if chart
            ]

            main_spatial = None
            if spatial_charts_list:
                main_spatial = (
                    spatial_charts_list[0]
                    if len(spatial_charts_list) == 1
                    else alt.layer(*spatial_charts_list)
                )

            # Determine final chart by combining with color bar if present
            final_chart = None
            if main_spatial and prop_cbar:
                final_chart = alt.vconcat(main_spatial, prop_cbar).configure_view(
                    stroke=None
                )
            elif main_spatial:  # Only main_spatial, no prop_cbar
                final_chart = main_spatial
            elif prop_cbar:  # Only prop_cbar, no main_spatial
                final_chart = prop_cbar
                final_chart = final_chart.configure_view(grid=False)

            if final_chart is None:
                # If no charts are available, return an empty chart
                final_chart = (
                    alt.Chart(pd.DataFrame())
                    .mark_point()
                    .properties(width=450, height=350)
                )

            final_chart = final_chart.configure_view(stroke="black", strokeWidth=1.5)

            return final_chart

    @property
    def post_process(self):
        """Get the current post-processing function.

        Returns:
            Callable | None: The post-processing function, or None if not set.
        """
        return self.post_process_func

    @post_process.setter
    def post_process(self, func: Callable | None):
        """Set the post-processing function.

        Args:
            func (Callable | None): Function to apply post-processing to the canvas.
                Should accept the canvas object as its first argument.
        """
        self.post_process_func = func
