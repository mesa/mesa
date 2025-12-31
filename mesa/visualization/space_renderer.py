"""Space rendering module for Mesa visualizations.

This module provides functionality to render Mesa model spaces with different
backends, supporting various space types and visualization components.
"""

import contextlib
import warnings
from collections.abc import Callable
from typing import Literal

import altair as alt
import numpy as np
import pandas as pd

import mesa
from mesa.discrete_space import (
    OrthogonalMooreGrid,
    OrthogonalVonNeumannGrid,
    VoronoiGrid,
)
from mesa.space import (
    ContinuousSpace,
    HexMultiGrid,
    HexSingleGrid,
    MultiGrid,
    NetworkGrid,
    SingleGrid,
    _HexGrid,
)
from mesa.visualization.backends import AltairBackend, MatplotlibBackend
from mesa.visualization.components import PropertyLayerStyle
from mesa.visualization.icon_altair_layer import build_altair_agent_chart
from mesa.visualization.icon_cache import IconCache
from mesa.visualization.space_drawers import (
    ContinuousSpaceDrawer,
    HexSpaceDrawer,
    NetworkSpaceDrawer,
    OrthogonalSpaceDrawer,
    VoronoiSpaceDrawer,
)

OrthogonalGrid = SingleGrid | MultiGrid | OrthogonalMooreGrid | OrthogonalVonNeumannGrid
HexGrid = HexSingleGrid | HexMultiGrid | mesa.discrete_space.HexGrid
Network = NetworkGrid | mesa.discrete_space.Network


class SpaceRenderer:
    """Renders Mesa spaces using different visualization backends.

    Supports multiple space types and backends for flexible visualization
    of agent-based models.
    """

    def __init__(
        self,
        model: mesa.Model,
        backend: Literal["matplotlib", "altair"] | None = "matplotlib",
        **kwargs,
    ):
        """Initialize the space renderer.

        Args:
            model: The Mesa model to render.
            backend: The visualization backend to use.
            **kwargs: Additional keyword arguments:
                - icon_mode: "off", "auto", or "force" (default: "off")
                - icon_auto_max_agents: Max agents for auto icon mode (default: 1500)
                - icon_culling: Enable culling for icons (default: True)
        """
        self.space = getattr(model, "grid", getattr(model, "space", None))
        self.space_drawer = self._get_space_drawer()
        self.space_mesh = None
        self.agent_mesh = None
        self.propertylayer_mesh = None
        self.post_process_func = None
        self._post_process_applied = False
        self.backend = backend

        # Icon rendering configuration
        self.icon_mode = kwargs.pop("icon_mode", "off")  # "off", "auto", "force"
        self.icon_auto_max_agents = kwargs.pop("icon_auto_max_agents", 1500)
        self.icon_culling = kwargs.pop("icon_culling", True)
        self._icon_cache = IconCache(backend=self.backend)

        if backend == "matplotlib":
            self.backend_renderer = MatplotlibBackend(self.space_drawer)
        elif backend == "altair":
            self.backend_renderer = AltairBackend(self.space_drawer)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        self.backend_renderer.initialize_canvas()

    def draw_agents(self, agent_portrayal: Callable, **kwargs):
        """Draw agents on the space."""
        self.agent_portrayal = agent_portrayal
        self.agent_kwargs = kwargs

        arguments = self.backend_renderer.collect_agent_data(
            self.space, agent_portrayal, default_size=self.space_drawer.s_default
        )
        arguments = self._map_coordinates(arguments)
        arguments = self._maybe_enrich_with_icons(arguments)

        # Use icon path for Altair if conditions are met
        if (
            self.backend == "altair"
            and self.icon_mode != "off"
            and "icon_rasters" in arguments
            and any(r is not None for r in arguments.get("icon_rasters", []))
        ):
            n_agents = len(arguments["size"])
            if self.icon_mode == "force" or n_agents <= self.icon_auto_max_agents:
                self.agent_mesh = build_altair_agent_chart(
                    arguments=arguments,
                    space_drawer=self.space_drawer,
                    chart_width=kwargs.get("chart_width", 450),
                    chart_height=kwargs.get("chart_height", 350),
                    title=kwargs.get("title", ""),
                    xlabel=kwargs.get("xlabel", ""),
                    ylabel=kwargs.get("ylabel", ""),
                    enable_culling=self.icon_culling,
                )
                return self.agent_mesh

        # Fall back to standard backend rendering
        self.agent_mesh = self.backend_renderer.draw_agents(
            arguments, **self.agent_kwargs
        )
        return self.agent_mesh

    def _get_space_drawer(self):
        """Get appropriate space drawer based on space type."""
        if isinstance(self.space, HexGrid | _HexGrid):
            return HexSpaceDrawer(self.space)
        elif isinstance(self.space, OrthogonalGrid):
            return OrthogonalSpaceDrawer(self.space)
        elif isinstance(
            self.space,
            ContinuousSpace | mesa.experimental.continuous_space.ContinuousSpace,
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
        """Map agent coordinates to appropriate space coordinates."""
        mapped_arguments = arguments.copy()

        if isinstance(self.space, OrthogonalGrid | VoronoiGrid | ContinuousSpace):
            mapped_arguments["loc"] = arguments["loc"].astype(float)

        elif isinstance(self.space, HexGrid):
            loc = arguments["loc"].astype(float)
            if loc.size > 0:
                loc[:, 0] = loc[:, 0] * self.space_drawer.x_spacing + (
                    (loc[:, 1] - 1) % 2
                ) * (self.space_drawer.x_spacing / 2)
                loc[:, 1] = loc[:, 1] * self.space_drawer.y_spacing
            mapped_arguments["loc"] = loc

        elif isinstance(self.space, Network):
            loc = arguments["loc"].astype(float)
            pos = np.asarray(list(self.space_drawer.pos.values()))
            x = loc[:, 0] if loc[:, 0] is not None else loc[:, 1]
            x = x.astype(int)
            with contextlib.suppress(IndexError):
                mapped_arguments["loc"] = pos[x]

        return mapped_arguments

    def _maybe_enrich_with_icons(self, arguments):
        """Enrich arguments with cached icon rasters/URLs if portrayal requests icons."""
        portrayals = arguments.get("portrayals")

        if portrayals is None or not isinstance(portrayals, (list, np.ndarray)):
            return arguments

        icon_rasters = []
        icon_names = []
        icon_sizes = []

        for p in portrayals:
            icon_name = p.get("icon")
            size = int(p.get("icon_size", p.get("size", self.space_drawer.s_default)))
            raster = (
                self._icon_cache.get_or_create(icon_name, size) if icon_name else None
            )

            icon_rasters.append(raster)
            icon_names.append(icon_name)
            icon_sizes.append(size)

        arguments["icon_names"] = icon_names
        arguments["icon_sizes"] = icon_sizes
        arguments["icon_rasters"] = icon_rasters

        # Group icons by name and size for batch rendering
        groups = {}
        for idx, (iname, isize) in enumerate(zip(icon_names, icon_sizes)):
            if iname:
                groups.setdefault((iname, isize), []).append(idx)
        arguments["icon_groups"] = groups

        return arguments

    def draw_structure(self, **kwargs):
        """Draw the space structure.

        Args:
            **kwargs: Additional keyword arguments for the drawing function.
                Checkout respective `SpaceDrawer` class on details how to pass **kwargs.

        Returns:
            The visual representation of the space structure.
        """
        self.space_kwargs = kwargs
        self.space_mesh = self.backend_renderer.draw_structure(**self.space_kwargs)
        return self.space_mesh

    def draw_propertylayer(self, propertylayer_portrayal: Callable | dict):
        """Draw property layers on the space.

        Args:
            propertylayer_portrayal: Function that returns PropertyLayerStyle
                or dict with portrayal parameters.

        Returns:
            The visual representation of the property layers.

        Raises:
            Exception: If no property layers are found on the space.
        """

        def _dict_to_callable(portrayal_dict):
            """Convert legacy dict portrayal to callable."""

            def style_callable(layer_object):
                layer_name = layer_object.name
                params = portrayal_dict.get(layer_name)

                warnings.warn(
                    "The propertylayer_portrayal dict is deprecated. "
                    "Please use a callable that returns a PropertyLayerStyle instance instead. "
                    "For more information, refer to the migration guide: "
                    "https://mesa.readthedocs.io/latest/migration_guide.html#defining-portrayal-components",
                    DeprecationWarning,
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
            property_layers = self.space.properties  # old style spaces
        except AttributeError:
            property_layers = self.space._mesa_property_layers  # new style spaces

        # Convert portrayal to callable if needed
        self.propertylayer_portrayal = (
            _dict_to_callable(propertylayer_portrayal)
            if isinstance(propertylayer_portrayal, dict)
            else propertylayer_portrayal
        )

        if sum(1 for layer in property_layers if layer != "empty") < 1:
            raise Exception("No property layers were found on the space.")

        self.propertylayer_mesh = self.backend_renderer.draw_propertylayer(
            self.space, property_layers, self.propertylayer_portrayal
        )
        return self.propertylayer_mesh

    def render(
        self,
        agent_portrayal: Callable | None = None,
        propertylayer_portrayal: Callable | dict | None = None,
        post_process: Callable | None = None,
        **kwargs,
    ):
        """Render the complete space with structure, agents, and property layers.

        Args:
            agent_portrayal: Function that returns AgentPortrayalStyle.
            propertylayer_portrayal: Function that returns PropertyLayerStyle or dict.
            post_process: Function to apply post-processing to the canvas.
            **kwargs: Additional keyword arguments:
                * ``space_kwargs`` (dict): Arguments for ``draw_structure()``.
                * ``agent_kwargs`` (dict): Arguments for ``draw_agents()``.
        """
        space_kwargs = kwargs.pop("space_kwargs", {})
        agent_kwargs = kwargs.pop("agent_kwargs", {})

        if self.space_mesh is None:
            self.draw_structure(**space_kwargs)
        if self.agent_mesh is None and agent_portrayal is not None:
            self.draw_agents(agent_portrayal, **agent_kwargs)
        if self.propertylayer_mesh is None and propertylayer_portrayal is not None:
            self.draw_propertylayer(propertylayer_portrayal)

        self.post_process_func = post_process
        return self

    @property
    def canvas(self):
        """Get the current canvas object."""
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
                structure = self.draw_structure(**self.space_kwargs)
            if self.agent_mesh:
                agents = self.draw_agents(self.agent_portrayal, **self.agent_kwargs)
            if self.propertylayer_mesh:
                prop_base, prop_cbar = self.draw_propertylayer(
                    self.propertylayer_portrayal
                )

            spatial_charts = [
                chart for chart in [structure, prop_base, agents] if chart
            ]

            main_spatial = None
            if spatial_charts:
                main_spatial = (
                    spatial_charts[0]
                    if len(spatial_charts) == 1
                    else alt.layer(*spatial_charts)
                )

            # Combine with color bar if present
            if main_spatial and prop_cbar:
                final_chart = alt.vconcat(main_spatial, prop_cbar).configure_view(
                    stroke=None
                )
            elif main_spatial:
                final_chart = main_spatial
            elif prop_cbar:
                final_chart = prop_cbar.configure_view(grid=False)
            else:
                # Return empty chart if nothing to display
                final_chart = (
                    alt.Chart(pd.DataFrame())
                    .mark_point()
                    .properties(width=450, height=350)
                )

            return final_chart.configure_view(stroke="black", strokeWidth=1.5)

    @property
    def post_process(self):
        """Get the current post-processing function."""
        return self.post_process_func

    @post_process.setter
    def post_process(self, func: Callable | None):
        """Set the post-processing function."""
        self.post_process_func = func
