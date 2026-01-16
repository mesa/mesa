# noqa: D100
import os
import warnings
from dataclasses import fields

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.figure import Figure
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

import mesa
from mesa.discrete_space import (
    OrthogonalMooreGrid,
    OrthogonalVonNeumannGrid,
)
from mesa.space import (
    HexMultiGrid,
    HexSingleGrid,
    MultiGrid,
    SingleGrid,
)
from mesa.visualization.backends.abstract_renderer import AbstractRenderer

OrthogonalGrid = SingleGrid | MultiGrid | OrthogonalMooreGrid | OrthogonalVonNeumannGrid
HexGrid = HexSingleGrid | HexMultiGrid | mesa.discrete_space.HexGrid


CORRECTION_FACTOR_MARKER_ZOOM = 0.01


class MatplotlibBackend(AbstractRenderer):
    """Matplotlib-based renderer for Mesa spaces.

    Provides visualization capabilities using Matplotlib for rendering
    space structures, agents, and property layers.
    """

    def __init__(self, space_drawer):
        """Initialize the Matplotlib backend.

        Args:
            space_drawer: An instance of a SpaceDrawer class that handles
                the drawing of the space structure.
        """
        super().__init__(space_drawer)

        self._active_colorbars = []

    def initialize_canvas(self, ax=None):
        """Initialize the matplotlib canvas.

        Args:
            ax (matplotlib.axes.Axes, optional): Existing axes to use.
                If None, creates new figure and axes.
        """
        if ax is None:
            fig = Figure(constrained_layout=True)
            ax = fig.add_subplot()
            self.fig = fig
        self.ax = ax

    def draw_structure(self, **kwargs):
        """Draw the space structure using matplotlib.

        Args:
            **kwargs: Additional arguments passed to the space drawer.
            Checkout respective `SpaceDrawer` class on details how to pass **kwargs.

        Returns:
            The matplotlib axes with the drawn structure.
        """
        return self.space_drawer.draw_matplotlib(self.ax, **kwargs)

    def collect_agent_data(self, space, agent_portrayal, default_size=None):
        """Collect plotting data for all agents in the space.

        Args:
            space: The Mesa space containing agents.
            agent_portrayal (Callable): Function that returns AgentPortrayalStyle for each agent.
            default_size (float, optional): Default marker size if not specified in portrayal.

        Returns:
            dict: Dictionary containing agent plotting data arrays.
        """
        # Initialize data collection arrays
        arguments = {
            "loc": [],
            "s": [],
            "c": [],
            "marker": [],
            "zorder": [],
            "alpha": [],
            "edgecolors": [],
            "linewidths": [],
        }
        # Import here to prevent circular imports
        from mesa.visualization.components import AgentPortrayalStyle  # noqa: PLC0415

        # Get default values from AgentPortrayalStyle
        style_fields = {f.name: f.default for f in fields(AgentPortrayalStyle)}
        class_default_size = style_fields.get("size")

        for agent in space.agents:
            portray_input = agent_portrayal(agent)

            if isinstance(portray_input, dict):
                warnings.warn(
                    (
                        "Returning a dict from agent_portrayal is deprecated and will be removed in Mesa 4.0. "
                        "Please return an AgentPortrayalStyle instance instead. "
                        "For more information, refer to the migration guide: "
                        "https://mesa.readthedocs.io/latest/migration_guide.html#defining-portrayal-components"
                    ),
                    FutureWarning,
                    stacklevel=2,
                )
                # Handle legacy dict input
                dict_data = portray_input.copy()
                agent_x, agent_y = self._get_agent_pos(agent, space)

                # Extract values with defaults
                aps = AgentPortrayalStyle(
                    x=agent_x,
                    y=agent_y,
                    size=dict_data.pop("size", style_fields.get("size")),
                    color=dict_data.pop("color", style_fields.get("color")),
                    marker=dict_data.pop("marker", style_fields.get("marker")),
                    zorder=dict_data.pop("zorder", style_fields.get("zorder")),
                    alpha=dict_data.pop("alpha", style_fields.get("alpha")),
                    edgecolors=dict_data.pop("edgecolors", None),
                    linewidths=dict_data.pop(
                        "linewidths", style_fields.get("linewidths")
                    ),
                )

                # Warn about unused keys
                if dict_data:
                    ignored_keys = list(dict_data.keys())
                    warnings.warn(
                        f"The following keys were ignored: {', '.join(ignored_keys)}",
                        UserWarning,
                        stacklevel=2,
                    )
            else:
                aps = portray_input
                # Set defaults if not provided
                if aps.x is None and aps.y is None:
                    aps.x, aps.y = self._get_agent_pos(agent, space)

            # Collect agent data
            arguments["loc"].append((aps.x, aps.y))

            # Determine final size
            size_to_collect = aps.size or default_size or class_default_size
            arguments["s"].append(size_to_collect)
            arguments["c"].append(aps.color)
            arguments["marker"].append(aps.marker)
            arguments["zorder"].append(aps.zorder)
            arguments["alpha"].append(aps.alpha)
            if aps.edgecolors is not None:
                arguments["edgecolors"].append(aps.edgecolors)
            arguments["linewidths"].append(aps.linewidths)

        # Convert to numpy arrays
        data = {
            k: (np.asarray(v, dtype=object) if k == "marker" else np.asarray(v))
            for k, v in arguments.items()
        }

        # Handle marker array specially to preserve tuples
        arr = np.empty(len(arguments["marker"]), dtype=object)
        arr[:] = arguments["marker"]
        data["marker"] = arr

        return data

    def _get_zoom_factor(self, ax, img):
        """Calculate zoom factor only once and cache the result."""
        ax.get_figure().canvas.draw()

        bbox = ax.get_window_extent().transformed(
            ax.get_figure().dpi_scale_trans.inverted()
        )
        width, height = (
            bbox.width * ax.get_figure().dpi,
            bbox.height * ax.get_figure().dpi,
        )

        xr = ax.get_xlim()
        yr = ax.get_ylim()

        x_pixel_per_data = width / (xr[1] - xr[0])
        y_pixel_per_data = height / (yr[1] - yr[0])

        zoom_x = (x_pixel_per_data / img.width) * CORRECTION_FACTOR_MARKER_ZOOM
        zoom_y = (y_pixel_per_data / img.height) * CORRECTION_FACTOR_MARKER_ZOOM

        return min(zoom_x, zoom_y)

    def draw_agents(self, arguments, **kwargs):
        """Draw agents on the backend's axes - optimized version.

        Args:
            arguments: Dictionary containing agent data arrays.
            **kwargs: Additional keyword arguments for customization.
            Checkout respective `SpaceDrawer` class on details how to pass **kwargs.

        Returns:
            matplotlib.axes.Axes: The Matplotlib Axes with the agents drawn upon it.
        """
        if arguments["loc"].size == 0:
            return None

        loc = arguments.pop("loc")
        loc_x, loc_y = loc[:, 0], loc[:, 1]
        marker = arguments.pop("marker")
        zorder = arguments.pop("zorder")
        malpha = arguments["alpha"]
        msize = arguments["s"]

        # Validate edge arguments
        for entry in ["edgecolors", "linewidths"]:
            if len(arguments[entry]) == 0:
                arguments.pop(entry)
            elif entry in kwargs:
                raise ValueError(
                    f"{entry} is specified in agent portrayal and via plotting kwargs, "
                    "you can only use one or the other"
                )

        # Cache for loaded images and their zoom factors
        image_cache = {}

        # Separate image and non-image markers
        unique_markers = set(marker)
        image_markers = set()
        regular_markers = set()

        for mark in unique_markers:
            if isinstance(mark, str | os.PathLike) and os.path.isfile(mark):
                image_markers.add(mark)
            else:
                regular_markers.add(mark)

        self.ax.get_figure().canvas.draw()

        for mark in image_markers:
            if mark not in image_cache:
                image = Image.open(mark)
                base_zoom = self._get_zoom_factor(self.ax, image)
                image_cache[mark] = {
                    "image": image,
                    "base_zoom": base_zoom,
                    "offset_images": {},
                }

            cache_entry = image_cache[mark]
            mask_marker = marker == mark

            unique_sizes = np.unique(msize[mask_marker])

            for m_size in unique_sizes:
                if m_size not in cache_entry["offset_images"]:
                    im = OffsetImage(
                        cache_entry["image"],
                        zoom=cache_entry["base_zoom"] * m_size,
                    )
                    im.image.axes = self.ax
                    cache_entry["offset_images"][m_size] = im

                im = cache_entry["offset_images"][m_size]

                size_marker_mask = mask_marker & (msize == m_size)

                unique_zorders = np.unique(zorder[size_marker_mask])
                unique_alphas = np.unique(malpha[size_marker_mask])

                for z_order in unique_zorders:
                    for m_alpha in unique_alphas:
                        final_mask = (
                            (zorder == z_order) & (malpha == m_alpha) & size_marker_mask
                        )
                        positions = loc[final_mask]

                        for x, y in positions:
                            ab = AnnotationBbox(
                                im,
                                (x, y),
                                frameon=False,
                                pad=0.0,
                                zorder=z_order,
                                **kwargs,
                            )
                            self.ax.add_artist(ab)

        for mark in regular_markers:
            mask_marker = marker == mark

            unique_zorders = np.unique(zorder[mask_marker])

            for z_order in unique_zorders:
                zorder_mask = (zorder == z_order) & mask_marker

                scatter_args = {k: v[zorder_mask] for k, v in arguments.items()}

                self.ax.scatter(
                    loc_x[zorder_mask],
                    loc_y[zorder_mask],
                    marker=mark,
                    zorder=z_order,
                    **scatter_args,
                    **kwargs,
                )

        return self.ax

    def draw_propertylayer(self, space, property_layers, propertylayer_portrayal):
        """Draw property layers using matplotlib backend.

        Args:
            space: The Mesa space object.
            property_layers (dict): Dictionary of property layers to visualize.
            propertylayer_portrayal (Callable): Function that returns PropertyLayerStyle.

        Returns:
            tuple: (matplotlib.axes.Axes, colorbar) - The matplotlib axes and colorbar objects.
        """
        # Draw each layer
        for layer_name in property_layers:
            if layer_name == "empty":
                continue

            layer = property_layers.get(layer_name)
            portrayal = propertylayer_portrayal(layer)

            if portrayal is None:
                continue

            data = layer.data.astype(float) if layer.data.dtype == bool else layer.data

            # Check dimensions
            if (space.width, space.height) != data.shape:
                warnings.warn(
                    f"Layer {layer_name} dimensions ({data.shape}) "
                    f"don't match space dimensions ({space.width}, {space.height})",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            # Get portrayal parameters
            color = portrayal.color
            colormap = portrayal.colormap
            alpha = portrayal.alpha
            vmin = portrayal.vmin if portrayal.vmin is not None else np.min(data)
            vmax = portrayal.vmax if portrayal.vmax is not None else np.max(data)

            # Set up colormap
            if color:
                rgba_color = to_rgba(color)
                cmap = LinearSegmentedColormap.from_list(
                    layer_name, [(0, 0, 0, 0), (*rgba_color[:3], alpha)]
                )
            elif colormap:
                cmap = colormap
                if isinstance(cmap, list):
                    cmap = LinearSegmentedColormap.from_list(layer_name, cmap)
                elif isinstance(cmap, str):
                    cmap = plt.get_cmap(cmap)
            else:
                raise ValueError(
                    f"PropertyLayer {layer_name} must include 'color' or 'colormap'"
                )

            # Draw based on space type
            if isinstance(space, OrthogonalGrid):
                if color:
                    data = data.T
                    normalized_data = (data - vmin) / (vmax - vmin)
                    rgba_data = np.full((*data.shape, 4), rgba_color)
                    rgba_data[..., 3] *= normalized_data * alpha
                    rgba_data = np.clip(rgba_data, 0, 1)
                    self.ax.imshow(rgba_data, origin="lower")
                else:
                    self.ax.imshow(
                        data.T,
                        cmap=cmap,
                        alpha=alpha,
                        vmin=vmin,
                        vmax=vmax,
                        origin="lower",
                    )
            elif isinstance(space, HexGrid):
                hexagons = self.space_drawer.hexagons
                norm = Normalize(vmin=vmin, vmax=vmax)
                colors = data.T.ravel()

                if color:
                    normalized_colors = np.clip(norm(colors), 0, 1)
                    rgba_colors = np.full((len(colors), 4), rgba_color)
                    rgba_colors[:, 3] = normalized_colors * alpha
                else:
                    rgba_colors = cmap(norm(colors))
                    rgba_colors[..., 3] *= alpha

                collection = PolyCollection(hexagons, facecolors=rgba_colors, zorder=-1)
                self.ax.add_collection(collection)
            else:
                raise NotImplementedError(
                    f"PropertyLayer visualization not implemented for {type(space)}"
                )

            # Add colorbar if requested
            cbar = None
            if portrayal.colorbar:
                norm = Normalize(vmin=vmin, vmax=vmax)
                sm = ScalarMappable(norm=norm, cmap=cmap)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=self.ax, label=layer_name)
                self._active_colorbars.append(cbar)
        return self.ax, cbar
