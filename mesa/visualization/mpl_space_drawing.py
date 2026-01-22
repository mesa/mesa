"""Helper functions for drawing mesa spaces with matplotlib.

These functions are used by the provided matplotlib components, but can also be used to quickly visualize
a space with matplotlib for example when creating a mp4 of a movie run or when needing a figure
for a paper.

"""

import itertools
import os
import warnings
from collections.abc import Callable
from dataclasses import fields
from functools import lru_cache
from itertools import pairwise
from typing import Any

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection, PatchCollection, PolyCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Polygon
from PIL import Image

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
)

CORRECTION_FACTOR_MARKER_ZOOM = 0.6
DEFAULT_MARKER_SIZE = 50

OrthogonalGrid = SingleGrid | MultiGrid | OrthogonalMooreGrid | OrthogonalVonNeumannGrid
HexGrid = HexSingleGrid | HexMultiGrid | mesa.discrete_space.HexGrid
Network = NetworkGrid | mesa.discrete_space.Network


def collect_agent_data(
    space: OrthogonalGrid | HexGrid | Network | ContinuousSpace | VoronoiGrid,
    agent_portrayal: Callable,
    default_size: float | None = None,
) -> dict:
    """Collect the plotting data for all agents in the space.

    Args:
        space: The space containing the Agents.
        agent_portrayal: A callable that is called with the agent and returns a AgentPortrayalStyle
        default_size: default size

    agent_portrayal should return a AgentPortrayalStyle, limited to size (size of marker), color (color of marker), zorder (z-order),
    marker (marker style), alpha, linewidths, and edgecolors.
    """

    def get_agent_pos(agent, space):
        """Helper function to get the agent position depending on the grid type."""
        if isinstance(space, NetworkGrid):
            agent_x, agent_y = agent.pos, agent.pos
        elif isinstance(space, Network):
            agent_x, agent_y = agent.cell.coordinate, agent.cell.coordinate
        else:
            agent_x = (
                agent.pos[0] if agent.pos is not None else agent.cell.coordinate[0]
            )
            agent_y = (
                agent.pos[1] if agent.pos is not None else agent.cell.coordinate[1]
            )
        return agent_x, agent_y

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

    # Importing AgentPortrayalStyle inside the function to prevent circular imports
    from mesa.visualization.components import AgentPortrayalStyle  # noqa: PLC0415

    # Get AgentPortrayalStyle defaults
    style_fields = {f.name: f.default for f in fields(AgentPortrayalStyle)}
    class_default_size = style_fields.get("size")

    for agent in space.agents:
        portray_input = agent_portrayal(agent)
        aps: AgentPortrayalStyle

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
            dict_data = portray_input.copy()

            agent_x, agent_y = get_agent_pos(agent, space)

            # Extract values from the dict, using defaults if not provided
            size_val = dict_data.pop("size", style_fields.get("size"))
            color_val = dict_data.pop("color", style_fields.get("color"))
            marker_val = dict_data.pop("marker", style_fields.get("marker"))
            zorder_val = dict_data.pop("zorder", style_fields.get("zorder"))
            alpha_val = dict_data.pop("alpha", style_fields.get("alpha"))
            edgecolors_val = dict_data.pop("edgecolors", None)
            linewidths_val = dict_data.pop("linewidths", style_fields.get("linewidths"))

            aps = AgentPortrayalStyle(
                x=agent_x,
                y=agent_y,
                size=size_val,
                color=color_val,
                marker=marker_val,
                zorder=zorder_val,
                alpha=alpha_val,
                edgecolors=edgecolors_val,
                linewidths=linewidths_val,
            )

            # Report list of unused data
            if dict_data:
                ignored_keys = list(dict_data.keys())
                warnings.warn(
                    f"The following keys from the returned dict were ignored: {', '.join(ignored_keys)}",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            aps = portray_input
            # default to agent's color if not provided
            if aps.edgecolors is None and not isinstance(
                aps.color, int | float | np.number
            ):
                aps.edgecolors = aps.color
            # get position if not specified
            if aps.x is None and aps.y is None:
                aps.x, aps.y = get_agent_pos(agent, space)

        # Collect common data from the AgentPortrayalStyle instance
        arguments["loc"].append((aps.x, aps.y))

        # Determine final size for collection
        size_to_collect = aps.size
        if size_to_collect is None:
            size_to_collect = default_size
        if size_to_collect is None:
            size_to_collect = class_default_size

        arguments["s"].append(size_to_collect)
        arguments["c"].append(aps.color)
        arguments["marker"].append(aps.marker)
        arguments["zorder"].append(aps.zorder)
        arguments["alpha"].append(aps.alpha)
        if aps.edgecolors is not None:
            arguments["edgecolors"].append(aps.edgecolors)
        arguments["linewidths"].append(aps.linewidths)

    data = {
        k: (np.asarray(v, dtype=object) if k == "marker" else np.asarray(v))
        for k, v in arguments.items()
    }
    # ensures that the tuples in marker dont get converted by numpy to an array resulting in a 2D array
    arr = np.empty(len(arguments["marker"]), dtype=object)
    arr[:] = arguments["marker"]
    data["marker"] = arr
    return data


def draw_space(
    space,
    agent_portrayal: Callable,
    propertylayer_portrayal: Callable | None = None,
    ax: Axes | None = None,
    **space_drawing_kwargs,
):
    """Draw a Matplotlib-based visualization of the space.

    Args:
        space: the space of the mesa model
        agent_portrayal: A callable that returns a AgnetPortrayalStyle specifying how to show the agent
        propertylayer_portrayal: A callable that returns a PropertyLayerStyle specifying how to show the property layer
        ax: the axes upon which to draw the plot
        space_drawing_kwargs: any additional keyword arguments to be passed on to the underlying function for drawing the space.

    Returns:
        Returns the Axes object with the plot drawn onto it.

    ``agent_portrayal`` is called with an agent and should return a AgentPortrayalStyle. Valid fields in this object are "color",
    "size", "marker", "zorder", alpha, linewidths, and edgecolors. Other field are ignored and will result in a user warning.

    """
    if ax is None:
        _, ax = plt.subplots()

    # https://stackoverflow.com/questions/67524641/convert-multiple-isinstance-checks-to-structural-pattern-matching
    match space:
        # order matters here given the class structure of old-style grid spaces
        case HexSingleGrid() | HexMultiGrid() | mesa.discrete_space.HexGrid():
            draw_hex_grid(space, agent_portrayal, ax=ax, **space_drawing_kwargs)
        case (
            mesa.space.SingleGrid()
            | OrthogonalMooreGrid()
            | OrthogonalVonNeumannGrid()
            | mesa.space.MultiGrid()
        ):
            draw_orthogonal_grid(space, agent_portrayal, ax=ax, **space_drawing_kwargs)
        case mesa.space.NetworkGrid() | mesa.discrete_space.Network():
            draw_network(space, agent_portrayal, ax=ax, **space_drawing_kwargs)
        case (
            mesa.space.ContinuousSpace()
            | mesa.experimental.continuous_space.ContinuousSpace()
        ):
            draw_continuous_space(space, agent_portrayal, ax=ax)
        case VoronoiGrid():
            draw_voronoi_grid(space, agent_portrayal, ax=ax)
        case _:
            raise ValueError(f"Unknown space type: {type(space)}")

    if propertylayer_portrayal:
        draw_property_layers(space, propertylayer_portrayal, ax=ax)

    return ax


@lru_cache(maxsize=1024, typed=True)
def _get_hexmesh(
    width: int, height: int, size: float = 1.0
) -> list[tuple[float, float]]:
    """Generate hexagon vertices for the mesh. Yields list of vertex coordinates for each hexagon."""

    # Helper function for getting the vertices of a hexagon given the center and size
    def _get_hex_vertices(
        center_x: float, center_y: float, size: float = 1.0
    ) -> list[tuple[float, float]]:
        """Get vertices for a hexagon centered at (center_x, center_y)."""
        vertices = [
            (center_x, center_y + size),  # top
            (center_x + size * np.sqrt(3) / 2, center_y + size / 2),  # top right
            (center_x + size * np.sqrt(3) / 2, center_y - size / 2),  # bottom right
            (center_x, center_y - size),  # bottom
            (center_x - size * np.sqrt(3) / 2, center_y - size / 2),  # bottom left
            (center_x - size * np.sqrt(3) / 2, center_y + size / 2),  # top left
        ]
        return vertices

    x_spacing = np.sqrt(3) * size
    y_spacing = 1.5 * size
    hexagons = []

    for row, col in itertools.product(range(height), range(width)):
        # Calculate center position with offset for even rows
        x = col * x_spacing + (row % 2 == 0) * (x_spacing / 2)
        y = row * y_spacing
        hexagons.append(_get_hex_vertices(x, y, size))

    return hexagons


def draw_property_layers(
    space, propertylayer_portrayal: dict[str, dict[str, Any]] | Callable, ax: Axes
):
    """Draw PropertyLayers on the given axes.

    Args:
        space (mesa.space._Grid): The space containing the PropertyLayers.
        propertylayer_portrayal (Callable): A function that accepts a property layer object
            and returns either a `PropertyLayerStyle` object defining its visualization,
            or `None` to skip drawing this particular layer.
        ax (matplotlib.axes.Axes): The axes to draw on.

    """
    # Importing here to avoid circular import issues
    from mesa.visualization.components import PropertyLayerStyle  # noqa: PLC0415

    def _propertylayer_portryal_dict_to_callable(
        propertylayer_portrayal: dict[str, dict[str, Any]],
    ):
        """Helper function to convert a propertylayer_portrayal dict to a callable that return a PropertyLayerStyle."""

        def style_callable(layer_object: Any):
            layer_name = layer_object.name
            params = propertylayer_portrayal.get(layer_name)

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
                return None  # Layer not specified in the dict, so skip.

            return PropertyLayerStyle(
                color=params.get("color"),
                colormap=params.get("colormap"),
                alpha=params.get(
                    "alpha", PropertyLayerStyle.alpha
                ),  # Use defaults defined in the dataclass itself
                vmin=params.get("vmin"),
                vmax=params.get("vmax"),
                colorbar=params.get("colorbar", PropertyLayerStyle.colorbar),
            )

        return style_callable

    try:
        # old style spaces
        property_layers = space.properties
    except AttributeError:
        # new style spaces
        property_layers = space._mesa_property_layers

    callable_portrayal: Callable[[Any], PropertyLayerStyle | None]
    if isinstance(propertylayer_portrayal, dict):
        callable_portrayal = _propertylayer_portryal_dict_to_callable(
            propertylayer_portrayal
        )
    else:
        callable_portrayal = propertylayer_portrayal

    for layer_name in property_layers:
        if layer_name == "empty":
            # Skipping empty layer, automatically generated
            continue

        layer = property_layers.get(layer_name, None)
        portrayal = callable_portrayal(layer)

        if portrayal is None:
            # Not visualizing layers that do not have a defined visual encoding.
            continue

        data = layer.data.astype(float) if layer.data.dtype == bool else layer.data

        if (space.width, space.height) != data.shape:
            warnings.warn(
                f"Layer {layer_name} dimensions ({data.shape}) do not match space dimensions ({space.width}, {space.height}).",
                UserWarning,
                stacklevel=2,
            )

        color = portrayal.color
        colormap = portrayal.colormap
        alpha = portrayal.alpha
        vmin = portrayal.vmin if portrayal.vmin else np.min(data)
        vmax = portrayal.vmax if portrayal.vmax else np.max(data)

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
                f"PropertyLayer {layer_name} portrayal must include 'color' or 'colormap'."
            )

        if isinstance(space, OrthogonalGrid):
            if color:
                data = data.T
                normalized_data = (data - vmin) / (vmax - vmin)
                rgba_data = np.full((*data.shape, 4), rgba_color)
                rgba_data[..., 3] *= normalized_data * alpha
                rgba_data = np.clip(rgba_data, 0, 1)
                ax.imshow(rgba_data, origin="lower")
            else:
                ax.imshow(
                    data.T,
                    cmap=cmap,
                    alpha=alpha,
                    vmin=vmin,
                    vmax=vmax,
                    origin="lower",
                )
        elif isinstance(space, HexGrid):
            width, height = data.shape
            hexagons = _get_hexmesh(width, height)
            norm = Normalize(vmin=vmin, vmax=vmax)
            colors = data.ravel()

            if color:
                normalized_colors = np.clip(norm(colors), 0, 1)
                rgba_colors = np.full((len(colors), 4), rgba_color)
                rgba_colors[:, 3] = normalized_colors * alpha
            else:
                rgba_colors = cmap(norm(colors))
                rgba_colors[..., 3] *= alpha
            collection = PolyCollection(hexagons, facecolors=rgba_colors, zorder=-1)
            ax.add_collection(collection)
        else:
            raise NotImplementedError(
                f"PropertyLayer visualization not implemented for {type(space)}."
            )
        if portrayal.colorbar:
            norm = Normalize(vmin=vmin, vmax=vmax)
            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            plt.colorbar(sm, ax=ax, label=layer_name)


def draw_orthogonal_grid(
    space: OrthogonalGrid,
    agent_portrayal: Callable,
    ax: Axes | None = None,
    draw_grid: bool = True,
    **kwargs,
):
    """Visualize a orthogonal grid.

    Args:
        space: the space to visualize
        agent_portrayal: a callable that is called with the agent and returns a AgentPortrayalStyle
        ax: a Matplotlib Axes instance. If none is provided a new figure and ax will be created using plt.subplots
        draw_grid: whether to draw the grid
        kwargs: additional keyword arguments passed to ax.scatter

    Returns:
        Returns the Axes object with the plot drawn onto it.

    ``agent_portrayal`` is called with an agent and should return a AgentPortrayalStyle. Valid fields in this object are "color",
    "size", "marker", "zorder", alpha, linewidths, and edgecolors. Other field are ignored and will result in a user warning.

    """
    if ax is None:
        _, ax = plt.subplots()

    # gather agent data
    s_default = (180 / max(space.width, space.height)) ** 2
    arguments = collect_agent_data(space, agent_portrayal, default_size=s_default)

    # further styling
    ax.set_xlim(-0.5, space.width - 0.5)
    ax.set_ylim(-0.5, space.height - 0.5)

    # plot the agents
    _scatter(ax, arguments, **kwargs)

    if draw_grid:
        # Draw grid lines
        for x in np.arange(-0.5, space.width - 0.5, 1):
            ax.axvline(x, color="gray", linestyle=":")
        for y in np.arange(-0.5, space.height - 0.5, 1):
            ax.axhline(y, color="gray", linestyle=":")

    return ax


def draw_hex_grid(
    space: HexGrid,
    agent_portrayal: Callable,
    ax: Axes | None = None,
    draw_grid: bool = True,
    **kwargs,
):
    """Visualize a hex grid.

    Args:
        space: the space to visualize
        agent_portrayal: a callable that is called with the agent and returns a AgentPortrayalStyle
        ax: a Matplotlib Axes instance. If none is provided a new figure and ax will be created using plt.subplots
        draw_grid: whether to draw the grid
        kwargs: additional keyword arguments passed to ax.scatter
    Returns:
        Returns the Axes object with the plot drawn onto it.

    ``agent_portrayal`` is called with an agent and should return a AgentPortrayalStyle. Valid fields in this object are "color",
    "size", "marker", "zorder", alpha, linewidths, and edgecolors. Other field are ignored and will result in a user warning.
    """
    if ax is None:
        _, ax = plt.subplots()

    # gather data
    s_default = (180 / max(space.width, space.height)) ** 2
    arguments = collect_agent_data(space, agent_portrayal, default_size=s_default)

    # Parameters for hexagon grid
    size = 1.0
    x_spacing = np.sqrt(3) * size
    y_spacing = 1.5 * size

    # Calculate proper bounds that account for the full hexagon width and height
    x_max = space.width * x_spacing + (space.height % 2) * (x_spacing / 2)
    y_max = space.height * y_spacing

    # Add padding that accounts for the hexagon points
    x_padding = (
        size * np.sqrt(3) / 2
    )  # Distance from center to rightmost point of hexagon
    y_padding = size  # Distance from center to topmost point of hexagon

    # Plot limits to perfectly contain the hexagonal grid
    # Determined through physical testing.
    ax.set_xlim(-2 * x_padding, x_max + x_padding)
    ax.set_ylim(-2 * y_padding, y_max + y_padding)

    loc = arguments["loc"].astype(float)
    # Calculate hexagon centers for agents if agents are present and plot them.
    if loc.size > 0:
        loc[:, 0] = loc[:, 0] * x_spacing + ((loc[:, 1] - 1) % 2) * (x_spacing / 2)
        loc[:, 1] = loc[:, 1] * y_spacing
        arguments["loc"] = loc

        # plot the agents
        _scatter(ax, arguments, **kwargs)

    def setup_hexmesh(width, height):
        """Helper function for creating the hexmesh with unique edges."""
        edges = set()

        # Generate edges for each hexagon
        hexagons = _get_hexmesh(width, height)
        for vertices in hexagons:
            # Edge logic, connecting each vertex to the next
            for v1, v2 in pairwise([*vertices, vertices[0]]):
                # Sort vertices to ensure consistent edge representation and avoid duplicates.
                edge = tuple(sorted([tuple(np.round(v1, 6)), tuple(np.round(v2, 6))]))
                edges.add(edge)

        return LineCollection(edges, linestyle=":", color="black", linewidth=1, alpha=1)

    if draw_grid:
        ax.add_collection(setup_hexmesh(space.width, space.height))

    return ax


def draw_network(
    space: Network,
    agent_portrayal: Callable,
    ax: Axes | None = None,
    draw_grid: bool = True,
    layout_alg=nx.spring_layout,
    layout_kwargs=None,
    **kwargs,
):
    """Visualize a network space.

    Args:
        space: the space to visualize
        agent_portrayal: a callable that is called with the agent and returns a AgentPortrayalStyle
        ax: a Matplotlib Axes instance. If none is provided a new figure and ax will be created using plt.subplots
        draw_grid: whether to draw the grid
        layout_alg: a networkx layout algorithm or other callable with the same behavior
        layout_kwargs: a dictionary of keyword arguments for the layout algorithm
        kwargs: additional keyword arguments passed to ax.scatter

    Returns:
        Returns the Axes object with the plot drawn onto it.

    ``agent_portrayal`` is called with an agent and should return a AgentPortrayalStyle. Valid fields in this object are "color",
    "size", "marker", "zorder", alpha, linewidths, and edgecolors. Other field are ignored and will result in a user warning.

    """
    if ax is None:
        _, ax = plt.subplots()
    if layout_kwargs is None:
        layout_kwargs = {"seed": 0}

    # gather locations for nodes in network
    graph = space.G
    pos = layout_alg(graph, **layout_kwargs)
    x, y = list(zip(*pos.values()))
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)

    width = xmax - xmin
    height = ymax - ymin
    x_padding = width / 20
    y_padding = height / 20

    # gather agent data
    s_default = (180 / max(width, height)) ** 2
    arguments = collect_agent_data(space, agent_portrayal, default_size=s_default)

    # this assumes that nodes are identified by an integer
    # which is true for default nx graphs but might user changeable
    pos = np.asarray(list(pos.values()))
    loc = arguments["loc"]

    # For network only one of x and y contains the correct coordinates
    x = loc[:, 0]
    if x is None:
        x = loc[:, 1]

    arguments["loc"] = pos[x]

    # further styling
    ax.set_axis_off()
    ax.set_xlim(xmin=xmin - x_padding, xmax=xmax + x_padding)
    ax.set_ylim(ymin=ymin - y_padding, ymax=ymax + y_padding)

    # plot the agents
    _scatter(ax, arguments, **kwargs)

    if draw_grid:
        # fixme we need to draw the empty nodes as well
        edge_collection = nx.draw_networkx_edges(
            graph, pos, ax=ax, alpha=0.5, style="--"
        )
        edge_collection.set_zorder(0)

    return ax


def draw_continuous_space(
    space: ContinuousSpace, agent_portrayal: Callable, ax: Axes | None = None, **kwargs
):
    """Visualize a continuous space.

    Args:
        space: the space to visualize
        agent_portrayal: a callable that is called with the agent and returns a AgentPortrayalStyle
        ax: a Matplotlib Axes instance. If none is provided a new figure and ax will be created using plt.subplots
        kwargs: additional keyword arguments passed to ax.scatter

    Returns:
        Returns the Axes object with the plot drawn onto it.

    ``agent_portrayal`` is called with an agent and should return a AgentPortrayalStyle. Valid fields in this object are "color",
    "size", "marker", "zorder", alpha, linewidths, and edgecolors. Other field are ignored and will result in a user warning.

    """
    if ax is None:
        _, ax = plt.subplots()

    # space related setup
    width = space.x_max - space.x_min
    x_padding = width / 20
    height = space.y_max - space.y_min
    y_padding = height / 20

    # gather agent data
    s_default = (180 / max(width, height)) ** 2
    arguments = collect_agent_data(space, agent_portrayal, default_size=s_default)

    # further visual styling
    border_style = "solid" if not space.torus else (0, (5, 10))
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color("black")
        spine.set_linestyle(border_style)

    ax.set_xlim(space.x_min - x_padding, space.x_max + x_padding)
    ax.set_ylim(space.y_min - y_padding, space.y_max + y_padding)

    # plot the agents
    _scatter(ax, arguments, **kwargs)

    return ax


def draw_voronoi_grid(
    space: VoronoiGrid,
    agent_portrayal: Callable,
    ax: Axes | None = None,
    draw_grid: bool = True,
    **kwargs,
):
    """Visualize a voronoi grid.

    Args:
        space: the space to visualize
        agent_portrayal: a callable that is called with the agent and returns a AgentPortrayalStyle
        ax: a Matplotlib Axes instance. If none is provided a new figure and ax will be created using plt.subplots
        draw_grid: whether to draw the grid or not
        kwargs: additional keyword arguments passed to ax.scatter

    Returns:
        Returns the Axes object with the plot drawn onto it.

    ``agent_portrayal`` is called with an agent and should return a AgentPortrayalStyle. Valid fields in this object are "color",
    "size", "marker", "zorder", alpha, linewidths, and edgecolors. Other field are ignored and will result in a user warning.

    """
    if ax is None:
        _, ax = plt.subplots()

    x_list = [i[0] for i in space.centroids_coordinates]
    y_list = [i[1] for i in space.centroids_coordinates]
    x_max = max(x_list)
    x_min = min(x_list)
    y_max = max(y_list)
    y_min = min(y_list)

    width = x_max - x_min
    x_padding = width / 20
    height = y_max - y_min
    y_padding = height / 20

    s_default = (180 / max(width, height)) ** 2
    arguments = collect_agent_data(space, agent_portrayal, default_size=s_default)

    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    _scatter(ax, arguments, **kwargs)

    def setup_voroinoimesh(cells):
        patches = []
        for cell in cells:
            patch = Polygon(cell.properties["polygon"])
            patches.append(patch)
        mesh = PatchCollection(
            patches, edgecolor="k", facecolor=(1, 1, 1, 0), linestyle="dotted", lw=1
        )
        return mesh

    if draw_grid:
        ax.add_collection(setup_voroinoimesh(space.all_cells.cells))
    return ax


def _get_zoom_factor(ax, img):
    ax.get_figure().canvas.draw()
    bbox = ax.get_window_extent().transformed(
        ax.get_figure().dpi_scale_trans.inverted()
    )  # in inches
    width, height = (
        bbox.width * ax.get_figure().dpi,
        bbox.height * ax.get_figure().dpi,
    )  # in pixel

    xr = ax.get_xlim()
    yr = ax.get_ylim()

    x_pixel_per_data = width / (xr[1] - xr[0])
    y_pixel_per_data = height / (yr[1] - yr[0])

    zoom_x = (x_pixel_per_data / img.width) * CORRECTION_FACTOR_MARKER_ZOOM
    zoom_y = (y_pixel_per_data / img.height) * CORRECTION_FACTOR_MARKER_ZOOM

    return min(zoom_x, zoom_y)


def _scatter(ax: Axes, arguments, **kwargs):
    """Helper function for plotting the agents.

    Args:
        ax: a Matplotlib Axes instance
        arguments: the agents specific arguments for plotting
        kwargs: additional keyword arguments for ax.scatter

    """
    loc = arguments.pop("loc")

    loc_x = loc[:, 0]
    loc_y = loc[:, 1]
    marker = arguments.pop("marker")
    zorder = arguments.pop("zorder")
    malpha = arguments.pop("alpha")
    msize = arguments.pop("s")

    # we check if edgecolor, linewidth, and alpha are specified
    # at the agent level, if not, we remove them from the arguments dict
    # and fallback to the default value in ax.scatter / use what is passed via **kwargs
    for entry in ["edgecolors", "linewidths"]:
        if len(arguments[entry]) == 0:
            arguments.pop(entry)
        else:
            if entry in kwargs:
                raise ValueError(
                    f"{entry} is specified in agent portrayal and via plotting kwargs, you can only use one or the other"
                )

    ax.get_figure().canvas.draw()
    for mark in set(marker):
        if isinstance(mark, (str | os.PathLike)) and os.path.isfile(mark):
            # images
            for m_size in np.unique(msize):
                image = Image.open(mark)
                im = OffsetImage(
                    image,
                    zoom=_get_zoom_factor(ax, image) * m_size / DEFAULT_MARKER_SIZE,
                )
                im.image.axes = ax

                mask_marker = [m == mark for m in list(marker)] & (m_size == msize)
                for z_order in np.unique(zorder[mask_marker]):
                    for m_alpha in np.unique(malpha[mask_marker]):
                        mask = (z_order == zorder) & (m_alpha == malpha) & mask_marker
                        for x, y in zip(loc_x[mask], loc_y[mask]):
                            ab = AnnotationBbox(
                                im,
                                (x, y),
                                frameon=False,
                                pad=0.0,
                                zorder=z_order,
                                **kwargs,
                            )
                            ax.add_artist(ab)

        else:
            # ordinary markers
            mask_marker = [m == mark for m in list(marker)]
            for z_order in np.unique(zorder[mask_marker]):
                zorder_mask = z_order == zorder & mask_marker
                ax.scatter(
                    loc_x[zorder_mask],
                    loc_y[zorder_mask],
                    marker=mark,
                    zorder=z_order,
                    **{k: v[zorder_mask] for k, v in arguments.items()},
                    **kwargs,
                )
