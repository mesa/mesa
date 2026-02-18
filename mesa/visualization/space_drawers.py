"""Mesa visualization space drawers.

This module provides the core logic for drawing spaces in Mesa, supporting
orthogonal grids, hexagonal grids, networks, continuous spaces, and Voronoi grids.
It includes implementations for both Matplotlib and Altair backends.
"""

from itertools import pairwise

import altair as alt
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

import mesa
from mesa.discrete_space import (
    OrthogonalMooreGrid,
    OrthogonalVonNeumannGrid,
    VoronoiGrid,
)
from mesa.experimental.continuous_space import ContinuousSpace

OrthogonalGrid = OrthogonalMooreGrid | OrthogonalVonNeumannGrid
HexGrid = mesa.discrete_space.HexGrid
Network = mesa.discrete_space.Network


class BaseSpaceDrawer:
    """Base class for all space drawers."""

    def __init__(self, space):
        """Initialize the base space drawer.

        Args:
            space: Grid/Space type to draw.
        """
        self.space = space
        self.viz_xmin = None
        self.viz_xmax = None
        self.viz_ymin = None
        self.viz_ymax = None

    def get_viz_limits(self):
        """Get visualization limits for the space.

        Returns:
            A tuple of (xmin, xmax, ymin, ymax) for visualization limits.
        """
        return (
            self.viz_xmin,
            self.viz_xmax,
            self.viz_ymin,
            self.viz_ymax,
        )


class OrthogonalSpaceDrawer(BaseSpaceDrawer):
    """Drawer for orthogonal grid spaces (SingleGrid, MultiGrid, Moore, VonNeumann)."""

    def __init__(self, space: OrthogonalGrid):
        """Initialize the orthogonal space drawer.

        Args:
            space: The orthogonal grid space to draw
        """
        super().__init__(space)
        self.s_default = (180 / max(self.space.width, self.space.height)) ** 2

        # Parameters for visualization limits
        self.viz_xmin = -0.5
        self.viz_xmax = self.space.width - 0.5
        self.viz_ymin = -0.5
        self.viz_ymax = self.space.height - 0.5

    def draw_matplotlib(self, ax=None, **draw_space_kwargs):
        """Draw the orthogonal grid using matplotlib.

        Args:
            ax: Matplotlib axes object to draw on
            **draw_space_kwargs: Additional keyword arguments for styling.

        Examples:
                figsize=(10, 10), color="blue", linewidth=2.

        Returns:
            The modified axes object
        """
        fig_kwargs = {
            "figsize": draw_space_kwargs.pop("figsize", (8, 8)),
            "dpi": draw_space_kwargs.pop("dpi", 100),
        }

        if ax is None:
            _, ax = plt.subplots(**fig_kwargs)

        # gridline styling kwargs
        line_kwargs = {
            "color": "gray",
            "linestyle": ":",
            "linewidth": 1,
            "alpha": 1,
        }
        line_kwargs.update(draw_space_kwargs)

        ax.set_xlim(self.viz_xmin, self.viz_xmax)
        ax.set_ylim(self.viz_ymin, self.viz_ymax)

        # Draw grid lines
        for x in np.arange(-0.5, self.space.width, 1):
            ax.axvline(x, **line_kwargs)
        for y in np.arange(-0.5, self.space.height, 1):
            ax.axhline(y, **line_kwargs)

        return ax

    def draw_altair(self, chart_width=450, chart_height=350, **draw_chart_kwargs):
        """Draw the orthogonal grid using Altair.

        Args:
            chart_width: Width for the shown chart
            chart_height: Height for the shown chart
            **draw_chart_kwargs: Additional keyword arguments for styling the chart.

        Examples:
                width=500, height=500, title="Grid".

        Returns:
            Altair chart object
        """
        # for axis and grid styling
        axis_kwargs = {
            "xlabel": draw_chart_kwargs.pop("xlabel", "X"),
            "ylabel": draw_chart_kwargs.pop("ylabel", "Y"),
            "grid_color": draw_chart_kwargs.pop("grid_color", "lightgray"),
            "grid_dash": draw_chart_kwargs.pop("grid_dash", [2, 2]),
            "grid_width": draw_chart_kwargs.pop("grid_width", 1),
            "grid_opacity": draw_chart_kwargs.pop("grid_opacity", 1),
        }

        # for chart properties
        chart_props = {
            "width": chart_width,
            "height": chart_height,
        }
        chart_props.update(draw_chart_kwargs)

        chart = (
            alt.Chart(pd.DataFrame([{}]))
            .mark_point(opacity=0)
            .encode(
                x=alt.X(
                    "X:Q",
                    title=axis_kwargs["xlabel"],
                    scale=alt.Scale(domain=[self.viz_xmin, self.viz_xmax], nice=False),
                    axis=alt.Axis(
                        grid=True,
                        gridColor=axis_kwargs["grid_color"],
                        gridDash=axis_kwargs["grid_dash"],
                        gridWidth=axis_kwargs["grid_width"],
                        gridOpacity=axis_kwargs["grid_opacity"],
                    ),
                ),
                y=alt.Y(
                    "Y:Q",
                    title=axis_kwargs["ylabel"],
                    scale=alt.Scale(domain=[self.viz_ymin, self.viz_ymax], nice=False),
                    axis=alt.Axis(
                        grid=True,
                        gridColor=axis_kwargs["grid_color"],
                        gridDash=axis_kwargs["grid_dash"],
                        gridWidth=axis_kwargs["grid_width"],
                        gridOpacity=axis_kwargs["grid_opacity"],
                    ),
                ),
            )
            .properties(**chart_props)
        )
        return chart


class HexSpaceDrawer(BaseSpaceDrawer):
    """Drawer for hexagonal grid spaces."""

    def __init__(self, space: HexGrid):
        """Initialize the hexagonal space drawer.

        Args:
            space: The hexagonal grid space to draw
        """
        super().__init__(space)
        self.s_default = (180 / max(self.space.width, self.space.height)) ** 2
        size = 1.0

        self.x_spacing = np.sqrt(3) * size
        self.y_spacing = 1.5 * size

        x_max = self.space.width * self.x_spacing + (self.space.height % 2) * (
            self.x_spacing / 2
        )
        y_max = self.space.height * self.y_spacing

        x_padding = size * np.sqrt(3) / 2
        y_padding = size

        self.hexagons = self._get_hexmesh(size)

        # Parameters for visualization limits
        self.viz_xmin = -1.8 * x_padding
        self.viz_xmax = x_max
        self.viz_ymin = -1.8 * y_padding
        self.viz_ymax = y_max

    def _get_hexmesh(self, size: float = 1.0) -> list[list[tuple[float, float]]]:
        """Generate hexagon vertices for the mesh. Yields list of list of vertex coordinates for each hexagon."""

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

        hexagons = []
        for cell in self.space.all_cells:
            cx, cy = cell.position
            hexagons.append(_get_hex_vertices(cx, cy, size))

        return hexagons

    def _get_unique_edges(self):
        """Helper method to extract unique edges from all hexagons."""
        edges = set()
        # Generate edges for each hexagon
        for vertices in self.hexagons:
            # Edge logic, connecting each vertex to the next
            for v1, v2 in pairwise([*vertices, vertices[0]]):
                # Sort vertices to ensure consistent edge representation and avoid duplicates.
                edge = tuple(sorted([tuple(np.round(v1, 6)), tuple(np.round(v2, 6))]))
                edges.add(edge)
        return edges

    def draw_matplotlib(self, ax=None, **draw_space_kwargs):
        """Draw the hexagonal grid using matplotlib.

        Args:
            ax: Matplotlib axes object to draw on
            **draw_space_kwargs: Additional keyword arguments for styling.

        Examples:
                figsize=(8, 8), color="red", alpha=0.5.

        Returns:
            The modified axes object
        """
        fig_kwargs = {
            "figsize": draw_space_kwargs.pop("figsize", (8, 8)),
            "dpi": draw_space_kwargs.pop("dpi", 100),
        }

        if ax is None:
            _, ax = plt.subplots(**fig_kwargs)

        line_kwargs = {
            "color": "black",
            "linestyle": ":",
            "linewidth": 1,
            "alpha": 0.8,
        }
        line_kwargs.update(draw_space_kwargs)

        ax.set_xlim(self.viz_xmin, self.viz_xmax)
        ax.set_ylim(self.viz_ymin, self.viz_ymax)
        ax.set_aspect("equal", adjustable="box")

        edges = self._get_unique_edges()
        ax.add_collection(LineCollection(list(edges), **line_kwargs))
        return ax

    def draw_altair(self, chart_width=450, chart_height=350, **draw_chart_kwargs):
        """Draw the hexagonal grid using Altair.

        Args:
            chart_width: Width for the shown chart
            chart_height: Height for the shown chart
            **draw_chart_kwargs: Additional keyword arguments for styling the chart.

        Examples:
                * Line properties like color, strokeDash, strokeWidth, opacity.
                * Other kwargs (e.g., width, title) apply to the chart.

        Returns:
            Altair chart object representing the hexagonal grid.
        """
        mark_kwargs = {
            "color": draw_chart_kwargs.pop("color", "black"),
            "strokeDash": draw_chart_kwargs.pop("strokeDash", [2, 2]),
            "strokeWidth": draw_chart_kwargs.pop("strokeWidth", 1),
            "opacity": draw_chart_kwargs.pop("opacity", 0.8),
        }

        chart_props = {
            "width": chart_width,
            "height": chart_height,
        }
        chart_props.update(draw_chart_kwargs)

        edge_data = []
        edges = self._get_unique_edges()

        for i, edge_tuple in enumerate(edges):
            p1, p2 = edge_tuple
            edge_data.append({"edge_id": i, "point_order": 0, "x": p1[0], "y": p1[1]})
            edge_data.append({"edge_id": i, "point_order": 1, "x": p2[0], "y": p2[1]})

        source = pd.DataFrame(edge_data)

        chart = (
            alt.Chart(source)
            .mark_line(**mark_kwargs)
            .encode(
                x=alt.X(
                    "x:Q",
                    scale=alt.Scale(domain=[self.viz_xmin, self.viz_xmax], zero=False),
                    axis=None,
                ),
                y=alt.Y(
                    "y:Q",
                    scale=alt.Scale(domain=[self.viz_ymin, self.viz_ymax], zero=False),
                    axis=None,
                ),
                detail="edge_id:N",
                order="point_order:Q",
            )
            .properties(**chart_props)
        )
        return chart


class NetworkSpaceDrawer(BaseSpaceDrawer):
    """Drawer for network-based spaces."""

    def __init__(
        self,
        space: Network,
        layout_alg=nx.spring_layout,
        layout_kwargs=None,
    ):
        """Initialize the network space drawer.

        Args:
            space: The network space to draw
            layout_alg: NetworkX layout algorithm to use
            layout_kwargs: Keyword arguments for the layout algorithm
        """
        super().__init__(space)
        self.layout_alg = layout_alg
        self.layout_kwargs = layout_kwargs if layout_kwargs is not None else {"seed": 0}

        # gather locations for nodes in network
        self.graph = self.space.G

        self.pos = {}

        for node_id, cell in self.space._cells.items():
            if getattr(cell, "_position", None) is not None:
                self.pos[node_id] = cell.position

        x, y = list(zip(*self.pos.values())) if self.pos else ([0], [0])
        xmin, xmax = min(x), max(x)
        ymin, ymax = min(y), max(y)

        width = xmax - xmin
        height = ymax - ymin
        self.s_default = (
            (180 / max(width, height)) ** 2 if width > 0 or height > 0 else 1
        )

        # Parameters for visualization limits
        self.viz_xmin = xmin - width / 20
        self.viz_xmax = xmax + width / 20
        self.viz_ymin = ymin - height / 20
        self.viz_ymax = ymax + height / 20

    def draw_matplotlib(self, ax=None, **draw_space_kwargs):
        """Draw the network using matplotlib.

        Args:
            ax: Matplotlib axes object to draw on.
            **draw_space_kwargs: Dictionaries of keyword arguments for styling.
                Can also handle zorder for both nodes and edges if passed.
                * ``node_kwargs``: A dict passed to nx.draw_networkx_nodes.
                * ``edge_kwargs``: A dict passed to nx.draw_networkx_edges.

        Returns:
            The modified axes object.
        """
        if ax is None:
            _, ax = plt.subplots()

        ax.set_axis_off()
        ax.set_xlim(self.viz_xmin, self.viz_xmax)
        ax.set_ylim(self.viz_ymin, self.viz_ymax)

        node_kwargs = {"alpha": 0.5}
        edge_kwargs = {"alpha": 0.5, "style": "--"}

        node_kwargs.update(draw_space_kwargs.get("node_kwargs", {}))
        edge_kwargs.update(draw_space_kwargs.get("edge_kwargs", {}))

        node_zorder = node_kwargs.pop("zorder", 1)
        edge_zorder = edge_kwargs.pop("zorder", 0)

        nodes = nx.draw_networkx_nodes(self.graph, self.pos, ax=ax, **node_kwargs)
        edges = nx.draw_networkx_edges(self.graph, self.pos, ax=ax, **edge_kwargs)

        if nodes:
            nodes.set_zorder(node_zorder)
        # In some matplotlib versions, edges can be a list of collections
        if isinstance(edges, list):
            for edge_collection in edges:
                edge_collection.set_zorder(edge_zorder)
        elif edges:
            edges.set_zorder(edge_zorder)

        return ax

    def draw_altair(self, chart_width=450, chart_height=350, **draw_chart_kwargs):
        """Draw the network using Altair.

        Args:
            chart_width: Width for the shown chart
            chart_height: Height for the shown chart
            **draw_chart_kwargs: Dictionaries for styling the chart.
                * ``node_kwargs``: A dict of properties for the node's mark_point.
                * ``edge_kwargs``: A dict of properties for the edge's mark_rule.
                * Other kwargs (e.g., title, width) are passed to chart.properties().

        Returns:
            Altair chart object representing the network.
        """
        nodes_df = pd.DataFrame(self.pos).T.reset_index()
        nodes_df.columns = ["node", "x", "y"]

        edges_df = pd.DataFrame(self.graph.edges(), columns=["source", "target"])
        edge_positions = edges_df.merge(
            nodes_df, how="left", left_on="source", right_on="node"
        ).merge(
            nodes_df,
            how="left",
            left_on="target",
            right_on="node",
            suffixes=("_source", "_target"),
        )

        node_mark_kwargs = {"filled": True, "opacity": 0.5, "size": 500}
        edge_mark_kwargs = {"opacity": 0.5, "strokeDash": [5, 3]}

        node_mark_kwargs.update(draw_chart_kwargs.pop("node_kwargs", {}))
        edge_mark_kwargs.update(draw_chart_kwargs.pop("edge_kwargs", {}))

        chart_props = {
            "width": chart_width,
            "height": chart_height,
        }
        chart_props.update(draw_chart_kwargs)

        edge_plot = (
            alt.Chart(edge_positions)
            .mark_rule(**edge_mark_kwargs)
            .encode(
                x=alt.X(
                    "x_source",
                    scale=alt.Scale(domain=[self.viz_xmin, self.viz_xmax]),
                    axis=None,
                ),
                y=alt.Y(
                    "y_source",
                    scale=alt.Scale(domain=[self.viz_ymin, self.viz_ymax]),
                    axis=None,
                ),
                x2="x_target",
                y2="y_target",
            )
        )

        node_plot = (
            alt.Chart(nodes_df)
            .mark_point(**node_mark_kwargs)
            .encode(x="x", y="y", tooltip=["node"])
        )

        chart = edge_plot + node_plot

        if chart_props:
            chart = chart.properties(**chart_props)

        return chart


class ContinuousSpaceDrawer(BaseSpaceDrawer):
    """Drawer for continuous spaces."""

    def __init__(self, space: ContinuousSpace):
        """Initialize the continuous space drawer.

        Args:
            space: The continuous space to draw
        """
        super().__init__(space)
        width = self.space.x_max - self.space.x_min
        height = self.space.y_max - self.space.y_min
        self.s_default = (
            (180 / max(width, height)) ** 2 if width > 0 or height > 0 else 1
        )

        x_padding = width / 20
        y_padding = height / 20

        self.viz_xmin = self.space.x_min - x_padding
        self.viz_xmax = self.space.x_max + x_padding
        self.viz_ymin = self.space.y_min - y_padding
        self.viz_ymax = self.space.y_max + y_padding

    def draw_matplotlib(self, ax=None, **draw_space_kwargs):
        """Draw the continuous space using matplotlib.

        Args:
            ax: Matplotlib axes object to draw on
            **draw_space_kwargs: Keyword arguments for styling the axis frame.

        Examples:
                linewidth=3, color="green"

        Returns:
            The modified axes object
        """
        if ax is None:
            _, ax = plt.subplots()

        border_style = "solid" if not self.space.torus else (0, (5, 10))
        spine_kwargs = {"linewidth": 1.5, "color": "black", "linestyle": border_style}
        spine_kwargs.update(draw_space_kwargs)

        for spine in ax.spines.values():
            spine.set(**spine_kwargs)

        ax.set_xlim(self.viz_xmin, self.viz_xmax)
        ax.set_ylim(self.viz_ymin, self.viz_ymax)

        return ax

    def draw_altair(self, chart_width=450, chart_height=350, **draw_chart_kwargs):
        """Draw the continuous space using Altair.

        Args:
            chart_width: Width for the shown chart
            chart_height: Height for the shown chart
            **draw_chart_kwargs: Keyword arguments for styling the chart's view properties.
                            See Altair's documentation for `configure_view`.

        Returns:
            An Altair Chart object representing the space.
        """
        chart_props = {"width": chart_width, "height": chart_height}
        chart_props.update(draw_chart_kwargs)

        chart = (
            alt.Chart(pd.DataFrame([{}]))
            .mark_rect(color="transparent")
            .encode(
                x=alt.X(scale=alt.Scale(domain=[self.viz_xmin, self.viz_xmax])),
                y=alt.Y(scale=alt.Scale(domain=[self.viz_ymin, self.viz_ymax])),
            )
            .properties(**chart_props)
        )

        return chart


class VoronoiSpaceDrawer(BaseSpaceDrawer):
    """Drawer for Voronoi diagram spaces."""

    def __init__(self, space: VoronoiGrid):
        """Initialize the Voronoi space drawer.

        Args:
            space: The Voronoi grid space to draw
        """
        super().__init__(space)
        # Use the Cell.position property for calculations
        positions = np.array([cell.position for cell in self.space.all_cells])
        if len(positions) > 0:
            x_min, y_min = positions.min(axis=0)
            x_max, y_max = positions.max(axis=0)
        else:
            x_max, x_min, y_max, y_min = 1, 0, 1, 0

        width = x_max - x_min
        height = y_max - y_min
        self.s_default = (
            (180 / max(width, height)) ** 2 if width > 0 or height > 0 else 1
        )

        # Parameters for visualization limits
        self.viz_xmin = x_min - width / 20
        self.viz_xmax = x_max + width / 20
        self.viz_ymin = y_min - height / 20
        self.viz_ymax = y_max + height / 20

    def _clip_line(self, p1, p2, box):
        """Clips a line segment using the Cohen-Sutherland algorithm.

        Returns the clipped line segment (p1, p2) or None if it's outside.
        """
        x1, y1 = p1
        x2, y2 = p2
        min_x, min_y, max_x, max_y = box

        # Define region codes
        INSIDE, LEFT, RIGHT, BOTTOM, TOP = 0, 1, 2, 4, 8  # noqa: N806

        def compute_outcode(x, y):
            code = INSIDE
            if x < min_x:
                code |= LEFT
            elif x > max_x:
                code |= RIGHT
            if y < min_y:
                code |= BOTTOM
            elif y > max_y:
                code |= TOP
            return code

        outcode1 = compute_outcode(x1, y1)
        outcode2 = compute_outcode(x2, y2)

        while True:
            if not (outcode1 | outcode2):  # Both points inside
                return (x1, y1), (x2, y2)
            elif outcode1 & outcode2:  # Both points share an outside region
                return None
            else:
                outcode_out = outcode1 if outcode1 else outcode2
                x, y = 0.0, 0.0

                # Check for horizontal line
                if y1 != y2:
                    if outcode_out & TOP:
                        x = x1 + (x2 - x1) * (max_y - y1) / (y2 - y1)
                        y = max_y
                    elif outcode_out & BOTTOM:
                        x = x1 + (x2 - x1) * (min_y - y1) / (y2 - y1)
                        y = min_y

                # Check for vertical line
                if x1 != x2:
                    if outcode_out & RIGHT:
                        y = y1 + (y2 - y1) * (max_x - x1) / (x2 - x1)
                        x = max_x
                    elif outcode_out & LEFT:
                        y = y1 + (y2 - y1) * (min_x - x1) / (x2 - x1)
                        x = min_x

                if outcode_out == outcode1:
                    x1, y1 = x, y
                    outcode1 = compute_outcode(x1, y1)
                else:
                    x2, y2 = x, y
                    outcode2 = compute_outcode(x2, y2)

    def _get_clipped_segments(self):
        """Helper method to perform the segment extraction, de-duplication and clipping logic."""
        clip_box = (
            self.viz_xmin,
            self.viz_ymin,
            self.viz_xmax,
            self.viz_ymax,
        )

        unique_segments = set()
        for cell in self.space.all_cells.cells:
            vertices = [tuple(v) for v in cell.properties["polygon"]]
            for p1, p2 in pairwise([*vertices, vertices[0]]):
                # Sort to avoid duplicate segments going in opposite directions
                unique_segments.add(tuple(sorted((p1, p2))))

        # Clip each unique segment
        final_segments = []
        for p1, p2 in unique_segments:
            clipped_segment = self._clip_line(p1, p2, clip_box)
            if clipped_segment:
                final_segments.append(clipped_segment)

        return final_segments, clip_box

    def draw_matplotlib(self, ax=None, **draw_space_kwargs):
        """Draw the Voronoi diagram using matplotlib.

        Args:
            ax: Matplotlib axes object to draw on
            **draw_space_kwargs: Keyword arguments passed to matplotlib's LineCollection.

        Examples:
                lw=2, alpha=0.5, colors='red'

        Returns:
            The modified axes object
        """
        if ax is None:
            _, ax = plt.subplots()

        final_segments, clip_box = self._get_clipped_segments()

        ax.set_xlim(clip_box[0], clip_box[2])
        ax.set_ylim(clip_box[1], clip_box[3])

        if final_segments:
            # Define default styles for the plot
            style_args = {"colors": "k", "linestyle": "dotted", "lw": 1}
            style_args.update(draw_space_kwargs)

            # Create the LineCollection with the final styles
            lc = LineCollection(final_segments, **style_args)
            ax.add_collection(lc)

        return ax

    def draw_altair(self, chart_width=450, chart_height=350, **draw_chart_kwargs):
        """Draw the Voronoi diagram using Altair.

        Args:
            chart_width: Width for the shown chart
            chart_height: Height for the shown chart
            **draw_chart_kwargs: Additional keyword arguments for styling the chart.

        Examples:
                * Line properties like color, strokeDash, strokeWidth, opacity.
                * Other kwargs (e.g., width, title) apply to the chart.

        Returns:
            An Altair Chart object representing the Voronoi diagram.
        """
        final_segments, clip_box = self._get_clipped_segments()

        # Prepare data
        final_data = []
        for i, (p1, p2) in enumerate(final_segments):
            final_data.append({"x": p1[0], "y": p1[1], "line_id": i})
            final_data.append({"x": p2[0], "y": p2[1], "line_id": i})

        df = pd.DataFrame(final_data)

        # Define default properties for the mark
        mark_kwargs = {
            "color": draw_chart_kwargs.pop("color", "black"),
            "strokeDash": draw_chart_kwargs.pop("strokeDash", [2, 2]),
            "strokeWidth": draw_chart_kwargs.pop("strokeWidth", 1),
            "opacity": draw_chart_kwargs.pop("opacity", 0.8),
        }

        chart_props = {"width": chart_width, "height": chart_height}
        chart_props.update(draw_chart_kwargs)

        chart = (
            alt.Chart(df)
            .mark_line(**mark_kwargs)
            .encode(
                x=alt.X(
                    "x:Q", scale=alt.Scale(domain=[clip_box[0], clip_box[2]]), axis=None
                ),
                y=alt.Y(
                    "y:Q", scale=alt.Scale(domain=[clip_box[1], clip_box[3]]), axis=None
                ),
                detail="line_id:N",
            )
            .properties(**chart_props)
        )
        return chart
