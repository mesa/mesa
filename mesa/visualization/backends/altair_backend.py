# noqa: D100
import warnings
from collections.abc import Callable
from dataclasses import fields
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb

import mesa
from mesa.discrete_space import (
    OrthogonalMooreGrid,
    OrthogonalVonNeumannGrid,
)
from mesa.space import (
    HexMultiGrid,
    HexSingleGrid,
    MultiGrid,
    NetworkGrid,
    SingleGrid,
)
from mesa.visualization.backends.abstract_renderer import AbstractRenderer

OrthogonalGrid = SingleGrid | MultiGrid | OrthogonalMooreGrid | OrthogonalVonNeumannGrid
HexGrid = HexSingleGrid | HexMultiGrid | mesa.discrete_space.HexGrid
Network = NetworkGrid | mesa.discrete_space.Network


class AltairBackend(AbstractRenderer):
    """Altair-based renderer for Mesa spaces.

    This module provides an Altair-based renderer for visualizing Mesa model spaces,
    agents, and property layers with interactive charting capabilities.
    """

    def initialize_canvas(self) -> None:
        """Initialize the Altair canvas."""
        self._canvas = None

    def draw_structure(self, **kwargs) -> alt.Chart:
        """Draw the space structure using Altair.

        Args:
            **kwargs: Additional arguments passed to the space drawer.
            Checkout respective `SpaceDrawer` class on details how to pass **kwargs.

        Returns:
            alt.Chart: The Altair chart representing the space structure.
        """
        return self.space_drawer.draw_altair(**kwargs)

    def collect_agent_data(
        self, space, agent_portrayal: Callable, default_size: float | None = None
    ):
        """Collect plotting data for all agents in the space for Altair.

        Args:
            space: The Mesa space containing agents.
            agent_portrayal: Callable that returns AgentPortrayalStyle for each agent.
            default_size: Default marker size if not specified in portrayal.

        Returns:
            dict: Dictionary containing agent plotting data arrays.
        """
        # Initialize data collection arrays
        arguments = {
            "loc": [],
            "size": [],
            "color": [],
            "shape": [],
            "order": [],  # z-order
            "opacity": [],
            "stroke": [],  # Stroke color
            "strokeWidth": [],
            "filled": [],
        }

        # Import here to avoid circular import issues
        from mesa.visualization.components import AgentPortrayalStyle  # noqa: PLC0415

        style_fields = {f.name: f.default for f in fields(AgentPortrayalStyle)}
        class_default_size = style_fields.get("size")

        # Marker mapping from Matplotlib to Altair
        marker_to_shape_map = {
            "o": "circle",
            "s": "square",
            "D": "diamond",
            "^": "triangle-up",
            "v": "triangle-down",
            "<": "triangle-left",
            ">": "triangle-right",
            "+": "cross",
            "x": "cross",  # Both '+' and 'x' map to cross in Altair
            ".": "circle",  # Small point becomes circle
            "1": "triangle-down",
            "2": "triangle-up",
            "3": "triangle-left",
            "4": "triangle-right",
        }

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
                agent_x, agent_y = self._get_agent_pos(agent, space)

                aps = AgentPortrayalStyle(
                    x=agent_x,
                    y=agent_y,
                    size=dict_data.pop("size", style_fields.get("size")),
                    color=dict_data.pop("color", style_fields.get("color")),
                    marker=dict_data.pop("marker", style_fields.get("marker")),
                    zorder=dict_data.pop("zorder", style_fields.get("zorder")),
                    alpha=dict_data.pop("alpha", style_fields.get("alpha")),
                    edgecolors=dict_data.pop(
                        "edgecolors", style_fields.get("edgecolors")
                    ),
                    linewidths=dict_data.pop(
                        "linewidths", style_fields.get("linewidths")
                    ),
                )
                if dict_data:
                    ignored_keys = list(dict_data.keys())
                    warnings.warn(
                        f"The following keys were ignored from dict portrayal: {', '.join(ignored_keys)}",
                        UserWarning,
                        stacklevel=2,
                    )
            else:
                aps = portray_input
                if aps.x is None and aps.y is None:
                    aps.x, aps.y = self._get_agent_pos(agent, space)

            arguments["loc"].append((aps.x, aps.y))

            size_to_collect = aps.size if aps.size is not None else default_size
            if size_to_collect is None:
                size_to_collect = class_default_size
            arguments["size"].append(size_to_collect)

            arguments["color"].append(
                aps.color if aps.color is not None else style_fields.get("color")
            )

            # Map marker to Altair shape if defined, else use raw marker
            raw_marker = (
                aps.marker if aps.marker is not None else style_fields.get("marker")
            )
            shape_value = marker_to_shape_map.get(raw_marker, raw_marker)
            if shape_value is None:
                warnings.warn(
                    f"Marker '{raw_marker}' is not supported in Altair. "
                    "Using 'circle' as default.",
                    UserWarning,
                    stacklevel=2,
                )
                shape_value = "circle"
            arguments["shape"].append(shape_value)

            arguments["order"].append(
                aps.zorder if aps.zorder is not None else style_fields.get("zorder")
            )
            arguments["opacity"].append(
                aps.alpha if aps.alpha is not None else style_fields.get("alpha")
            )
            arguments["stroke"].append(aps.edgecolors)
            arguments["strokeWidth"].append(
                aps.linewidths
                if aps.linewidths is not None
                else style_fields.get("linewidths")
            )

            # FIXME: Make filled user-controllable
            filled_value = True
            arguments["filled"].append(filled_value)

        final_data = {}
        for k, v in arguments.items():
            if k == "shape":
                # Ensure shape is an object array
                arr = np.empty(len(v), dtype=object)
                arr[:] = v
                final_data[k] = arr
            elif k in ["x", "y", "size", "order", "opacity", "strokeWidth"]:
                final_data[k] = np.asarray(v, dtype=float)
            else:
                final_data[k] = np.asarray(v)

        return final_data

    def draw_agents(
        self, arguments, chart_width: int = 450, chart_height: int = 350, **kwargs
    ):
        """Draw agents using Altair backend.

        Args:
            arguments: Dictionary containing agent data arrays.
            chart_width: Width of the chart.
            chart_height: Height of the chart.
            **kwargs: Additional keyword arguments for customization.
            Checkout respective `SpaceDrawer` class on details how to pass **kwargs.

        Returns:
            alt.Chart: The Altair chart representing the agents, or None if no agents.
        """
        if arguments["loc"].size == 0:
            return None

        # To get a continuous scale for color the domain should be between [0, 1]
        # that's why changing the the domain of strokeWidth beforehand.
        stroke_width = [data / 10 for data in arguments["strokeWidth"]]

        # Agent data preparation
        df_data = {
            "x": arguments["loc"][:, 0],
            "y": arguments["loc"][:, 1],
            "size": arguments["size"],
            "shape": arguments["shape"],
            "opacity": arguments["opacity"],
            "strokeWidth": stroke_width,
            "original_color": arguments["color"],
            "is_filled": arguments["filled"],
            "original_stroke": arguments["stroke"],
        }
        df = pd.DataFrame(df_data)

        # To ensure distinct shapes according to agent portrayal
        unique_shape_names_in_data = df["shape"].unique().tolist()

        fill_colors = []
        stroke_colors = []
        for i in range(len(df)):
            filled = df["is_filled"][i]
            main_color = df["original_color"][i]
            stroke_spec = (
                df["original_stroke"][i]
                if isinstance(df["original_stroke"][i], str)
                else None
            )
            if filled:
                fill_colors.append(main_color)
                stroke_colors.append(stroke_spec)
            else:
                fill_colors.append(None)
                stroke_colors.append(main_color)
        df["viz_fill_color"] = fill_colors
        df["viz_stroke_color"] = stroke_colors

        # Extract additional parameters from kwargs
        # FIXME: Add more parameters to kwargs
        title = kwargs.pop("title", "")
        xlabel = kwargs.pop("xlabel", "")
        ylabel = kwargs.pop("ylabel", "")

        # Tooltip list for interactivity
        # FIXME: Add more fields to tooltip (preferably from agent_portrayal)
        tooltip_list = ["x", "y"]

        # Handle custom colormapping
        cmap = kwargs.pop("cmap", "viridis")
        vmin = kwargs.pop("vmin", None)
        vmax = kwargs.pop("vmax", None)

        color_is_numeric = pd.api.types.is_numeric_dtype(df["original_color"])
        if color_is_numeric:
            color_min = vmin if vmin is not None else df["original_color"].min()
            color_max = vmax if vmax is not None else df["original_color"].max()

            fill_encoding = alt.Fill(
                "original_color:Q",
                scale=alt.Scale(scheme=cmap, domain=[color_min, color_max]),
            )
        else:
            fill_encoding = alt.Fill(
                "viz_fill_color:N",
                scale=None,
                title="Color",
            )

        # Determine space dimensions
        xmin, xmax, ymin, ymax = self.space_drawer.get_viz_limits()

        chart = (
            alt.Chart(df)
            .mark_point()
            .encode(
                x=alt.X(
                    "x:Q",
                    title=xlabel,
                    scale=alt.Scale(type="linear", domain=[xmin, xmax]),
                    axis=None,
                ),
                y=alt.Y(
                    "y:Q",
                    title=ylabel,
                    scale=alt.Scale(type="linear", domain=[ymin, ymax]),
                    axis=None,
                ),
                size=alt.Size("size:Q", legend=None, scale=alt.Scale(domain=[0, 50])),
                shape=alt.Shape(
                    "shape:N",
                    scale=alt.Scale(
                        domain=unique_shape_names_in_data,
                        range=unique_shape_names_in_data,
                    ),
                    title="Shape",
                ),
                opacity=alt.Opacity(
                    "opacity:Q",
                    title="Opacity",
                    scale=alt.Scale(domain=[0, 1], range=[0, 1]),
                ),
                fill=fill_encoding,
                stroke=alt.Stroke("viz_stroke_color:N", scale=None),
                strokeWidth=alt.StrokeWidth(
                    "strokeWidth:Q", scale=alt.Scale(domain=[0, 1])
                ),
                tooltip=tooltip_list,
            )
            .properties(title=title, width=chart_width, height=chart_height)
        )

        return chart

    def draw_propertylayer(
        self,
        space,
        property_layers: dict[str, Any],
        propertylayer_portrayal: Callable,
        chart_width: int = 450,
        chart_height: int = 350,
    ):
        """Draw property layers using Altair backend.

        Args:
            space: The Mesa space object containing the property layers.
            property_layers: A dictionary of property layers to draw.
            propertylayer_portrayal: A function that returns PropertyLayerStyle
                that contains the visualization parameters.
            chart_width: The width of the chart.
            chart_height: The height of the chart.

        Returns:
            alt.Chart: A tuple containing the base chart and the color bar chart.
        """
        main_charts = []

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

            df = pd.DataFrame(
                {
                    "x": np.repeat(np.arange(data.shape[0]), data.shape[1]),
                    "y": np.tile(np.arange(data.shape[1] - 1, -1, -1), data.shape[0]),
                    "value": data.flatten(),
                }
            )

            if color:
                # For a single color gradient, we define the range from transparent to solid.
                rgb = to_rgb(color)
                r, g, b = (int(c * 255) for c in rgb)

                min_color = f"rgba({r},{g},{b},0)"
                max_color = f"rgba({r},{g},{b},{alpha})"
                opacity = 1
                color_scale = alt.Scale(
                    range=[min_color, max_color], domain=[vmin, vmax]
                )

            elif colormap:
                cmap = colormap
                color_scale = alt.Scale(scheme=cmap, domain=[vmin, vmax])
                opacity = alpha

            else:
                raise ValueError(
                    f"PropertyLayer {layer_name} portrayal must include 'color' or 'colormap'."
                )

            current_chart = (
                alt.Chart(df)
                .mark_rect(opacity=opacity)
                .encode(
                    x=alt.X("x:O", axis=None),
                    y=alt.Y("y:O", axis=None),
                    color=alt.Color(
                        "value:Q",
                        scale=color_scale,
                        title=layer_name,
                        legend=alt.Legend(title=layer_name, orient="bottom")
                        if portrayal.colorbar
                        else None,
                    ),
                )
                .properties(width=chart_width, height=chart_height)
            )

            if current_chart is not None:
                main_charts.append(current_chart)

        base = alt.layer(*main_charts).resolve_scale(color="independent")
        return base
