"""Altair-specific icon rendering layer for Mesa visualizations.

Builds a layered chart with mark_point (for non-icon agents) and mark_image
(for icon agents), with optional culling for performance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import altair as alt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from mesa.visualization.space_drawers import SpaceDrawer


def build_altair_agent_chart(
    arguments: dict[str, Any],
    space_drawer: SpaceDrawer,
    chart_width: int = 450,
    chart_height: int = 350,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    enable_culling: bool = True,
) -> alt.Chart:
    """Build layered Altair chart for agents with icon support.

    Args:
        arguments: Dictionary with agent data (loc, size, color, icon_rasters, etc.)
        space_drawer: SpaceDrawer instance for getting visualization limits
        chart_width: Chart width in pixels
        chart_height: Chart height in pixels
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        enable_culling: If True, skip off-screen agents for performance
    Returns:
        alt.Chart: Layered chart with icons and/or markers
    """
    if arguments["loc"].size == 0:
        return None

    # Extract data
    icon_rasters = arguments.get("icon_rasters", [])
    icon_names = arguments.get("icon_names", [])

    # Get space limits
    xmin, xmax, ymin, ymax = space_drawer.get_viz_limits()

    # Prepare DataFrame
    icon_sizes = arguments.get("icon_sizes", arguments["size"])
    df_data = {
        "x": arguments["loc"][:, 0],
        "y": arguments["loc"][:, 1],
        "size": arguments["size"],
        "icon_size": icon_sizes
        if isinstance(icon_sizes, (list, np.ndarray))
        else [icon_sizes] * len(arguments["size"]),
        "shape": arguments.get("shape", ["circle"] * len(arguments["size"])),
        "opacity": arguments.get("opacity", [1.0] * len(arguments["size"])),
        "color": arguments.get("color", ["#1f77b4"] * len(arguments["size"])),
    }

    # Convert icon_rasters list to pandas Series, handling None values
    if icon_rasters:
        # Ensure icon_rasters is the same length as other arrays
        if len(icon_rasters) != len(df_data["x"]):
            # Pad or truncate to match
            icon_rasters = list(icon_rasters[: len(df_data["x"])])
            while len(icon_rasters) < len(df_data["x"]):
                icon_rasters.append(None)

        df_data["icon_url"] = icon_rasters
        df_data["icon_name"] = (
            icon_names[: len(df_data["x"])]
            if icon_names
            else [None] * len(df_data["x"])
        )

    df = pd.DataFrame(df_data)

    # Optional culling: remove off-screen agents
    if enable_culling:
        # Add small margin for partially visible icons
        margin = 50  # pixels
        x_margin = (xmax - xmin) * margin / chart_width
        y_margin = (ymax - ymin) * margin / chart_height

        df = df[
            (df["x"] >= xmin - x_margin)
            & (df["x"] <= xmax + x_margin)
            & (df["y"] >= ymin - y_margin)
            & (df["y"] <= ymax + y_margin)
        ].copy()

    if df.empty:
        return None

    # Split into icon and non-icon agents
    has_icons = "icon_url" in df.columns
    if has_icons:
        df_icons = df[df["icon_url"].notna()].copy()
        df_points = df[df["icon_url"].isna()].copy()
    else:
        df_icons = pd.DataFrame()
        df_points = df.copy()

    layers = []
    tooltip_list = ["x", "y"]

    # Icon layer
    if not df_icons.empty:
        # Use consistent icon display size
        icon_display_size = 24  # pixels

        icon_chart = (
            alt.Chart(df_icons)
            .mark_image(
                width=icon_display_size,
                height=icon_display_size,
                align="center",
                baseline="middle",
            )
            .encode(
                x=alt.X(
                    "x:Q",
                    title=xlabel,
                    scale=alt.Scale(domain=[xmin, xmax]),
                    axis=None,
                ),
                y=alt.Y(
                    "y:Q",
                    title=ylabel,
                    scale=alt.Scale(domain=[ymin, ymax]),
                    axis=None,
                ),
                url=alt.Url("icon_url:N"),
                opacity=alt.Opacity(
                    "opacity:Q",
                    scale=alt.Scale(domain=[0, 1]),
                ),
                tooltip=tooltip_list,
            )
        )
        layers.append(icon_chart)

    # Point layer (fallback for non-icon agents)
    if not df_points.empty:
        point_chart = (
            alt.Chart(df_points)
            .mark_point(filled=True)
            .encode(
                x=alt.X(
                    "x:Q",
                    title=xlabel,
                    scale=alt.Scale(domain=[xmin, xmax]),
                    axis=None,
                ),
                y=alt.Y(
                    "y:Q",
                    title=ylabel,
                    scale=alt.Scale(domain=[ymin, ymax]),
                    axis=None,
                ),
                size=alt.Size("size:Q", legend=None, scale=alt.Scale(domain=[0, 50])),
                shape=alt.Shape("shape:N"),
                color=alt.Color("color:N", scale=None),
                opacity=alt.Opacity(
                    "opacity:Q",
                    scale=alt.Scale(domain=[0, 1]),
                ),
                tooltip=tooltip_list,
            )
        )
        layers.append(point_chart)

    # Combine layers
    if not layers:
        return None
    chart = layers[0] if len(layers) == 1 else alt.layer(*layers)

    chart = chart.properties(
        title=title,
        width=chart_width,
        height=chart_height,
    )
    return chart
