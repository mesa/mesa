"""Altair based solara components for visualization mesa spaces."""

import warnings
from collections.abc import Callable

import altair as alt
import numpy as np
import pandas as pd
import solara
from matplotlib.colors import to_rgb

import mesa
from mesa.discrete_space import DiscreteSpace, Grid
from mesa.space import ContinuousSpace, PropertyLayer, _Grid
from mesa.visualization.utils import FigureAltair, update_counter


def make_space_altair(*args, **kwargs):  # noqa: D103
    warnings.warn(
        "make_space_altair has been renamed to make_altair_space",
        FutureWarning,
        stacklevel=2,
    )
    return make_altair_space(*args, **kwargs)


def make_altair_space(
    agent_portrayal,
    propertylayer_portrayal=None,
    post_process=None,
    **space_drawing_kwargs,
):
    """Create an Altair-based space visualization component.

    Args:
        agent_portrayal: Function to portray agents.
        propertylayer_portrayal: Dictionary of PropertyLayer portrayal specifications
        post_process :A user specified callable that will be called with the Chart instance from Altair. Allows for fine tuning plots (e.g., control ticks)
        space_drawing_kwargs : not yet implemented

    ``agent_portrayal`` is called with an agent and should return a dict. Valid fields in this dict are "color",
    "size", "marker", and "zorder". Other field are ignored and will result in a user warning.


    Returns:
        function: A function that creates a SpaceMatplotlib component
    """
    if agent_portrayal is None:

        def agent_portrayal(a):
            return {"id": a.unique_id}

    def MakeSpaceAltair(model):
        return SpaceAltair(
            model,
            agent_portrayal,
            propertylayer_portrayal=propertylayer_portrayal,
            post_process=post_process,
        )

    return MakeSpaceAltair


@solara.component
def SpaceAltair(
    model,
    agent_portrayal,
    propertylayer_portrayal=None,
    dependencies: list[any] | None = None,
    post_process=None,
):
    """Create an Altair-based space visualization component.

    Returns:
        a solara FigureAltair instance
    """
    update_counter.get()
    space = getattr(model, "grid", None)
    if space is None:
        # Sometimes the space is defined as model.space instead of model.grid
        space = model.space

    chart = _draw_grid(space, agent_portrayal, propertylayer_portrayal)
    # Apply post-processing if provided
    if post_process is not None:
        chart = post_process(chart)

    return FigureAltair(chart)


def _portrayal_to_dict(portrayal_result, agent):
    """Convert AgentPortrayalStyle to dict or return dict as-is."""
    from mesa.visualization.components import AgentPortrayalStyle  # noqa

    if isinstance(portrayal_result, AgentPortrayalStyle):
        # Convert AgentPortrayalStyle to dict for Altair
        # Altair only uses 'color' and 'size' from the portrayal
        agent_data = {"id": getattr(agent, "unique_id", id(agent))}
        if portrayal_result.color is not None:
            agent_data["color"] = portrayal_result.color
        if portrayal_result.size is not None:
            agent_data["size"] = portrayal_result.size
        return agent_data
    else:
        return portrayal_result


def _get_agent_data_old__discrete_space(space, agent_portrayal):
    """Format agent portrayal data for old-style discrete spaces.

    Args:
        space: the mesa.space._Grid instance
        agent_portrayal: the agent portrayal callable

    Returns:
        list of dicts

    """
    all_agent_data = []
    for content, (x, y) in space.coord_iter():
        if not content:
            continue
        if not hasattr(content, "__iter__"):
            # Is a single grid
            content = [content]  # noqa: PLW2901
        for agent in content:
            # use all data from agent portrayal, and add x,y coordinates
            agent_data = _portrayal_to_dict(agent_portrayal(agent), agent)
            agent_data["x"] = x
            agent_data["y"] = y
            all_agent_data.append(agent_data)
    return all_agent_data


def _get_agent_data_new_discrete_space(space: DiscreteSpace, agent_portrayal):
    """Format agent portrayal data for new-style discrete spaces.

    Args:
        space: the mesa.discrete_space.Grid instance
        agent_portrayal: the agent portrayal callable

    Returns:
        list of dicts

    """
    all_agent_data = []

    for cell in space.all_cells:
        for agent in cell.agents:
            agent_data = _portrayal_to_dict(agent_portrayal(agent), agent)
            agent_data["x"] = cell.coordinate[0]
            agent_data["y"] = cell.coordinate[1]
            all_agent_data.append(agent_data)
    return all_agent_data


def _get_agent_data_continuous_space(space: ContinuousSpace, agent_portrayal):
    """Format agent portrayal data for continuous space.

    Args:
        space: the ContinuousSpace instance
        agent_portrayal: the agent portrayal callable

    Returns:
        list of dicts
    """
    all_agent_data = []
    for agent in space._agent_to_index:
        agent_data = _portrayal_to_dict(agent_portrayal(agent), agent)
        agent_data["x"] = agent.pos[0]
        agent_data["y"] = agent.pos[1]
        all_agent_data.append(agent_data)
    return all_agent_data


def _draw_grid(space, agent_portrayal, propertylayer_portrayal):
    match space:
        case Grid():
            all_agent_data = _get_agent_data_new_discrete_space(space, agent_portrayal)
        case _Grid():
            all_agent_data = _get_agent_data_old__discrete_space(space, agent_portrayal)
        case ContinuousSpace():
            all_agent_data = _get_agent_data_continuous_space(space, agent_portrayal)
        case _:
            raise NotImplementedError(
                f"visualizing {type(space)} is currently not supported through altair"
            )

    invalid_tooltips = ["color", "size", "x", "y"]

    x_y_type = "ordinal" if not isinstance(space, ContinuousSpace) else "nominal"

    encoding_dict = {
        # no x-axis label
        "x": alt.X("x", axis=None, type=x_y_type),
        # no y-axis label
        "y": alt.Y("y", axis=None, type=x_y_type),
        "tooltip": [
            alt.Tooltip(key, type=alt.utils.infer_vegalite_type_for_pandas([value]))
            for key, value in all_agent_data[0].items()
            if key not in invalid_tooltips
        ],
    }
    has_color = "color" in all_agent_data[0]
    if has_color:
        unique_colors = list({agent["color"] for agent in all_agent_data})
        encoding_dict["color"] = alt.Color(
            "color:N",
            scale=alt.Scale(domain=unique_colors, range=unique_colors),
        )
    has_size = "size" in all_agent_data[0]
    if has_size:
        encoding_dict["size"] = alt.Size("size", type="quantitative")

    agent_chart = (
        alt.Chart(
            alt.Data(values=all_agent_data), encoding=alt.Encoding(**encoding_dict)
        )
        .mark_point(filled=True)
        .properties(width=300, height=300)
    )
    base_chart = None
    cbar_chart = None

    # This is the default value for the marker size, which auto-scales according to the grid area.
    if not has_size:
        length = min(space.width, space.height)
        agent_chart = agent_chart.mark_point(size=30000 / length**2, filled=True)

    if propertylayer_portrayal is not None:
        chart_width = agent_chart.properties().width
        chart_height = agent_chart.properties().height
        base_chart, cbar_chart = chart_property_layers(
            space=space,
            propertylayer_portrayal=propertylayer_portrayal,
            chart_width=chart_width,
            chart_height=chart_height,
        )

        base_chart = alt.layer(base_chart, agent_chart)
    else:
        base_chart = agent_chart
    if cbar_chart is not None:
        base_chart = alt.vconcat(base_chart, cbar_chart).configure_view(stroke=None)
    return base_chart


def chart_property_layers(space, propertylayer_portrayal, chart_width, chart_height):
    """Creates Property Layers in the Altair Components.

    Args:
        space: the ContinuousSpace instance
        propertylayer_portrayal: Dictionary of PropertyLayer portrayal specifications
        chart_width: width of the agent chart to maintain consistency with the property charts
        chart_height: height of the agent chart to maintain consistency with the property charts
    Returns:
        Altair Chart
    """
    try:
        # old style spaces
        property_layers = space.properties
    except AttributeError:
        # new style spaces
        property_layers = space._mesa_property_layers
    base = None
    bar_chart = None
    for layer_name, portrayal in propertylayer_portrayal.items():
        layer = property_layers.get(layer_name, None)
        if not isinstance(
            layer,
            PropertyLayer | mesa.discrete_space.property_layer.PropertyLayer,
        ):
            continue

        data = layer.data.astype(float) if layer.data.dtype == bool else layer.data

        if (space.width, space.height) != data.shape:
            warnings.warn(
                f"Layer {layer_name} dimensions ({data.shape}) do not match space dimensions ({space.width}, {space.height}).",
                UserWarning,
                stacklevel=2,
            )
        alpha = portrayal.get("alpha", 1)
        vmin = portrayal.get("vmin", np.min(data))
        vmax = portrayal.get("vmax", np.max(data))
        colorbar = portrayal.get("colorbar", True)

        # Prepare data for Altair (convert 2D array to a long-form DataFrame)
        df = pd.DataFrame(
            {
                "x": np.repeat(np.arange(data.shape[0]), data.shape[1]),
                "y": np.tile(np.arange(data.shape[1]), data.shape[0]),
                "value": data.flatten(),
            }
        )

        if "color" in portrayal:
            # Create a function to map values to RGBA colors with proper opacity scaling
            def apply_rgba(val, vmin=vmin, vmax=vmax, alpha=alpha, portrayal=portrayal):
                """Maps data values to RGBA colors with opacity based on value magnitude.

                Args:
                    val: The data value to convert
                    vmin: The smallest value for which the color is displayed in the colorbar
                    vmax: The largest value for which the color is displayed in the colorbar
                    alpha: The opacity of the color
                    portrayal: The specifics of the current property layer in the iterative loop

                Returns:
                    String representation of RGBA color
                """
                # Normalize value to range [0,1] and clamp
                normalized = max(0, min((val - vmin) / (vmax - vmin), 1))

                # Scale opacity by alpha parameter
                opacity = normalized * alpha

                # Convert color to RGB components
                rgb_color = to_rgb(portrayal["color"])
                r = int(rgb_color[0] * 255)
                g = int(rgb_color[1] * 255)
                b = int(rgb_color[2] * 255)

                return f"rgba({r}, {g}, {b}, {opacity:.2f})"

            # Apply color mapping to each value in the dataset
            df["color"] = df["value"].apply(apply_rgba)

            # Create chart for the property layer
            chart = (
                alt.Chart(df)
                .mark_rect()
                .encode(
                    x=alt.X("x:O", axis=None),
                    y=alt.Y("y:O", axis=None),
                    fill=alt.Fill("color:N", scale=None),
                )
                .properties(width=chart_width, height=chart_height, title=layer_name)
            )
            base = alt.layer(chart, base) if base is not None else chart

            # Add colorbar if specified in portrayal
            if colorbar:
                # Extract RGB components from base color
                rgb_color = to_rgb(portrayal["color"])
                r_int = int(rgb_color[0] * 255)
                g_int = int(rgb_color[1] * 255)
                b_int = int(rgb_color[2] * 255)

                # Define gradient endpoints
                min_color = f"rgba({r_int},{g_int},{b_int},0)"
                max_color = f"rgba({r_int},{g_int},{b_int},{alpha:.2f})"

                # Define colorbar dimensions
                colorbar_height = 20
                colorbar_width = chart_width

                # Create dataframe for gradient visualization
                df_gradient = pd.DataFrame({"x": [0, 1], "y": [0, 1]})

                # Create evenly distributed tick values
                axis_values = np.linspace(vmin, vmax, 11)
                tick_positions = np.linspace(0, colorbar_width, 11)

                # Prepare data for axis and labels
                axis_data = pd.DataFrame({"value": axis_values, "x": tick_positions})

                # Create colorbar with linear gradient
                colorbar_chart = (
                    alt.Chart(df_gradient)
                    .mark_rect(
                        x=0,
                        y=0,
                        width=colorbar_width,
                        height=colorbar_height,
                        color=alt.Gradient(
                            gradient="linear",
                            stops=[
                                alt.GradientStop(color=min_color, offset=0),
                                alt.GradientStop(color=max_color, offset=1),
                            ],
                            x1=0,
                            x2=1,  # Horizontal gradient
                            y1=0,
                            y2=0,  # Keep y constant
                        ),
                    )
                    .encode(
                        x=alt.value(chart_width / 2),  # Center colorbar
                        y=alt.value(0),
                    )
                    .properties(width=colorbar_width, height=colorbar_height)
                )

                # Add tick marks to colorbar
                axis_chart = (
                    alt.Chart(axis_data)
                    .mark_tick(thickness=2, size=8)
                    .encode(x=alt.X("x:Q", axis=None), y=alt.value(colorbar_height - 2))
                )

                # Add value labels below tick marks
                text_labels = (
                    alt.Chart(axis_data)
                    .mark_text(baseline="top", fontSize=10, dy=0)
                    .encode(
                        x=alt.X("x:Q"),
                        text=alt.Text("value:Q", format=".1f"),
                        y=alt.value(colorbar_height + 10),
                    )
                )

                # Add title to colorbar
                title = (
                    alt.Chart(pd.DataFrame([{"text": layer_name}]))
                    .mark_text(
                        fontSize=12,
                        fontWeight="bold",
                        baseline="bottom",
                        align="center",
                    )
                    .encode(
                        text="text:N",
                        x=alt.value(colorbar_width / 2),
                        y=alt.value(colorbar_height + 40),
                    )
                )

                # Combine all colorbar components
                combined_colorbar = alt.layer(
                    colorbar_chart, axis_chart, text_labels, title
                ).properties(width=colorbar_width, height=colorbar_height + 50)

                bar_chart = (
                    alt.vconcat(bar_chart, combined_colorbar)
                    .resolve_scale(color="independent")
                    .configure_view(stroke=None)
                    if bar_chart is not None
                    else combined_colorbar
                )

        elif "colormap" in portrayal:
            cmap = portrayal.get("colormap", "viridis")
            cmap_scale = alt.Scale(scheme=cmap, domain=[vmin, vmax])

            chart = (
                alt.Chart(df)
                .mark_rect(opacity=alpha)
                .encode(
                    x=alt.X("x:O", axis=None),
                    y=alt.Y("y:O", axis=None),
                    color=alt.Color(
                        "value:Q",
                        scale=cmap_scale,
                        title=layer_name,
                        legend=alt.Legend(title=layer_name) if colorbar else None,
                    ),
                )
                .properties(width=chart_width, height=chart_height)
            )
            base = alt.layer(chart, base) if base is not None else chart

        else:
            raise ValueError(
                f"PropertyLayer {layer_name} portrayal must include 'color' or 'colormap'."
            )
    return base, bar_chart


def make_altair_plot_component(
    measure: str | dict[str, str] | list[str] | tuple[str],
    post_process: Callable | None = None,
    page: int = 0,
    grid=False,
):
    """Create a plotting function for a specified measure.

    Args:
        measure (str | dict[str, str] | list[str] | tuple[str]): Measure(s) to plot.
        post_process: a user-specified callable to do post-processing called with the Axes instance.
        page: Page number where the plot should be displayed.
        grid: Bool to draw grid or not.

    Returns:
        (function, page): A tuple of a function that creates a PlotAltair component and a page number.
    """

    def MakePlotAltair(model):
        return PlotAltair(model, measure, post_process=post_process, grid=grid)

    return (MakePlotAltair, page)


@solara.component
def PlotAltair(model, measure, post_process: Callable | None = None, grid=False):
    """Create an Altair-based plot for a measure or measures.

    Args:
        model (mesa.Model): The model instance.
        measure (str | dict[str, str] | list[str] | tuple[str]): Measure(s) to plot.
            If a dict is given, keys are measure names and values are colors.
        post_process: A user-specified callable for post-processing, called
            with the Altair Chart instance.
        grid: Bool to draw grid or not.

    Returns:
        solara.FigureAltair: A component for rendering the plot.
    """
    update_counter.get()
    df = model.datacollector.get_model_vars_dataframe().reset_index()
    df = df.rename(columns={"index": "Step"})

    y_title = "Value"
    if isinstance(measure, str):
        measures_to_plot = [measure]
        y_title = measure
    elif isinstance(measure, list | tuple):
        measures_to_plot = list(measure)
    elif isinstance(measure, dict):
        measures_to_plot = list(measure.keys())

    df_long = df.melt(
        id_vars=["Step"],
        value_vars=measures_to_plot,
        var_name="Measure",
        value_name="Value",
    )

    chart = (
        alt.Chart(df_long)
        .mark_line()
        .encode(
            x=alt.X("Step:Q", axis=alt.Axis(tickMinStep=1, title="Step", grid=grid)),
            y=alt.Y("Value:Q", axis=alt.Axis(title=y_title, grid=grid)),
            tooltip=["Step", "Measure", "Value"],
        )
        .properties(width=450, height=350)
        .interactive()
    )

    if len(measures_to_plot) > 0:
        color_args = {}
        if isinstance(measure, dict):
            color_args["scale"] = alt.Scale(
                domain=list(measure.keys()), range=list(measure.values())
            )
        chart = chart.encode(color=alt.Color("Measure:N", **color_args))

    if post_process is not None:
        chart = post_process(chart)

    return FigureAltair(chart)
