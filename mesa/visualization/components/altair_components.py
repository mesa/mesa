"""Altair based solara components for plotting model measures."""

from collections.abc import Callable

import altair as alt
import solara

from mesa.visualization.utils import update_counter


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

    return solara.FigureAltair(chart)
