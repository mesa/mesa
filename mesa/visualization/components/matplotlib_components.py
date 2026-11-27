"""Matplotlib based solara components for plotting model measures."""

from __future__ import annotations

import warnings
from collections.abc import Callable

import matplotlib.pyplot as plt
import solara
from matplotlib.figure import Figure

from mesa.visualization.utils import update_counter


def make_plot_measure(*args, **kwargs):  # noqa: D103
    warnings.warn(
        "make_plot_measure has been renamed to make_plot_component",
        FutureWarning,
        stacklevel=2,
    )
    return make_mpl_plot_component(*args, **kwargs)


def make_mpl_plot_component(
    measure: str | dict[str, str] | list[str] | tuple[str],
    post_process: Callable | None = None,
    page: int = 0,
    save_format="png",
):
    """Create a plotting function for a specified measure.

    Args:
        measure (str | dict[str, str] | list[str] | tuple[str]): Measure(s) to plot.
        post_process: a user-specified callable to do post-processing called with the Axes instance.
        page: Page number where the plot should be displayed.
        save_format: save format of figure in solara backend

    Returns:
        (function, page): A tuple of a function that creates a PlotMatplotlib component and a page number.
    """

    def MakePlotMatplotlib(model):
        return PlotMatplotlib(
            model, measure, post_process=post_process, save_format=save_format
        )

    return (MakePlotMatplotlib, page)


@solara.component
def PlotMatplotlib(
    model,
    measure,
    dependencies: list[any] | None = None,
    post_process: Callable | None = None,
    save_format="png",
):
    """Create a Matplotlib-based plot for a measure or measures.

    Args:
        model (mesa.Model): The model instance.
        measure (str | dict[str, str] | list[str] | tuple[str]): Measure(s) to plot.
        dependencies (list[any] | None): Optional dependencies for the plot.
        post_process: a user-specified callable to do post-processing called with the Axes instance.
        save_format: format used for saving the figure.

    Returns:
        solara.FigureMatplotlib: A component for rendering the plot.
    """
    update_counter.get()
    fig = Figure()
    ax = fig.subplots()
    df = model.datacollector.get_model_vars_dataframe()
    if isinstance(measure, str):
        ax.plot(df.loc[:, measure])
        ax.set_ylabel(measure)
    elif isinstance(measure, dict):
        for m, color in measure.items():
            ax.plot(df.loc[:, m], label=m, color=color)
        ax.legend(loc="best")
    elif isinstance(measure, list | tuple):
        for m in measure:
            ax.plot(df.loc[:, m], label=m)
        ax.legend(loc="best")

    ax.set_xlabel("Step")
    # Set integer x axis
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    if post_process is not None:
        post_process(ax)

    solara.FigureMatplotlib(
        fig, format=save_format, bbox_inches="tight", dependencies=dependencies
    )
