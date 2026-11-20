"""Custom visualization components."""

from __future__ import annotations

from collections.abc import Callable

from .altair_components import make_altair_plot_component
from .matplotlib_components import make_mpl_plot_component
from .portrayal_components import AgentPortrayalStyle, PropertyLayerStyle

__all__ = [
    "AgentPortrayalStyle",
    "PropertyLayerStyle",
    "make_mpl_plot_component",
    "make_plot_component",
]


def make_plot_component(
    measure: str | dict[str, str] | list[str] | tuple[str],
    post_process: Callable | None = None,
    backend: str = "matplotlib",
    page: int = 0,
    **plot_drawing_kwargs,
):
    """Create a plotting function for a specified measure using the specified backend.

    Args:
        measure (str | dict[str, str] | list[str] | tuple[str]): Measure(s) to plot.
        post_process: a user-specified callable to do post-processing called with the Axes instance.
        backend: the backend to use {"matplotlib", "altair"}
        page: Page number where the plot should be displayed (default 0).
        plot_drawing_kwargs: additional keyword arguments to pass onto the backend specific function for making a plotting component

    Returns:
        (function, page): A tuple of a function and page number that creates a plot component on that specific page.
    """
    if backend == "matplotlib":
        return make_mpl_plot_component(
            measure, post_process, page, **plot_drawing_kwargs
        )
    elif backend == "altair":
        return make_altair_plot_component(
            measure, post_process, page, **plot_drawing_kwargs
        )
    else:
        raise ValueError(
            f"unknown backend {backend}, must be one of matplotlib, altair"
        )
