"""Custom visualization components."""

from __future__ import annotations

from collections.abc import Callable

from .altair_components import (
    SpaceAltair,
    make_altair_plot_component,
    make_altair_space,
)
from .matplotlib_components import (
    SpaceMatplotlib,
    make_mpl_plot_component,
    make_mpl_space_component,
)
from .portrayal_components import AgentPortrayalStyle, PropertyStyle

__all__ = [
    "AgentPortrayalStyle",
    "PropertyStyle",
    "SpaceAltair",
    "SpaceMatplotlib",
    "make_altair_space",
    "make_mpl_plot_component",
    "make_mpl_space_component",
    "make_plot_component",
    "make_space_component",
]


def make_space_component(
    agent_portrayal: Callable | None = None,
    property_portrayal: dict | None = None,
    post_process: Callable | None = None,
    backend: str = "matplotlib",
    **space_drawing_kwargs,
) -> SpaceMatplotlib | SpaceAltair:
    """Create a Matplotlib-based space visualization component.

    Args:
        agent_portrayal: Function to portray agents.
        property_portrayal: Dictionary of Property portrayal specifications
        post_process : a callable that will be called with the Axes instance. Allows for fine-tuning plots (e.g., control ticks)
        backend: the backend to use {"matplotlib", "altair"}
        space_drawing_kwargs : additional keyword arguments to be passed on to the underlying backend specific space drawer function. See
                               the functions for drawing the various spaces for the appropriate backend further details.


    Returns:
        function: A function that creates a space component
    """
    if backend == "matplotlib":
        return make_mpl_space_component(
            agent_portrayal,
            property_portrayal,
            post_process,
            **space_drawing_kwargs,
        )
    elif backend == "altair":
        return make_altair_space(
            agent_portrayal,
            property_portrayal,
            post_process,
            **space_drawing_kwargs,
        )
    else:
        raise ValueError(
            f"unknown backend {backend}, must be one of matplotlib, altair"
        )


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
