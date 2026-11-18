"""Solara based visualization for Mesa models.

.. note::
    SolaraViz is experimental and still in active development in Mesa 3.x. While we attempt to minimize them, there might be API breaking changes in minor releases.

"""

from .command_console import CommandConsole
from .components import make_plot_component
from .solara_viz import JupyterViz, SolaraViz
from .space_renderer import SpaceRenderer
from .user_param import Slider

__all__ = [
    "CommandConsole",
    "JupyterViz",
    "Slider",
    "SolaraViz",
    "SpaceRenderer",
    "make_plot_component",
]
