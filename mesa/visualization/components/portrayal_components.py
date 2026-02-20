"""Portrayal Components Module.

This module defines data structures for styling visual elements in Mesa agent-based model visualizations.
It provides user-facing classes to specify how agents and property layers should appear in the rendered space.

Classes:
1. AgentPortrayalStyle: Controls the appearance of individual agents (e.g., color, shape, size, etc.).
2. PropertyStyle: Controls the appearance of background property layers (e.g., color gradients or uniform fills).

These components are designed to be passed into Mesa visualizations to customize and standardize how data is presented.
"""

from dataclasses import dataclass
from typing import Any

ColorLike = str | tuple | int | float


@dataclass
class AgentPortrayalStyle:
    """Represents the visual styling options for an agent in a visualization.

    User facing component to control how agents are drawn.
    Allows specifying properties like color, size,
    marker shape, position, and other plot attributes.

    x, y are determined automatically according to the agent's type
    (normal/CellAgent) and position in the space if not manually declared.

    Example:
        >>> def agent_portrayal(agent):
        >>>     return AgentPortrayalStyle(
        >>>         x=agent.cell.coordinate[0],
        >>>         y=agent.cell.coordinate[1],
        >>>         color="red",
        >>>         marker="o",
        >>>         size=20,
        >>>         zorder=2,
        >>>         alpha=0.8,
        >>>         edgecolors="black",
        >>>         linewidths=1.5
        >>>     )
        >>>
        >>> # or for a default agent portrayal
        >>> def agent_portrayal(agent):
        >>>     return AgentPortrayalStyle()
    """

    x: float | None = None
    y: float | None = None
    color: ColorLike | None = "tab:blue"
    marker: str | None = "o"
    size: int | float | None = 50
    zorder: int | None = 1
    alpha: float | None = 1.0
    edgecolors: str | tuple | None = None
    linewidths: float | int | None = 1.0

    def update(self, *updates_fields: tuple[str, Any]):
        """Updates attributes from variable (field_name, new_value) tuple arguments.

        Example:
            >>> def agent_portrayal(agent):
            >>>     primary_style = AgentPortrayalStyle(color="blue", marker="^", size=10, x=agent.pos[0], y=agent.pos[1])
            >>>     if agent.type == 1:
            >>>         primary_style.update(("color", "red"), ("size", 30))
            >>>     return primary_style
        """
        for field_to_change, field_to_change_to in updates_fields:
            if hasattr(self, field_to_change):
                setattr(self, field_to_change, field_to_change_to)
            else:
                raise AttributeError(
                    f"'{type(self).__name__}' object has no attribute '{field_to_change}'"
                )

    def __post_init__(self):
        """Validate that color and size are non-negative."""
        if isinstance(self.color, int | float) and self.color < 0:
            raise ValueError("Scalar color values must be non-negative")
        if isinstance(self.size, int | float) and self.size < 0:
            raise ValueError("Size must be a non-negative number")


@dataclass
class PropertyStyle:
    """Represents the visual styling options for a property layer in a visualization.

    User facing component to control how property layers are drawn.
    Allows specifying properties like colormap, single color, value limits
    (vmin, vmax), transparency (alpha) and colorbar visibility.

    Note: vmin and vmax are the lower and upper bounds for the colorbar and the data is
    normalized between these values for color/colormap rendering. If they are not
    declared the values are automatically determined from the data range.

    Note: You can specify either a 'colormap' (for varying data) or a single
    'color' (for a uniform layer appearance), but not both simultaneously.

    Example:
        >>> def property_portrayal(layer):
        >>>     return PropertyStyle(colormap="viridis", vmin=0, vmax=100, alpha=0.5, colorbar=True)
        >>> # or for a uniform color layer
        >>> def property_portrayal(layer):
        >>>     return PropertyStyle(color="lightblue", alpha=0.8, colorbar=False)
    """

    colormap: str | None = None
    color: str | None = None
    alpha: float = 0.8
    colorbar: bool = True
    vmin: float | None = None
    vmax: float | None = None

    def __post_init__(self):
        """Validate that color and colormap are not simultaneously specified."""
        if self.color is not None and self.colormap is not None:
            raise ValueError("Specify either 'color' or 'colormap', not both.")
        if self.color is None and self.colormap is None:
            raise ValueError("Specify one of 'color' or 'colormap'")
