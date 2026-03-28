"""Cell spaces for active, property-rich spatial modeling in Mesa.

Cell spaces extend Mesa's spatial modeling capabilities by making the space itself active -
each position (cell) can have properties and behaviors rather than just containing agents.
This enables more sophisticated environmental modeling and agent-environment interactions.

Key components:
- Cells: Active positions that can have properties and contain agents
- CellAgents: Agents that understand how to interact with cells
- Spaces: Different cell organization patterns (grids, networks, etc.)

This is particularly useful for models where the environment plays an active role,
like resource growth, pollution diffusion, or infrastructure networks. The cell
space system is experimental and under active development.
"""

from __future__ import annotations

from typing import Any

from mesa.discrete_space.cell import Cell
from mesa.discrete_space.cell_agent import (
    CellAgent,
    FixedAgent,
    Grid2DMovingAgent,
)
from mesa.discrete_space.cell_collection import CellCollection
from mesa.discrete_space.discrete_space import DiscreteSpace
from mesa.discrete_space.grid import (
    Grid,
    HexGrid,
    OrthogonalMooreGrid,
    OrthogonalVonNeumannGrid,
)
from mesa.discrete_space.voronoi import VoronoiGrid

_NETWORK_IMPORT_ERROR: ModuleNotFoundError | None = None
try:
    from mesa.discrete_space.network import Network
except ModuleNotFoundError as e:
    _NETWORK_IMPORT_ERROR = e


def __getattr__(name: str) -> Any:
    if name != "Network":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    if _NETWORK_IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "Network space requires the optional dependency 'networkx'. "
            'Install it via `pip install "mesa[network]"`.'
        ) from _NETWORK_IMPORT_ERROR

    from mesa.discrete_space.network import Network as Network  # noqa: PLC0415

    return Network


__all__ = [
    "Cell",
    "CellAgent",
    "CellCollection",
    "DiscreteSpace",
    "FixedAgent",
    "Grid",
    "Grid2DMovingAgent",
    "HexGrid",
    "OrthogonalMooreGrid",
    "OrthogonalVonNeumannGrid",
    "VoronoiGrid",
]

if _NETWORK_IMPORT_ERROR is None:
    __all__.append("Network")
