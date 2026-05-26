"""Generate a compact warehouse layout for the meta-agent example."""

from __future__ import annotations

import random
import string
from random import Random

import numpy as np

DEFAULT_ROWS = 8
DEFAULT_COLS = 8
DEFAULT_HEIGHT = 2
LOADING_DOCK_COORDS = [(0, 0, 0), (0, 2, 0)]
CHARGING_STATION_COORDS = [(7, 5, 0), (7, 7, 0)]


def generate_item_code(rng: Random) -> str:
    """Generate a short random inventory code."""
    letter = rng.choice(string.ascii_uppercase)
    number = rng.randint(10, 99)
    return f"{letter}{number}"


def make_warehouse(
    rows: int = DEFAULT_ROWS,
    cols: int = DEFAULT_COLS,
    height: int = DEFAULT_HEIGHT,
    rng: Random | None = None,
) -> np.ndarray:
    """Generate a 3D warehouse array with loading docks and inventory."""
    rng = rng or random.Random(0)

    warehouse = np.full((rows, cols, height), " ", dtype=object)

    for r, c, h in LOADING_DOCK_COORDS:
        warehouse[r, c, h] = "LD"

    for r, c, h in CHARGING_STATION_COORDS:
        warehouse[r, c, h] = "CS"

    for r in range(2, rows - 1, 3):
        for c in range(1, cols, 3):
            for h in range(height):
                warehouse[r, c, h] = generate_item_code(rng)

    return warehouse
