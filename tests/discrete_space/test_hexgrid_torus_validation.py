"""Test validation for HexGrid torus configurations."""

import random

import pytest

from mesa.discrete_space import HexGrid


def test_hexgrid_torus_odd_dimensions_error():
    """Test that HexGrid raises ValueError when torus=True and dimensions are odd."""
    # Helper to assert error
    with pytest.raises(
        ValueError,
        match="HexGrid with torus=True requires both width and height to be even",
    ):
        HexGrid((5, 5), random=random.Random(42), torus=True)

    with pytest.raises(
        ValueError,
        match="HexGrid with torus=True requires both width and height to be even",
    ):
        HexGrid((5, 6), random=random.Random(42), torus=True)

    with pytest.raises(
        ValueError,
        match="HexGrid with torus=True requires both width and height to be even",
    ):
        HexGrid((6, 5), random=random.Random(42), torus=True)

    # Valid cases should not raise
    grid1 = HexGrid((6, 6), random=random.Random(42), torus=True)
    grid2 = HexGrid((5, 5), random=random.Random(42), torus=False)
    assert grid1 is not None
    assert grid2 is not None
