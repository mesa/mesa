"""Test validation for HexGrid torus configurations."""

import pytest

from mesa.discrete_space import HexGrid


def test_hexgrid_torus_odd_dimensions_error():
    """Test that HexGrid raises ValueError when torus=True and dimensions are odd."""
    # Helper to assert error
    with pytest.raises(
        ValueError,
        match="HexGrid with torus=True requires both width and height to be even",
    ):
        HexGrid((5, 5), torus=True)

    with pytest.raises(
        ValueError,
        match="HexGrid with torus=True requires both width and height to be even",
    ):
        HexGrid((5, 6), torus=True)

    with pytest.raises(
        ValueError,
        match="HexGrid with torus=True requires both width and height to be even",
    ):
        HexGrid((6, 5), torus=True)

    # Valid cases should not raise
    try:
        HexGrid((6, 6), torus=True)
        HexGrid((5, 5), torus=False)
    except ValueError:
        pytest.fail("Valid HexGrid configurations should not raise ValueError")
