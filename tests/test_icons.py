"""Tests for bundled SVG icon helpers (listing and retrieval)."""

import pytest

from mesa.visualization import icons


def test_list_icons_contains_smiley():
    """list_icons() includes 'smiley' (adjust if needed)."""
    names = icons.list_icons()
    assert "smiley" in names
    assert "sad_face" in names
    assert "neutral_face" in names


def test_get_icon_svg_returns_text():
    """get_icon_svg('smiley') returns SVG text containing '<svg'."""
    svg = icons.get_icon_svg("smiley")
    assert "<svg" in svg

    svg = icons.get_icon_svg("sad_face")
    assert "<svg" in svg

    svg = icons.get_icon_svg("neutral_face")
    assert "<svg" in svg


def test_get_icon_svg_not_found():
    """get_icon_svg() raises for missing icon."""
    with pytest.raises(FileNotFoundError):
        icons.get_icon_svg("this-does-not-exist")


def test_get_icon_png_returns_bytes():
    """get_icon_png() returns PNG bytes if available."""
    try:
        png_bytes = icons.get_icon_png("smiley", size=32)
        assert isinstance(png_bytes, bytes)
        # PNG files start with PNG signature
        assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    except FileNotFoundError:
        pytest.skip("PNG icons not yet generated")


def test_get_icon_png_not_found():
    """get_icon_png() raises for missing icon."""
    with pytest.raises(FileNotFoundError):
        icons.get_icon_png("this-does-not-exist", size=32)
