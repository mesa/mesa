"""Tests for the portrayal components in Mesa visualization."""

import re
from dataclasses import is_dataclass

import pytest

from mesa.visualization.components import AgentPortrayalStyle, PropertyLayerStyle


def test_agent_portrayal_style_is_dataclass():
    """Test if AgentPortrayalStyle is a dataclass."""
    assert is_dataclass(AgentPortrayalStyle)


def test_agent_portrayal_style_defaults():
    """Test default values of AgentPortrayalStyle."""
    style = AgentPortrayalStyle()
    assert style.x is None
    assert style.y is None
    assert style.color == "tab:blue"
    assert style.marker == "o"
    assert style.size == 50
    assert style.zorder == 1
    assert style.alpha == 1.0
    assert style.edgecolors is None
    assert style.linewidths == 1.0


def test_agent_portrayal_style_custom_initialization():
    """Test custom initialization of AgentPortrayalStyle."""
    style = AgentPortrayalStyle(
        x=10,
        y=20,
        color="red",
        marker="^",
        size=100,
        zorder=2,
        alpha=0.5,
        edgecolors="black",
        linewidths=2.0,
    )
    assert style.x == 10
    assert style.y == 20
    assert style.color == "red"
    assert style.marker == "^"
    assert style.size == 100
    assert style.zorder == 2
    assert style.alpha == 0.5
    assert style.edgecolors == "black"
    assert style.linewidths == 2.0


def test_agent_portrayal_style_update_attributes():
    """Test updating attributes of AgentPortrayalStyle."""
    style = AgentPortrayalStyle()
    style.update(("marker", "s"), ("size", 75), ("alpha", 0.7))
    assert style.marker == "s"
    assert style.size == 75
    assert style.alpha == 0.7


def test_agent_portrayal_style_update_non_existent_attribute():
    """Test updating a non-existent attribute raises AttributeError."""
    style = AgentPortrayalStyle()
    with pytest.raises(
        AttributeError,
        match=re.escape("'AgentPortrayalStyle' object has no attribute 'shape'"),
    ):
        style.update(("shape", "triangle"))


def test_agent_portrayal_style_update_with_no_arguments():
    """Test updating AgentPortrayalStyle with no arguments does not change the style."""
    original_style = AgentPortrayalStyle(color="blue")
    updated_style = AgentPortrayalStyle(color="blue")
    updated_style.update()
    assert updated_style.color == original_style.color  # Ensure no change


def test_property_layer_style_is_dataclass():
    """Test if PropertyLayerStyle is a dataclass."""
    assert is_dataclass(PropertyLayerStyle)


def test_property_layer_style_default_values_with_colormap():
    """Test default values of PropertyLayerStyle with colormap."""
    style = PropertyLayerStyle(colormap="viridis")
    assert style.colormap == "viridis"
    assert style.color is None
    assert style.alpha == 0.8
    assert style.colorbar is True
    assert style.vmin is None
    assert style.vmax is None


def test_property_layer_style_default_values_with_color():
    """Test default values of PropertyLayerStyle with color."""
    style = PropertyLayerStyle(color="red")
    assert style.colormap is None
    assert style.color == "red"
    assert style.alpha == 0.8
    assert style.colorbar is True
    assert style.vmin is None
    assert style.vmax is None


def test_property_layer_style_custom_initialization_with_colormap():
    """Test custom initialization of PropertyLayerStyle with colormap."""
    style = PropertyLayerStyle(
        colormap="plasma", alpha=0.5, colorbar=False, vmin=0, vmax=1
    )
    assert style.colormap == "plasma"
    assert style.color is None
    assert style.alpha == 0.5
    assert style.colorbar is False
    assert style.vmin == 0
    assert style.vmax == 1


def test_property_layer_style_custom_initialization_with_color():
    """Test custom initialization of PropertyLayerStyle with color."""
    style = PropertyLayerStyle(color="blue", alpha=0.9, colorbar=False)
    assert style.colormap is None
    assert style.color == "blue"
    assert style.alpha == 0.9
    assert style.colorbar is False


def test_property_layer_style_post_init_both_color_and_colormap_error():
    """Test error when both color and colormap are specified."""
    with pytest.raises(
        ValueError, match=re.escape("Specify either 'color' or 'colormap', not both.")
    ):
        PropertyLayerStyle(colormap="viridis", color="red")


def test_property_layer_style_post_init_neither_color_nor_colormap_error():
    """Test error when neither color nor colormap is specified."""
    with pytest.raises(
        ValueError, match=re.escape("Specify one of 'color' or 'colormap'")
    ):
        PropertyLayerStyle()
