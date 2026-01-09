"""Tests for the user_param module."""

import pytest

from mesa.visualization.user_param import Slider, UserParam


class TestUserParam:
    """Tests for the UserParam base class."""

    def test_maybe_raise_error_valid(self):
        """Test that maybe_raise_error does not raise when valid is True."""
        param = UserParam()
        param.label = "test_label"
        # Should not raise
        param.maybe_raise_error("test_type", valid=True)

    def test_maybe_raise_error_invalid(self):
        """Test that maybe_raise_error raises ValueError when valid is False."""
        param = UserParam()
        param.label = "test_label"
        with pytest.raises(ValueError) as excinfo:
            param.maybe_raise_error("test_type", valid=False)
        assert "Missing or malformed inputs for 'test_type'" in str(excinfo.value)
        assert "test_label" in str(excinfo.value)


class TestSlider:
    """Tests for the Slider class."""

    def test_slider_initialization_with_int_values(self):
        """Test Slider initializes correctly with integer values."""
        slider = Slider(
            label="Test Slider",
            value=50,
            min=0,
            max=100,
            step=1,
        )
        assert slider.label == "Test Slider"
        assert slider.value == 50
        assert slider.min == 0
        assert slider.max == 100
        assert slider.step == 1
        assert slider.is_float_slider is False

    def test_slider_initialization_with_float_values(self):
        """Test Slider initializes correctly with float values."""
        slider = Slider(
            label="Float Slider",
            value=0.5,
            min=0.0,
            max=1.0,
            step=0.1,
        )
        assert slider.label == "Float Slider"
        assert slider.value == 0.5
        assert slider.min == 0.0
        assert slider.max == 1.0
        assert slider.step == 0.1
        assert slider.is_float_slider is True

    def test_slider_auto_detects_float_from_value(self):
        """Test Slider auto-detects float when value is a float."""
        slider = Slider(
            label="Test",
            value=5.5,
            min=0,
            max=10,
            step=1,
        )
        assert slider.is_float_slider is True

    def test_slider_auto_detects_float_from_min(self):
        """Test Slider auto-detects float when min is a float."""
        slider = Slider(
            label="Test",
            value=5,
            min=0.0,
            max=10,
            step=1,
        )
        assert slider.is_float_slider is True

    def test_slider_auto_detects_float_from_max(self):
        """Test Slider auto-detects float when max is a float."""
        slider = Slider(
            label="Test",
            value=5,
            min=0,
            max=10.0,
            step=1,
        )
        assert slider.is_float_slider is True

    def test_slider_auto_detects_float_from_step(self):
        """Test Slider auto-detects float when step is a float."""
        slider = Slider(
            label="Test",
            value=5,
            min=0,
            max=10,
            step=0.5,
        )
        assert slider.is_float_slider is True

    def test_slider_explicit_dtype_int(self):
        """Test Slider respects explicit dtype=int."""
        slider = Slider(
            label="Test",
            value=5.5,
            min=0.0,
            max=10.0,
            step=0.5,
            dtype=int,
        )
        assert slider.is_float_slider is False

    def test_slider_explicit_dtype_float(self):
        """Test Slider respects explicit dtype=float."""
        slider = Slider(
            label="Test",
            value=5,
            min=0,
            max=10,
            step=1,
            dtype=float,
        )
        assert slider.is_float_slider is True

    def test_slider_raises_when_value_missing(self):
        """Test Slider raises ValueError when value is None."""
        with pytest.raises(ValueError) as excinfo:
            Slider(
                label="Test",
                value=None,
                min=0,
                max=10,
            )
        assert "Missing or malformed inputs" in str(excinfo.value)

    def test_slider_raises_when_min_missing(self):
        """Test Slider raises ValueError when min is None."""
        with pytest.raises(ValueError) as excinfo:
            Slider(
                label="Test",
                value=5,
                min=None,
                max=10,
            )
        assert "Missing or malformed inputs" in str(excinfo.value)

    def test_slider_raises_when_max_missing(self):
        """Test Slider raises ValueError when max is None."""
        with pytest.raises(ValueError) as excinfo:
            Slider(
                label="Test",
                value=5,
                min=0,
                max=None,
            )
        assert "Missing or malformed inputs" in str(excinfo.value)

    def test_slider_get_method(self):
        """Test Slider get() method retrieves attributes correctly."""
        slider = Slider(
            label="Test",
            value=50,
            min=0,
            max=100,
            step=5,
        )
        assert slider.get("label") == "Test"
        assert slider.get("value") == 50
        assert slider.get("min") == 0
        assert slider.get("max") == 100
        assert slider.get("step") == 5
        assert slider.get("is_float_slider") is False

    def test_slider_default_step(self):
        """Test Slider uses default step of 1."""
        slider = Slider(
            label="Test",
            value=5,
            min=0,
            max=10,
        )
        assert slider.step == 1

    def test_slider_empty_label(self):
        """Test Slider works with empty label."""
        slider = Slider(
            label="",
            value=5,
            min=0,
            max=10,
        )
        assert slider.label == ""
