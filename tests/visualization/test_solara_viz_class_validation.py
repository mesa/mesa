"""Tests for passing a Model class instead of instance to SolaraViz."""

import pytest
import solara

from mesa import Model
from mesa.visualization import SolaraViz


class DummyModel(Model):
    """A dummy model for validating SolaraViz class vs instance checks."""

    def __init__(self):
        """Initialize the dummy model."""
        super().__init__()


def test_solara_viz_rejects_class_instead_of_instance():
    """Verify TypeError properly raised when passing a Model class."""
    with pytest.raises(TypeError, match="initialized model instance"):
        solara.render(
            SolaraViz(DummyModel, components=[], model_params={}), handle_error=False
        )


def test_solara_viz_rejects_reactive_class():
    """Verify TypeError properly raised when passing a reactive-wrapped Model class."""
    reactive_model = solara.reactive(DummyModel)
    with pytest.raises(TypeError, match="initialized model instance"):
        solara.render(
            SolaraViz(reactive_model, components=[], model_params={}),
            handle_error=False,
        )


def test_solara_viz_error_message_contains_hint():
    """Verify error message specifically provides users a hint regarding instantiation."""
    with pytest.raises(TypeError) as exc_info:
        solara.render(
            SolaraViz(DummyModel, components=[], model_params={}), handle_error=False
        )

    assert (
        "Did you mean: SolaraViz(DummyModel(), ...) instead of SolaraViz(DummyModel, ...)?"
        in str(exc_info.value)
    )


def test_solara_viz_accepts_model_instance():
    """Verify an instantiated model passes the instance validation check."""
    model_instance = DummyModel()
    # Should not raise any TypeError representing the validation error
    solara.render(
        SolaraViz(model_instance, components=[], model_params={}), handle_error=False
    )


def test_solara_viz_accepts_reactive_model_instance():
    """Verify a reactive-wrapped instantiated model passes the instance validation check."""
    model_instance = DummyModel()
    reactive_model = solara.reactive(model_instance)
    # Should not raise any validation error
    solara.render(
        SolaraViz(reactive_model, components=[], model_params={}), handle_error=False
    )
