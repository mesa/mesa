"""Tests for the custom Dict parameter example (Boltzmann Wealth model)."""

import ipyvuetify as v
import pytest
import solara

from mesa.examples.basic.boltzmann_wealth_model import custom_params
from mesa.visualization.user_param import Slider


def render(user_params, on_change=None):
    """Render CustomUserInputs and return the render context."""
    element = custom_params.CustomUserInputs(
        user_params, on_change=on_change or (lambda name, value: None)
    )
    _, rc = solara.render(element, handle_error=False)
    return rc


DICT_PARAM = {
    "type": "Dict",
    "label": "Grid",
    "entries": {
        "width": {
            "value": 10,
            "label": "Width",
            "type": "SliderInt",
            "min": 5,
            "max": 50,
            "step": 1,
        },
        "ratio": {
            "value": 0.5,
            "label": "Ratio",
            "type": "SliderFloat",
            "min": 0.0,
            "max": 1.0,
            "step": 0.1,
        },
        "name": {"value": "abc", "label": "Name"},  # defaults to text input
        "count": {"value": 3, "label": "Count"},  # text input, numeric
    },
}


def test_extract_dict_param():
    spec = {
        "type": "Dict",
        "entries": {
            "leaf": {"value": 1},
            "nested": {"inner": {"value": 2}},
            "direct": 3,
        },
    }
    assert custom_params.extract_dict_param(spec) == {
        "leaf": 1,
        "nested": {"inner": 2},
        "direct": 3,
    }


def test_extract_dict_param_empty():
    assert custom_params.extract_dict_param({"type": "Dict"}) == {}


def test_renders_all_supported_types():
    rc = render(
        {
            "int": {
                "type": "SliderInt",
                "value": 5,
                "label": "Int",
                "min": 0,
                "max": 10,
                "step": 1,
            },
            "float": {
                "type": "SliderFloat",
                "value": 0.5,
                "label": "Float",
                "min": 0.0,
                "max": 1.0,
                "step": 0.1,
            },
            "select": {
                "type": "Select",
                "value": "a",
                "label": "Select",
                "values": ["a", "b"],
            },
            "check": {"type": "Checkbox", "value": True, "label": "Check"},
            "text": {"type": "InputText", "value": "hi", "label": "Text"},
            "slider_obj": Slider(label="Obj", value=5, min=1, max=10, step=1),
            "grid": DICT_PARAM,
        }
    )
    rc.close()


def test_unsupported_type_raises():
    with pytest.raises(ValueError, match="not a supported input type"):
        render({"bad": {"type": "Nope", "value": 1}})


def test_checkbox_change_calls_on_change():
    calls = []
    rc = render(
        {"check": {"type": "Checkbox", "value": True, "label": "Check"}},
        on_change=lambda name, value: calls.append((name, value)),
    )
    rc.find(v.Checkbox).widget.v_model = False
    assert calls == [("check", False)]
    rc.close()


def test_dict_field_change_calls_on_change():
    calls = []
    rc = render(
        {"grid": DICT_PARAM},
        on_change=lambda name, value: calls.append((name, value)),
    )
    sliders = {w.label: w for w in rc.find(v.Slider).widgets}

    sliders["Width"].v_model = 20
    name, value = calls[-1]
    assert name == "grid"
    assert value["width"] == 20

    sliders["Ratio"].v_model = 0.7
    name, value = calls[-1]
    assert name == "grid"
    assert value["ratio"] == 0.7
    assert value["width"] == 20  # earlier change is preserved

    rc.close()
