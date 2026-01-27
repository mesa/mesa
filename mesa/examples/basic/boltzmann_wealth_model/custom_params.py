"""Custom parameter types and UI components for Mesa visualizations.

This module demonstrates how to extend Mesa's parameter system with custom types
using the parameter extractor registry.
"""

import solara

from mesa.visualization.solara_viz import register_param_extractor
from mesa.visualization.user_param import Slider


def extract_dict_param(spec):
    """Extract default values from a nested dictionary parameter specification.

    This extractor handles nested dictionaries where leaf nodes have a "value" key.

    Args:
        spec: Parameter specification with structure:
            {
                "type": "Dict",
                "entries": {
                    "key1": {"value": default_value1},
                    "key2": {"value": default_value2},
                }
            }

    Returns:
        dict: Extracted default values as a dictionary
    """
    entries = spec.get("entries", {})
    result = {}

    for key, value in entries.items():
        if isinstance(value, dict) and "value" in value:
            # Leaf node with explicit value
            result[key] = value["value"]
        elif isinstance(value, dict):
            # Nested dictionary - recurse
            result[key] = extract_dict_param({"entries": value})
        else:
            # Direct value
            result[key] = value

    return result


# Register the Dict extractor
register_param_extractor("Dict", extract_dict_param)


@solara.component
def DictInput(name, options, on_change):
    """Render a Dict parameter as a collapsible section with individual controls.

    Args:
        name: Parameter name
        options: Dict parameter specification with "entries"
        on_change: Callback function for value changes
    """
    # State for tracking the dict values
    dict_values = solara.use_reactive({})

    # Initialize dict values from entries
    def initialize():
        initial_values = extract_dict_param(options)
        dict_values.set(initial_values)

    solara.use_effect(initialize, [])

    # Handler for individual field changes
    def field_change_handler(field_name, value):
        new_dict = {**dict_values.value, field_name: value}
        dict_values.set(new_dict)
        on_change(name, new_dict)

    label = options.get("label", name)

    # Render as expansion panel
    with solara.v.ExpansionPanels(v_model=[0], multiple=True):
        with solara.v.ExpansionPanel():
            with solara.v.ExpansionPanelHeader():
                solara.Text(f"{label}")
            with solara.v.ExpansionPanelContent():
                entries = options.get("entries", {})
                for key, value_spec in entries.items():
                    if isinstance(value_spec, dict) and "value" in value_spec:
                        # Render based on the field's metadata
                        field_label = value_spec.get("label", key)
                        field_type = value_spec.get("type", "InputText")
                        current_value = dict_values.value.get(key, value_spec["value"])

                        if field_type == "SliderInt":
                            solara.SliderInt(
                                field_label,
                                value=current_value,
                                on_value=lambda v, k=key: field_change_handler(k, v),
                                min=value_spec.get("min", 0),
                                max=value_spec.get("max", 100),
                                step=value_spec.get("step", 1),
                            )
                        elif field_type == "SliderFloat":
                            solara.SliderFloat(
                                field_label,
                                value=current_value,
                                on_value=lambda v, k=key: field_change_handler(k, v),
                                min=value_spec.get("min", 0.0),
                                max=value_spec.get("max", 1.0),
                                step=value_spec.get("step", 0.1),
                            )
                        else:
                            # Default to text input
                            solara.InputText(
                                field_label,
                                value=str(current_value),
                                on_value=lambda v, k=key: field_change_handler(
                                    k, int(v) if v.isdigit() else v
                                ),
                            )


@solara.component
def CustomUserInputs(user_params, on_change=None):
    """Extended UserInputs that supports Dict type parameters.

    This component extends Mesa's default UserInputs to handle custom Dict
    parameters, rendering them as collapsible sections with individual controls.

    Args:
        user_params: Dictionary of parameter specifications
        on_change: Callback function called with (name, value) when values change
    """
    for name, options in user_params.items():

        def change_handler(value, name=name):
            on_change(name, value)

        # Handle Slider objects
        if isinstance(options, Slider):
            slider_class = (
                solara.SliderFloat if options.is_float_slider else solara.SliderInt
            )
            slider_class(
                options.label,
                value=options.value,
                on_value=change_handler,
                min=options.min,
                max=options.max,
                step=options.step,
            )
            continue

        # Get label and type
        label = options.get("label", name)
        input_type = options.get("type")

        # Handle Dict type with collapsible UI
        if input_type == "Dict":
            DictInput(name, options, on_change)
            continue

        # Handle standard Mesa input types
        if input_type == "SliderInt":
            solara.SliderInt(
                label,
                value=options.get("value"),
                on_value=change_handler,
                min=options.get("min"),
                max=options.get("max"),
                step=options.get("step"),
            )
        elif input_type == "SliderFloat":
            solara.SliderFloat(
                label,
                value=options.get("value"),
                on_value=change_handler,
                min=options.get("min"),
                max=options.get("max"),
                step=options.get("step"),
            )
        elif input_type == "Select":
            solara.Select(
                label,
                value=options.get("value"),
                on_value=change_handler,
                values=options.get("values"),
            )
        elif input_type == "Checkbox":
            solara.Checkbox(
                label=label,
                on_value=change_handler,
                value=options.get("value"),
            )
        elif input_type == "InputText":
            solara.InputText(
                label=label,
                on_value=change_handler,
                value=options.get("value"),
            )
        else:
            raise ValueError(f"{input_type} is not a supported input type")
