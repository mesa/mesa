# mesa/visualization/solara_viz.py
from typing import Any

import solara


def is_solara_available() -> bool:
    """Check if Solara is installed."""
    try:
        import solara

        return True
    except ImportError:
        return False


def visualize_model(model: Any):
    """Solara visualization for Mesa models.
    Shows agents on a grid dynamically.
    """
    if not is_solara_available():
        return solara.Text("Solara is not available.")

    width = getattr(model.grid, "width", 10)
    height = getattr(model.grid, "height", 10)

    # Build grid matrix
    grid_matrix = [[0 for _ in range(width)] for _ in range(height)]
    for agent in model.schedule.agents:
        x, y = agent.pos
        grid_matrix[y][x] = 1  # mark agent presence

    def render_grid():
        return "\n".join(
            " ".join("🟩" if cell else "⬜" for cell in row) for row in grid_matrix
        )

<<<<<<< HEAD
    for name in model_parameters:
        if (
            model_parameters[name].default == inspect.Parameter.empty
            and name not in model_params
            and name != "self"
            and name != "kwargs"
        ):
            raise ValueError(f"Missing required model parameter: {name}")
    for name in model_params:
        if name not in model_parameters and "kwargs" not in model_parameters:
            raise ValueError(f"Invalid model parameter: {name}")


@solara.component
def UserInputs(user_params, on_change=None):
    """Initialize user inputs for configurable model parameters.

    Currently supports :class:`solara.SliderInt`, :class:`solara.SliderFloat`,
    :class:`solara.Select`, and :class:`solara.Checkbox`.

    Args:
        user_params: Dictionary with options for the input, including label, min and max values, and other fields specific to the input type.
        on_change: Function to be called with (name, value) when the value of an input changes.
    """
    for name, options in user_params.items():

        def change_handler(value, name=name):
            on_change(name, value)

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

        # label for the input is "label" from options or name
        label = options.get("label", name)
        input_type = options.get("type")
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


def make_initial_grid_layout(num_components):
    """Create an initial grid layout for visualization components.

    Args:
        num_components: Number of components to display

    Returns:
        list: Initial grid layout configuration
    """
    return [
        {
            "i": i,
            "w": 6,
            "h": 10,
            "moved": False,
            "x": 6 * (i % 2),
            "y": 16 * (i - i % 2),
        }
        for i in range(num_components)
    ]


def copy_renderer(renderer: SpaceRenderer, model: Model):
    """Create a new renderer instance with the same configuration as the original."""
    new_renderer = renderer.__class__(model=model, backend=renderer.backend)

    attributes_to_copy = [
        "agent_portrayal",
        "propertylayer_portrayal",
        "space_kwargs",
        "agent_kwargs",
        "space_mesh",
        "agent_mesh",
        "propertylayer_mesh",
        "post_process_func",
    ]

    for attr in attributes_to_copy:
        if hasattr(renderer, attr):
            value_to_copy = getattr(renderer, attr)
            setattr(new_renderer, attr, value_to_copy)

    return new_renderer


@solara.component
def ShowSteps(model):
    """Display the current step of the model."""
    update_counter.get()
    return solara.Text(f"Step: {model.steps}")


def is_solara_available() -> bool:
    """Check whether Solara is installed and importable.

    Returns:
    -------
    bool
        True if solara can be imported, False otherwise.
    """
    try:
        import solara  # noqa: F401

        return True
    except ImportError:
        return False
=======
    return solara.VBox(
        [
            solara.Text(f"Step: {getattr(model, 'steps', '?')}"),
            solara.Text(render_grid()),
        ]
    )
>>>>>>> 4fa239ca (Add initial Solara visualization for Mesa models)
