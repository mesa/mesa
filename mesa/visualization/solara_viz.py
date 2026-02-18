"""Mesa visualization module for creating interactive model visualizations.

This module provides components to create browser- and Jupyter notebook-based visualizations of
Mesa models, allowing users to watch models run step-by-step and interact with model parameters.

Key features:
    - SolaraViz: Main component for creating visualizations, supporting grid displays and plots
    - ModelController: Handles model execution controls (step, play, pause, reset)
    - UserInputs: Generates UI elements for adjusting model parameters

The module uses Solara for rendering in Jupyter notebooks or as standalone web applications.
It supports various types of visualizations including matplotlib plots, agent grids, and
custom visualization components.

Usage:
    1. Define an agent_portrayal function to specify how agents should be displayed
    2. Set up model_params to define adjustable parameters
    3. Create a SolaraViz instance with your model, parameters, and desired measures
    4. Display the visualization in a Jupyter notebook or run as a Solara app

See the Visualization Tutorial and example models for more details.
"""

from __future__ import annotations

import asyncio
import collections
import inspect
import itertools
import threading
import time
import traceback
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import altair as alt
import pandas as pd
import reacton.core
import solara
import solara.lab

import mesa.visualization.components.altair_components as components_altair
from mesa.experimental.devs.simulator import Simulator
from mesa.experimental.scenarios import Scenario
from mesa.mesa_logging import create_module_logger, function_logger
from mesa.visualization.command_console import CommandConsole
from mesa.visualization.space_renderer import SpaceRenderer
from mesa.visualization.user_param import Slider
from mesa.visualization.utils import force_update, update_counter

if TYPE_CHECKING:
    from mesa.model import Model

_mesa_logger = create_module_logger()


@solara.component
@function_logger(__name__)
def SolaraViz(
    model: Model | solara.Reactive[Model],
    renderer: SpaceRenderer | None = None,
    components: list[tuple[reacton.core.Component], int]
    | list[tuple[Callable[[Model], reacton.core.Component], 0]]
    | Literal["default"] = [],  # noqa: B006
    *,
    play_interval: int = 100,
    render_interval: int = 1,
    simulator: Simulator | None = None,
    model_params=None,
    name: str | None = None,
    use_threads: bool = False,
    **console_kwargs,
):
    """Solara visualization component.

    This component provides a visualization interface for a given model using Solara.
    It supports various visualization components and allows for interactive model
    stepping and parameter adjustments.

    Args:
        model (Model | solara.Reactive[Model]): A Model instance or a reactive Model.
            This is the main model to be visualized. If a non-reactive model is provided,
            it will be converted to a reactive model.
        renderer (SpaceRenderer): A SpaceRenderer instance to render the model's space.
        components (list[tuple[solara.component], int] | Literal["default"], optional): List of solara
            (components, page) or functions that return a solara (component, page).
            These components are used to render different parts of the model visualization.
            Defaults to "default", which uses the default Altair space visualization on page 0.
        play_interval (int, optional): Interval for playing the model steps in milliseconds.
            This controls the speed of the model's automatic stepping. Defaults to 100 ms.
        render_interval (int, optional): Controls how often plots are updated during a simulation,
            allowing users to skip intermediate steps and update graphs less frequently.
        use_threads: Flag for indicating whether to utilize multi-threading for model execution.
            When checked, the model will utilize multiple threads,adjust based on system capabilities.
        simulator: A simulator that controls the model (optional)
        model_params (dict, optional): Parameters for (re-)instantiating a model.
            Can include user-adjustable parameters and fixed parameters. Defaults to None.
        name (str | None, optional): Name of the visualization. Defaults to the model's class name.
        **console_kwargs (dict, optional): Arguments to pass to the command console.
            Currently supported arguments:
            - additional_imports: Dictionary of names to objects to import into the command console.
                - Example:
                    >>> console_kwargs = {"additional_imports": {"numpy": np}}
                    >>> SolaraViz(model, console_kwargs=console_kwargs)

    Returns:
        solara.component: A Solara component that renders the visualization interface for the model.

    Example:
        >>> model = MyModel()
        >>> page = SolaraViz(model)
        >>> page

    Notes:
        - The `model` argument can be either a direct model instance or a reactive model. If a direct
          model instance is provided, it will be converted to a reactive model using `solara.use_reactive`.
        - The `play_interval` argument controls the speed of the model's automatic stepping. A lower
          value results in faster stepping, while a higher value results in slower stepping.
        - The `render_interval` argument determines how often plots are updated during simulation. Higher values
          reduce update frequency, resulting in faster execution.
    """
    if components == "default":
        components = [
            (
                components_altair.make_altair_space(
                    agent_portrayal=None,
                    propertylayer_portrayal=None,
                    post_process=None,
                ),
                0,
            )
        ]
    if model_params is None:
        model_params = {}

    # Convert model to reactive
    if not isinstance(model, solara.Reactive):
        model = solara.use_reactive(model)  # noqa: RUF100  # noqa: SH102

    reactive_model_parameters = solara.use_reactive({})
    reactive_play_interval = solara.use_reactive(play_interval)
    reactive_render_interval = solara.use_reactive(render_interval)
    reactive_use_threads = solara.use_reactive(use_threads)

    # Make a copy of the components to avoid modifying the original list
    display_components = list(components)
    # Create space component based on the renderer
    if renderer is not None:
        if isinstance(renderer, SpaceRenderer):
            renderer = solara.use_reactive(renderer)  # noqa: RUF100  # noqa: SH102
        display_components.insert(0, (create_space_component(renderer.value), 0))

    with solara.AppBar():
        solara.AppBarTitle(name if name else type(model.value).__name__)
        solara.lab.ThemeToggle()

    with solara.Sidebar(), solara.Column():
        with solara.Card("Controls"):
            solara.SliderInt(
                label="Play Interval (ms)",
                value=reactive_play_interval,
                on_value=lambda v: reactive_play_interval.set(v),
                min=1,
                max=500,
                step=10,
            )
            solara.SliderInt(
                label="Render Interval (steps)",
                value=reactive_render_interval,
                on_value=lambda v: reactive_render_interval.set(v),
                min=1,
                max=100,
                step=2,
            )
            if reactive_use_threads.value:
                solara.Text("Increase play interval to avoid skipping plots")

            def set_reactive_use_threads(value):
                reactive_use_threads.set(value)

            solara.Checkbox(
                label="Use Threads",
                value=reactive_use_threads,
                on_value=set_reactive_use_threads,
            )

            if not isinstance(simulator, Simulator):
                ModelController(
                    model,
                    renderer=renderer,
                    model_parameters=reactive_model_parameters,
                    play_interval=reactive_play_interval,
                    render_interval=reactive_render_interval,
                    use_threads=reactive_use_threads,
                )
            else:
                SimulatorController(
                    model,
                    simulator,
                    renderer=renderer,
                    model_parameters=reactive_model_parameters,
                    play_interval=reactive_play_interval,
                    render_interval=reactive_render_interval,
                    use_threads=reactive_use_threads,
                )
        with solara.Card("Model Parameters"):
            ModelCreator(
                model, model_params, model_parameters=reactive_model_parameters
            )
        with solara.Card("Information"):
            ShowSteps(model.value)
        if (
            CommandConsole in display_components
        ):  # If command console in components show it in sidebar
            display_components.remove(CommandConsole)
            additional_imports = console_kwargs.get("additional_imports", {})
            with solara.Card("Command Console"):
                CommandConsole(model.value, additional_imports=additional_imports)

    # Render the main components view
    ComponentsView(display_components, model.value)


def create_space_component(renderer: SpaceRenderer):
    """Create a space visualization component for the given renderer."""

    def SpaceVisualizationComponent(model: Model):
        """Component that renders the model's space using the provided renderer."""
        return SpaceRendererComponent(model, renderer)

    return SpaceVisualizationComponent


@solara.component
def SpaceRendererComponent(
    model: Model,
    renderer: SpaceRenderer,
    # FIXME: Manage dependencies properly
    dependencies: list[Any] | None = None,
):
    """Render the space of a model using a SpaceRenderer.

    Args:
        model (Model): The model whose space is to be rendered.
        renderer: A SpaceRenderer instance to render the model's space.
        dependencies (list[any], optional): List of dependencies for the component.
    """
    update_counter.get()

    # update renderer's space according to the model's space/grid
    renderer.space = getattr(model, "grid", getattr(model, "space", None))

    viz_dependencies = [update_counter.value]
    if dependencies:
        viz_dependencies.extend(dependencies)

    if renderer.backend == "matplotlib":
        # Clear the previous plotted data and agents
        all_artists = [
            renderer.canvas.lines[:],
            renderer.canvas.collections[:],
            renderer.canvas.patches[:],
            renderer.canvas.images[:],
            renderer.canvas.artists[:],
        ]

        # Remove duplicate colorbars from the canvas
        for cbar in renderer.backend_renderer._active_colorbars:
            cbar.remove()
        renderer.backend_renderer._active_colorbars.clear()

        # Chain them together into a single iterable
        for artist in itertools.chain.from_iterable(all_artists):
            artist.remove()

        if renderer.space_mesh:
            renderer.draw_structure()
        if renderer.agent_mesh:
            renderer.draw_agents()
        if renderer.propertylayer_mesh:
            renderer.draw_propertylayer()

        if renderer.post_process and not renderer._post_process_applied:
            renderer.post_process(renderer.canvas)
            renderer._post_process_applied = True

        solara.FigureMatplotlib(
            renderer.canvas.get_figure(),
            format="png",
            bbox_inches="tight",
            dependencies=viz_dependencies,
        )
        return None
    else:
        structure = renderer.space_mesh if renderer.space_mesh else None
        agents = renderer.agent_mesh if renderer.agent_mesh else None
        propertylayer = renderer.propertylayer_mesh or None

        if renderer.space_mesh:
            structure = renderer.draw_structure()
        if renderer.agent_mesh:
            agents = renderer.draw_agents()
        if renderer.propertylayer_mesh:
            propertylayer = renderer.draw_propertylayer()

        spatial_charts_list = [
            chart for chart in [structure, propertylayer, agents] if chart
        ]

        final_chart = None
        if spatial_charts_list:
            final_chart = (
                spatial_charts_list[0]
                if len(spatial_charts_list) == 1
                else alt.layer(*spatial_charts_list).resolve_axis(
                    x="independent", y="independent"
                )
            )

        if final_chart is None:
            # If no charts are available, return an empty chart
            final_chart = (
                alt.Chart(pd.DataFrame()).mark_point().properties(width=450, height=350)
            )

        if renderer.post_process:
            final_chart = renderer.post_process(final_chart)

        final_chart = final_chart.configure_view(stroke="black", strokeWidth=1.5)

        solara.FigureAltair(final_chart, on_click=None, on_hover=None)
        return None


def _wrap_component(
    component: reacton.core.Component | Callable[[Model], reacton.core.Component],
) -> reacton.core.Component:
    """Wrap a component in an auto-updated Solara component if needed."""
    if isinstance(component, reacton.core.Component):
        return component

    @solara.component
    def WrappedComponent(model):
        update_counter.get()
        return component(model)

    return WrappedComponent


@solara.component
def ComponentsView(
    components: list[tuple[reacton.core.Component], int]
    | list[tuple[Callable[[Model], reacton.core.Component], int]],
    model: Model,
):
    """Display a list of components.

    Args:
        components: List of (components, page) to display
        model: Model instance to pass to each component
    """
    # State for current tab and layouts
    current_tab_index, set_current_tab_index = solara.use_state(0)
    layouts, set_layouts = solara.use_state({})

    # Backward's compatibility, page = 0 if not passed.
    for i, comp in enumerate(components):
        if not isinstance(comp, tuple):
            components[i] = (comp, 0)

    # Build pages mapping
    pages = collections.defaultdict(list)
    for component, page_index in components:
        pages[page_index].append(_wrap_component(component))

    # Fill in missing page indices for sequential tab order
    all_indices = sorted(pages.keys())
    if len(all_indices) > 1:
        min_page, max_page = all_indices[0], all_indices[-1]
        all_indices = list(range(min_page, max_page + 1))
        for idx in all_indices:
            pages.setdefault(idx, [])

    sorted_page_indices = all_indices

    # Keep layouts in sync with pages
    def sync_layouts():
        if not components:  # Handle empty case inside the effect
            return

        current_keys = set(pages.keys())
        layout_keys = set(layouts.keys())

        # Add layouts for new pages
        new_layouts = {
            index: make_initial_grid_layout(len(pages[index]))
            for index in current_keys - layout_keys
        }

        # Remove layouts for deleted pages
        cleaned_layouts = {k: v for k, v in layouts.items() if k in current_keys}

        if new_layouts or len(cleaned_layouts) != len(layouts):
            set_layouts({**cleaned_layouts, **new_layouts})

    solara.use_effect(sync_layouts, list(pages.keys()))

    # Allow early return (which is now safe because all hooks have been called)
    if not components:
        return

    # Tab Navigation
    with solara.v.Tabs(v_model=current_tab_index, on_v_model=set_current_tab_index):
        for index in sorted_page_indices:
            solara.v.Tab(children=[f"Page {index}"])

    with solara.v.Window(v_model=current_tab_index):
        for i, page_id in enumerate(sorted_page_indices):
            with solara.v.WindowItem():
                # Use tab index 'i' instead of 'page_id' because solara.v.Window
                # Tracks active tabs using 0-based indexing, which may differ from page IDs.
                if i == current_tab_index:
                    page_components = pages[page_id]
                    page_layout = layouts.get(page_id)

                    if page_layout:

                        def on_layout_change(new_layout, current_page_id=page_id):
                            set_layouts(
                                lambda old: {**old, current_page_id: new_layout}
                            )

                        solara.GridDraggable(
                            items=[c(model) for c in page_components],
                            grid_layout=page_layout,
                            resizable=True,
                            draggable=True,
                            on_grid_layout=on_layout_change,
                        )


JupyterViz = SolaraViz


def _get_scenario_defaults(scenario: Scenario) -> dict[str, Any]:
    """Return defaults for a scenario instance, validating scenario type."""
    scenario_class = type(scenario)
    if not issubclass(scenario_class, Scenario):
        raise ValueError(
            "Expected model.scenario to be a subclass instance of mesa.experimental.scenarios.Scenario."
        )

    defaults = getattr(scenario, "_scenario_defaults", {}).copy()
    defaults["rng"] = None

    return defaults


def _build_model_init_kwargs(
    model: Model,
    model_parameters: dict[str, Any],
    *,
    add_scenario_when_empty: bool,
    require_model_accepts_scenario: bool,
) -> dict[str, Any]:
    """Build kwargs for model re-instantiation, splitting scenario/model params."""
    kwargs: dict[str, Any] = {}
    scenario_kwargs: dict[str, Any] = {}

    if getattr(model, "scenario", None):
        scenario = model.scenario
        scenario_class = type(scenario)
        scenario_defaults = _get_scenario_defaults(scenario)
        model_init_params = inspect.signature(type(model).__init__).parameters

        for key, value in model_parameters.items():
            if key in scenario_defaults and key not in model_init_params:
                scenario_kwargs[key] = value
            else:
                kwargs[key] = value

        has_kwargs = any(p.kind == p.VAR_KEYWORD for p in model_init_params.values())
        accepts_scenario = "scenario" in model_init_params or has_kwargs
        should_add_scenario = add_scenario_when_empty or bool(scenario_kwargs)

        if should_add_scenario and (
            accepts_scenario or not require_model_accepts_scenario
        ):
            kwargs["scenario"] = scenario_class(**scenario_kwargs)
    else:
        kwargs = {**model_parameters}

    return kwargs


@solara.component
def ModelController(
    model: solara.Reactive[Model],
    *,
    renderer: solara.Reactive[SpaceRenderer] | None = None,
    model_parameters: dict | solara.Reactive[dict] = None,
    play_interval: int | solara.Reactive[int] = 100,
    render_interval: int | solara.Reactive[int] = 1,
    use_threads: bool | solara.Reactive[bool] = False,
):
    """Create controls for model execution (step, play, pause, reset).

    Args:
        model: Reactive model instance
        renderer: SpaceRenderer instance to render the model's space.
        model_parameters: Reactive parameters for (re-)instantiating a model.
        play_interval: Interval for playing the model steps in milliseconds.
        render_interval: Controls how often the plots are updated during simulation steps.Higher value reduce update frequency.
        use_threads: Flag for indicating whether to utilize multi-threading for model execution.
    """
    playing = solara.use_reactive(False)
    running = solara.use_reactive(True)

    if model_parameters is None:
        model_parameters = {}
    model_parameters = solara.use_reactive(model_parameters)
    visualization_pause_event = solara.use_memo(lambda: threading.Event(), [])

    error_message = solara.use_reactive(None)

    def step():
        try:
            while running.value and playing.value:
                time.sleep(play_interval.value / 1000)
                do_step()
                if use_threads.value:
                    visualization_pause_event.set()
        except Exception as e:
            error_message.value = f"error in step: {e}"
            traceback.print_exc()
            return

    def visualization_task():
        if use_threads.value:
            try:
                while playing.value and running.value:
                    visualization_pause_event.wait()
                    visualization_pause_event.clear()
                    force_update()

            except Exception as e:
                error_message.value = f"error in visualization: {e}"
                traceback.print_exc()

    solara.lab.use_task(
        step, dependencies=[playing.value, running.value], prefer_threaded=True
    )

    solara.use_thread(
        visualization_task,
        dependencies=[playing.value, running.value],
    )

    @function_logger(__name__)
    def do_step():
        """Advance the model by the number of steps specified by the render_interval slider."""
        if playing.value:
            for _ in range(render_interval.value):
                model.value.step()
                running.value = model.value.running
                if not playing.value:
                    break
            if not use_threads.value:
                force_update()

        else:
            for _ in range(render_interval.value):
                model.value.step()
                running.value = model.value.running
            force_update()

    @function_logger(__name__)
    def do_reset():
        """Reset the model to its initial state."""
        error_message.set(None)
        playing.value = False
        running.value = True
        visualization_pause_event.clear()
        _mesa_logger.log(
            10,
            f"creating new {type(model.value)} instance with {model_parameters.value}",
        )
        kwargs = _build_model_init_kwargs(
            model.value,
            model_parameters.value,
            add_scenario_when_empty=True,
            require_model_accepts_scenario=True,
        )

        model.value = type(model.value)(**kwargs)
        if renderer is not None:
            renderer.value = copy_renderer(renderer.value, model.value)
            force_update()

    @function_logger(__name__)
    def do_play_pause():
        """Toggle play/pause."""
        playing.value = not playing.value

    with solara.Row(justify="space-between"):
        solara.Button(label="Reset", color="primary", on_click=do_reset)
        solara.Button(
            label="▶" if not playing.value else "❚❚",
            color="primary",
            on_click=do_play_pause,
            disabled=not running.value,
        )
        solara.Button(
            label="Step",
            color="primary",
            on_click=do_step,
            disabled=playing.value or not running.value,
        )

    if error_message.value:
        solara.Error(label=error_message.value)


@solara.component
def SimulatorController(
    model: solara.Reactive[Model],
    simulator,
    renderer: solara.Reactive[SpaceRenderer] | None = None,
    *,
    model_parameters: dict | solara.Reactive[dict] = None,
    play_interval: int | solara.Reactive[int] = 100,
    render_interval: int | solara.Reactive[int] = 1,
    use_threads: bool | solara.Reactive[bool] = False,
):
    """Create controls for model execution (step, play, pause, reset).

    Args:
        model: Reactive model instance
        simulator: Simulator instance
        renderer: SpaceRenderer instance to render the model's space.
        model_parameters: Reactive parameters for (re-)instantiating a model.
        play_interval: Interval for playing the model steps in milliseconds.
        render_interval: Controls how often the plots are updated during simulation steps.Higher values reduce update frequency.
        use_threads: Flag for indicating whether to utilize multi-threading for model execution.

    Notes:
        The `step button` increments the step by the value specified in the `render_interval` slider.
        This behavior ensures synchronization between simulation steps and plot updates.
    """
    playing = solara.use_reactive(False)
    running = solara.use_reactive(True)
    if model_parameters is None:
        model_parameters = {}
    model_parameters = solara.use_reactive(model_parameters)
    visualization_pause_event = solara.use_memo(lambda: threading.Event(), [])
    pause_step_event = solara.use_memo(lambda: threading.Event(), [])

    error_message = solara.use_reactive(None)

    def step():
        try:
            while running.value and playing.value:
                time.sleep(play_interval.value / 1000)
                if use_threads.value:
                    pause_step_event.wait()
                    pause_step_event.clear()
                do_step()
                if use_threads.value:
                    visualization_pause_event.set()
        except Exception as e:
            error_message.value = f"error in step: {e}"
            traceback.print_exc()

    def visualization_task():
        if use_threads.value:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                pause_step_event.set()
                while playing.value and running.value:
                    visualization_pause_event.wait()
                    visualization_pause_event.clear()
                    force_update()
                    pause_step_event.set()
            except Exception as e:
                error_message.value = f"error in visualization: {e}"
                traceback.print_exc()
                return

    solara.lab.use_task(
        step, dependencies=[playing.value, running.value], prefer_threaded=False
    )
    solara.lab.use_task(visualization_task, dependencies=[playing.value])

    def do_step():
        """Advance the model by the number of steps specified by the render_interval slider."""
        if playing.value:
            for _ in range(render_interval.value):
                simulator.run_for(1)
                running.value = model.value.running
                if not playing.value:
                    break
            if not use_threads.value:
                force_update()

        else:
            for _ in range(render_interval.value):
                simulator.run_for(1)
                running.value = model.value.running
            force_update()

    def do_reset():
        """Reset the model to its initial state."""
        error_message.set(None)
        playing.value = False
        running.value = True
        simulator.reset()
        visualization_pause_event.clear()
        pause_step_event.clear()

        kwargs = _build_model_init_kwargs(
            model.value,
            model_parameters.value,
            add_scenario_when_empty=False,
            require_model_accepts_scenario=False,
        )

        kwargs["simulator"] = simulator

        model.value = type(model.value)(**kwargs)
        if renderer is not None:
            renderer.value = copy_renderer(renderer.value, model.value)
            force_update()

    def do_play_pause():
        """Toggle play/pause."""
        playing.value = not playing.value

    with solara.Row(justify="space-between"):
        solara.Button(label="Reset", color="primary", on_click=do_reset)
        solara.Button(
            label="▶" if not playing.value else "❚❚",
            color="primary",
            on_click=do_play_pause,
            disabled=not running.value,
        )
        solara.Button(
            label="Step",
            color="primary",
            on_click=do_step,
            disabled=playing.value or not running.value,
        )
    if error_message.value:
        solara.Error(label=error_message.value)


def split_model_params(model_params):
    """Split model parameters into user-adjustable and fixed parameters.

    Args:
        model_params: Dictionary of all model parameters

    Returns:
        tuple: (user_adjustable_params, fixed_params)
    """
    model_params_input = {}
    model_params_fixed = {}
    for k, v in model_params.items():
        if check_param_is_fixed(v):
            model_params_fixed[k] = v
        else:
            model_params_input[k] = v
    return model_params_input, model_params_fixed


def check_param_is_fixed(param):
    """Check if a parameter is fixed (not user-adjustable).

    Args:
        param: Parameter to check

    Returns:
        bool: True if parameter is fixed, False otherwise
    """
    if isinstance(param, Slider):
        return False
    if not isinstance(param, dict):
        return True
    if "type" not in param:
        return True


@solara.component
def ModelCreator(
    model: solara.Reactive[Model],
    user_params: dict,
    *,
    model_parameters: dict | solara.Reactive[dict] = None,
):
    """Solara component for creating and managing a model instance with user-defined parameters.

    This component allows users to create a model instance with specified parameters and seed.
    It provides an interface for adjusting model parameters and reseeding the model's random
    number generator.

    Args:
        model: A reactive model instance. This is the main model to be created and managed.
        user_params: Parameters for (re-)instantiating a model. Can include user-adjustable parameters and fixed parameters. Defaults to None.
        model_parameters: reactive parameters for reinitializing the model

    Returns:
        solara.component: A Solara component that renders the model creation and management interface.

    Example:
        >>> model = solara.reactive(MyModel())
        >>> model_params = {
        >>>     "param1": {"type": "slider", "value": 10, "min": 0, "max": 100},
        >>>     "param2": {"type": "slider", "value": 5, "min": 1, "max": 10},
        >>> }
        >>> creator = ModelCreator(model, model_params)
        >>> creator

    Notes:
        - The `model_params` argument should be a dictionary where keys are parameter names and values either fixed values
          or are dictionaries containing parameter details such as type, value, min, and max.
        - The `seed` argument ensures reproducibility by setting the initial seed for the model's random number generator.
        - The component provides an interface for adjusting user-defined parameters and reseeding the model.
    """
    if model_parameters is None:
        model_parameters = {}
    model_parameters = solara.use_reactive(model_parameters)

    solara.use_effect(
        lambda: _check_model_params(model.value, user_params),
        [model.value],
    )
    user_adjust_params, fixed_params = split_model_params(user_params)

    # Use solara.use_effect to run the initialization code only once
    solara.use_effect(
        # set model_parameters to the default values for all parameters
        lambda: model_parameters.set(
            {
                **fixed_params,
                **{k: v.get("value") for k, v in user_adjust_params.items()},
            }
        ),
        [],
    )

    @function_logger(__name__)
    def on_change(name, value):
        model_parameters.value = {**model_parameters.value, name: value}

    UserInputs(user_adjust_params, on_change=on_change)


def _check_model_params(model_or_func, model_params):
    """Check if model parameters are valid for the model's initialization function.

    Args:
        model_or_func: Model instance or its initialization function
        model_params: Dictionary of model parameters

    Raises:
        ValueError: If a parameter is not valid for the model's initialization function
    """
    if inspect.isfunction(model_or_func) or inspect.ismethod(model_or_func):
        init_func = model_or_func
        model = None
    else:
        model = model_or_func
        init_func = type(model).__init__

    model_parameters = inspect.signature(init_func).parameters

    has_var_positional = any(
        param.kind == inspect.Parameter.VAR_POSITIONAL
        for param in model_parameters.values()
    )

    if has_var_positional:
        raise ValueError(
            "Mesa's visualization requires the use of keyword arguments to ensure the parameters are passed to Solara correctly. Please ensure all model parameters are of form param=value"
        )

    has_var_keyword = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in model_parameters.values()
    )

    scenario_defaults = {}
    if model is not None and getattr(model, "scenario", None):
        scenario_defaults = _get_scenario_defaults(model.scenario)

    for name in model_parameters:
        if (
            model_parameters[name].default == inspect.Parameter.empty
            and name not in model_params
            and name not in ["self", "kwargs", "args", "scenario"]
            and not has_var_keyword
        ):
            raise ValueError(f"Missing required model parameter: {name}")

    for name in model_params:
        if (
            name not in model_parameters
            and name not in scenario_defaults
            and name not in ["rng", "seed", "simulator"]
            and not has_var_keyword
        ):
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
    new_renderer = type(renderer)(model=model, backend=renderer.backend)

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
        if getattr(renderer, attr, None) is not None:
            value_to_copy = getattr(renderer, attr)
            setattr(new_renderer, attr, value_to_copy)

    return new_renderer


@solara.component
def ShowSteps(model):
    """Display the current step of the model."""
    update_counter.get()
    return solara.Text(f"Time: {model.time}")
