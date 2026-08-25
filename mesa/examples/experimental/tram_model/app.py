import solara
from matplotlib.figure import Figure

from mesa.examples.experimental.tram_model.model import TramScenario, TransitSystem
from mesa.examples.experimental.tram_model.utils import (
    POSITION_COLOR,
    SCALE,
    SPEED_COLOR,
    TITLE_SIZE,
    draw_track_view,
    tram_portrayal,
)
from mesa.visualization import Slider, SolaraViz, make_plot_component
from mesa.visualization.utils import update_counter

model_params = {
    "rng": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "n_stations": Slider("Number of stations", 20, 2, 40, 1),
    "station_spacing": Slider("Station spacing (m)", 200, 100, 1000, 50),
    "cruise_speed": Slider("Cruise speed (m/s)", 15, 5, 30, 1),
    "acceleration_rate": Slider("Acceleration (m/s^2)", 2.0, 0.5, 5.0, 0.5),
    "deceleration_rate": Slider("Deceleration (m/s^2)", 3.0, 0.5, 5.0, 0.5),
    "dwell_time": Slider("Dwell time (s)", 5, 0, 30, 1),
}


@solara.component
def plot_track(model):
    """Draw the route as a line of stations with the tram at its current position."""
    update_counter.get()
    tram = model.tram
    style = tram_portrayal(tram)

    fig = Figure(figsize=(7 * SCALE, 2.1 * SCALE))
    ax = fig.subplots()

    draw_track_view(ax, model, style)
    ax.set_title(
        f"t={model.time:.1f}s   |   {tram.position:.1f} m   |   "
        f"{tram.speed:.1f} m/s   |   {style['phase']}",
        color=style["color"],
        fontsize=TITLE_SIZE,
    )
    fig.tight_layout()

    solara.FigureMatplotlib(fig)


SpeedPlot = make_plot_component({"Speed": SPEED_COLOR})
PositionPlot = make_plot_component({"Position": POSITION_COLOR})


model = TransitSystem(scenario=TramScenario())

page = SolaraViz(
    model,
    components=[SpeedPlot, PositionPlot, plot_track],
    model_params=model_params,
    name="Tram Route Model",
)
page  # noqa
