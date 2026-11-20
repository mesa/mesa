"""tests for matplotlib components."""

import solara
from matplotlib.ticker import MaxNLocator, MultipleLocator

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.visualization.components.matplotlib_components import PlotMatplotlib


def test_plot_matplotlib_post_process_can_change_x_axis():
    """post_process runs after the default x label and ticks, so it can change them."""
    model = Model(rng=42)
    model.datacollector = DataCollector(model_reporters={"value": lambda m: 1})
    for _ in range(3):
        model.datacollector.collect(model)

    axes = []
    every_five_steps = MultipleLocator(5)

    def keep_x_axis(ax):
        axes.append(ax)

    def change_x_axis(ax):
        axes.append(ax)
        ax.set_xlabel("Time Step")
        ax.xaxis.set_major_locator(every_five_steps)

    for post_process in (keep_x_axis, change_x_axis):
        solara.render(
            PlotMatplotlib(model, "value", post_process=post_process),
            handle_error=False,
        )

    default_ax, changed_ax = axes
    # the defaults still apply when post_process leaves the x axis alone;
    # exact type, because matplotlib's default AutoLocator subclasses MaxNLocator
    assert default_ax.get_xlabel() == "Step"
    assert type(default_ax.xaxis.get_major_locator()) is MaxNLocator
    # x axis changes made in post_process are kept
    assert changed_ax.get_xlabel() == "Time Step"
    assert changed_ax.xaxis.get_major_locator() is every_five_steps
