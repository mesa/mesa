"""Solara related utils."""

import re

import altair as alt
import solara


def _get_altair_major_version() -> int:
    version = getattr(alt, "__version__", "")
    match = re.match(r"^(\d+)", version)
    return int(match.group(1)) if match else 0


@solara.component
def FigureAltair(
    chart,
    on_click=None,
    on_hover=None,
):
    """Render an Altair chart inside a Solara component.

    This component adapts to different Altair versions:
    - For Altair v6 and above, it converts the chart to a VegaLite spec
      and renders it using `solara.widgets.VegaLite.element`.
    - For older Altair versions, it falls back to `solara.FigureAltair`.

    Args:
        chart: Altair chart object.
        on_click: Optional callback function for click events.
        on_hover: Optional callback function for hover events.

    Returns:
        Solara component displaying the Altair chart.
    """
    if _get_altair_major_version() >= 6:
        spec = chart.to_dict() if hasattr(chart, "to_dict") else chart
        return solara.widgets.VegaLite.element(
            spec=spec,
            on_click=on_click,
            listen_to_click=on_click is not None,
            on_hover=on_hover,
            listen_to_hover=on_hover is not None,
        )

    return solara.FigureAltair(chart, on_click=on_click, on_hover=on_hover)


update_counter = solara.reactive(0)


def force_update():  # noqa: D103
    update_counter.value += 1
