"""Drawing helpers for the Tram Route Model visualization.

Everything here is presentation: the palette, the type scale, and the
matplotlib code that turns a tram into a picture of a tram. app.py wires
these into Solara components.
"""

from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.transforms import blended_transform_factory

RAIL_COLOR = "#5c6470"
SLEEPER_COLOR = "#b9a58a"
PLATFORM_COLOR = "#c8cdd4"
WINDOW_COLOR = "#e8f1f8"
HEADLIGHT_COLOR = "#ffd54f"
TRIM_COLOR = "#1f2933"

# The figure is drawn at double size, so type and marks are scaled to match.
SCALE = 2
TITLE_SIZE = 11 * SCALE
LABEL_SIZE = 10 * SCALE
TICK_SIZE = 9 * SCALE
SPEED_COLOR = "#2a78d6"
POSITION_COLOR = "#eb6834"

PHASE_COLORS = {
    "accelerating": "#008300",
    "coasting": "#4a3aa7",
    "braking": "#e34948",
    "dwelling at station": "#a15c00",
    "route complete": "#6b7280",
}


def phase(tram):
    """Name what the tram is doing right now."""
    if tram.route_complete:
        return "route complete"
    if tram.acceleration > 0:
        return "accelerating"
    if tram.acceleration < 0:
        return "braking"
    if tram.speed > 0:
        return "coasting"
    return "dwelling at station"


def tram_portrayal(tram):
    """Describe how to draw the tram, given what it is doing right now."""
    # A plain dict rather than an AgentPortrayalStyle: that describes a scatter
    # marker (x/y in a space, marker, size) for SpaceRenderer to draw, but this
    # model has no space and the tram is a composite of patches. It also has no
    # field for the phase name, which would have to ride in `tooltip`.
    current = phase(tram)
    return {
        "phase": current,
        "position": tram.position,
        "color": PHASE_COLORS[current],
        "edgecolor": TRIM_COLOR,
        "linewidth": 1.1,
        "zorder": 4,
    }


def draw_track(ax, route, span):
    """Draw the ballast, rail and station platforms along the route."""
    sleeper = span / 90
    x = route[0]
    while x <= route[-1]:
        ax.plot([x, x], [0.19, 0.26], color=SLEEPER_COLOR, linewidth=2, zorder=1)
        x += sleeper

    ax.plot(
        [route[0], route[-1]], [0.27, 0.27], color=RAIL_COLOR, linewidth=2.5, zorder=2
    )

    # Sized off the gap between stops so platforms never merge into one band.
    spacing = span / max(len(route) - 1, 1)
    platform = min(span * 0.018, spacing * 0.3)
    for station in route:
        ax.add_patch(
            Rectangle(
                (station - platform / 2, 0.28),
                platform,
                0.06,
                facecolor=PLATFORM_COLOR,
                edgecolor="#9aa3ad",
                linewidth=0.5,
                zorder=2,
            )
        )


def draw_tram(ax, style, route, heading):
    """Draw the tram body, windows and wheels centred on its position.

    The body is drawn with x in axes fraction and y in data coordinates, so it
    keeps the same shape however long the route is.

    Args:
        ax: Axes to draw on
        style: Mapping from tram_portrayal with the tram's position and appearance
        route: Ordered station positions, used to place the tram along the axis
        heading: Signed speed, which end of the tram the headlight goes on
    """
    span = route[-1] - route[0]
    margin = span * 0.04
    low, high = route[0] - margin, route[-1] + margin
    centre = (style["position"] - low) / (high - low)

    transform = blended_transform_factory(ax.transAxes, ax.transData)
    width, base, height = 0.07, 0.30, 0.26
    left = centre - width / 2

    ax.add_patch(
        FancyBboxPatch(
            (left, base),
            width,
            height,
            boxstyle="round,pad=0,rounding_size=0.012",
            facecolor=style["color"],
            edgecolor=style["edgecolor"],
            linewidth=style["linewidth"],
            transform=transform,
            zorder=style["zorder"],
        )
    )
    # Darker skirt so the body reads as a vehicle rather than a bar.
    ax.add_patch(
        Rectangle(
            (left, base),
            width,
            height * 0.22,
            facecolor=style["edgecolor"],
            alpha=0.25,
            edgecolor="none",
            transform=transform,
            zorder=5,
        )
    )

    window_y, window_h = base + height * 0.45, height * 0.34
    for i in range(4):
        ax.add_patch(
            Rectangle(
                (left + width * (0.1 + 0.21 * i), window_y),
                width * 0.13,
                window_h,
                facecolor=WINDOW_COLOR,
                edgecolor="none",
                transform=transform,
                zorder=5,
            )
        )

    # Headlight at the leading end.
    nose = left + width * (0.94 if heading >= 0 else 0.06)
    ax.plot(
        [nose],
        [base + height * 0.3],
        marker="o",
        markersize=3,
        color=HEADLIGHT_COLOR,
        transform=transform,
        zorder=6,
    )

    ax.plot(
        [left + width * 0.25, left + width * 0.75],
        [base - 0.015, base - 0.015],
        linestyle="none",
        marker="o",
        markersize=5,
        color=style["edgecolor"],
        transform=transform,
        zorder=3,
    )

    # Pantograph arm and contact bar reaching up to the overhead line.
    ax.plot(
        [centre - width * 0.12, centre + width * 0.1],
        [base + height, 0.74],
        color=style["edgecolor"],
        linewidth=style["linewidth"],
        transform=transform,
        zorder=4,
    )
    ax.plot(
        [centre + width * 0.02, centre + width * 0.18],
        [0.74, 0.74],
        color=style["edgecolor"],
        linewidth=1.6,
        transform=transform,
        zorder=4,
    )


def draw_track_view(ax, model, style):
    """Set up the track axis and draw the route with the tram on it."""
    route = model.route
    span = route[-1] - route[0]
    margin = span * 0.04

    # Overhead line the pantograph runs against.
    ax.plot(
        [route[0] - margin, route[-1] + margin],
        [0.75, 0.75],
        color=RAIL_COLOR,
        linewidth=1,
        zorder=1,
    )

    ax.set_xlim(route[0] - margin, route[-1] + margin)
    ax.set_ylim(0.1, 0.95)

    draw_track(ax, route, span)
    draw_tram(ax, style, route, model.tram.speed)

    ax.set_yticks([])
    ax.set_xlabel("position along route (m)", fontsize=LABEL_SIZE)
    ax.tick_params(axis="x", labelsize=TICK_SIZE)
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
