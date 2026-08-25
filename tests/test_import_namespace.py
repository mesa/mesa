"""Test if namespsaces importing work better."""

import pytest


def test_import():
    """This tests the new, simpler Mesa namespace.

    See https://github.com/mesa/mesa/pull/1294.
    """
    import mesa  # noqa: PLC0415
    from mesa.datacollection import DataCollector  # noqa: PLC0415

    _ = DataCollector
    _ = mesa.DataCollector


def test_simulator_classes_removed():
    """Simulator, ABMSimulator, and DEVSimulator must stay removed.

    They were deprecated in Mesa 3.5.0 and removed in Mesa 4.0 along with the
    entire ``mesa.experimental.devs`` package (#3132, #3277, #3530). This pins
    the exact error a user hits so it can't silently regress, and so it stays
    consistent with the migration guide.
    """
    import mesa  # noqa: PLC0415

    with pytest.raises(ModuleNotFoundError):
        import mesa.experimental.devs  # noqa: PLC0415

    with pytest.raises(ModuleNotFoundError):
        from mesa.experimental.devs.simulator import (  # noqa: F401, PLC0415
            ABMSimulator,
        )

    with pytest.raises(ModuleNotFoundError):
        from mesa.experimental.devs.simulator import (  # noqa: F401, PLC0415
            DEVSimulator,
        )

    for name in ("Simulator", "ABMSimulator", "DEVSimulator"):
        assert not hasattr(mesa, name)


def test_simulator_replacement_api_present():
    """The scheduling API that replaced the Simulator classes must exist.

    ``Model.run_for``/``run_until``/``schedule_event``/``schedule_recurring``
    and the ``mesa.time`` primitives are what the migration guide points
    users to; this fails loudly if any of them are renamed or dropped.
    """
    import mesa  # noqa: PLC0415

    for method in (
        "run_for",
        "run_until",
        "schedule_event",
        "schedule_recurring",
    ):
        assert hasattr(mesa.Model, method)

    for name in ("Event", "EventGenerator", "EventList", "Priority", "Schedule"):
        assert hasattr(mesa.time, name)
