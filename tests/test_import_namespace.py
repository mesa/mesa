"""Test if namespsaces importing work better."""

import pytest

from mesa import meta_agents


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


def test_meta_agents():
    """Meta-agents live at mesa.meta_agents, not mesa.experimental.meta_agents."""
    import mesa  # noqa: PLC0415

    with pytest.raises(ModuleNotFoundError):
        import mesa.experimental.meta_agents  # noqa: PLC0415

    assert hasattr(mesa, "meta_agents")
    from mesa.meta_agents import MetaAgents  # noqa: PLC0415

    assert MetaAgents is mesa.meta_agents.MetaAgents

def test_lazy_submodule_getattr_and_dir():
    """Excercise the lazy __getattr__/__dir__ added for #2343.
    mesa/__init__.py and mesa/experimental/__init__.py lazy-load their
    optional-dependency submodules via a module-level __getattr__ (PEP 562)
    instead of importing them eagerly. This excercises that __getattr__/__dir__
    directly, independent of whichever other test happensto import a given
    submodule a different way first.
    """
    import mesa
    import mesa.discrete_space
    import mesa.experimental
    import mesa.experimental.actions
    import mesa.experimental.continuous_space
    import mesa meta_agents
    import mesa.time

    for name, expected in(
        ("discrete_space", mesa.discrete_space),
        ("experimental", mesa.experimental),
        ("meta_agents", mesa.meta_agents),
        ("time", mesa.time),
    ):
        assert name in mesa.__dir__()
        assert mesa.__getattr__(name) is expected

    for name, expected in(
        ("actions", mesa.experimental.actions),
        ("continuous_space", mesa.experimental.continuous_space),
    ):
        assert name in dir(mesa.experimental)
        assert mesa.experimental.__getattr__(name) is expected

    with pytest.raises(AttributeError):
        mesa.__getattr__("not_a_real_submodule")
    with pytest.raises(AttributeError):
        mesa.experimental.__getattr__("not_a_real_submodule")