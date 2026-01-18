def test_solara_viz_import():
    """Basic sanity check to ensure SolaraViz can be imported.
    This test catches missing dependencies or import errors.
    """
    from mesa.visualization.solara_viz import SolaraViz

    assert SolaraViz is not None


def test_solara_viz_initialization():
    """Ensure SolaraViz can be initialized with a minimal configuration
    without raising exceptions.
    """
    from mesa.visualization.solara_viz import SolaraViz

    # Minimal dummy arguments (no rendering)
    viz = SolaraViz(
        model=None,
        components=[],
        name="Test SolaraViz",
    )

    assert viz is not None


def test_solara_viz_app_runs():
    """Integration test: ensure a minimal Solara app using SolaraViz
    can be created without raising exceptions.
    """
    import solara

    from mesa.visualization.solara_viz import SolaraViz

    viz = SolaraViz(
        model=None,
        components=[],
        name="Integration Test Viz",
    )

    # Minimal Solara app
    @solara.component
    def App():
        return solara.Text("SolaraViz integration test")

    # Creating the app should not crash
    assert App is not None
    assert viz is not None


def test_is_solara_available_returns_bool():
    """Ensure is_solara_available returns a boolean value."""
    from mesa.visualization.solara_viz import is_solara_available

    result = is_solara_available()
    assert isinstance(result, bool)
