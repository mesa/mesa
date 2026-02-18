"""Test if namespsaces importing work better."""


def test_import():
    """This tests the new, simpler Mesa namespace.

    See https://github.com/mesa/mesa/pull/1294.
    """
    import mesa  # noqa: PLC0415
    from mesa.datacollection import DataCollector  # noqa: PLC0415

    _ = DataCollector
    _ = mesa.DataCollector
