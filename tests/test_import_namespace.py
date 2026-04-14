"""Test if namespsaces importing work better."""


def test_import():
    """This tests the new, simpler Mesa namespace.

    See https://github.com/mesa/mesa/pull/1294.
    """
    import mesa
    from mesa.datacollection import DataCollector

    _ = DataCollector
    _ = mesa.DataCollector
