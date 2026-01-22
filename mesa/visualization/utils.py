"""Solara related utils."""

import solara

update_counter = solara.reactive(0)


def force_update():  # noqa: D103
    update_counter.value += 1
