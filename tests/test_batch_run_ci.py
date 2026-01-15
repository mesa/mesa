"""CI smoke test for batch_run.

Ensures batch_run executes example models without crashing.
"""

from mesa.batchrunner import batch_run
from mesa.examples.basic.boltzmann_wealth_model.model import BoltzmannWealth


def test_batch_run_executes_boltzmann_wealth():
    results = batch_run(
        BoltzmannWealth,
        parameters={
            "n": [10],
            "width": [10],
            "height": [10],
        },
        iterations=1,
        max_steps=5,
    )

    assert results is not None
    assert len(results) > 0
