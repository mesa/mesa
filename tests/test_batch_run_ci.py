"""CI smoke test for batch_run.

Ensures batch_run executes example models without crashing.
"""

from mesa.batchrunner import batch_run
from mesa.examples.basic.boltzmann_wealth_model.model import BoltzmannWealth


def test_batch_run_executes_boltzmann_wealth():
    """Ensure batch_run executes the Boltzmann Wealth example without errors."""
    results = batch_run(
        BoltzmannWealth,
        parameters={"n": 10},
        rng=range(2),
        max_steps=5,
    )

    assert results
    assert len(results) > 0
