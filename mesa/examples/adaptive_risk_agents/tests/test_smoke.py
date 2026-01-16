"""Smoke tests for the Adaptive Risk Agents example.

These tests only verify that the example runs without crashing.
They intentionally avoid checking model outcomes or behavior.
"""

from mesa.examples.adaptive_risk_agents.model import AdaptiveRiskModel


def test_model_initializes():
    model = AdaptiveRiskModel(n_agents=10, seed=42)
    assert model is not None


def test_model_steps_without_error():
    model = AdaptiveRiskModel(n_agents=10, seed=42)
    for _ in range(5):
        model.step()
