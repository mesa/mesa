"""Run script for the Adaptive Risk Agents example.

This script runs the model for a fixed number of steps and prints
aggregate statistics to illustrate how agent behavior evolves over time.

Intentionally simple:
- No DataCollector
- No batch_run
- No visualization
"""

from __future__ import annotations

from mesa.examples.adaptive_risk_agents.model import AdaptiveRiskModel


def run_model(*, n_agents: int = 50, steps: int = 100, seed: int | None = None) -> None:
    """Run the AdaptiveRiskModel and print summary statistics."""
    model = AdaptiveRiskModel(n_agents=n_agents, seed=seed)

    for step in range(steps):
        model.step()

        total_risk = 0.0
        count = 0

        for agent in model.agents:
            total_risk += agent.risk_preference
            count += 1

        avg_risk = total_risk / count if count > 0 else 0.0

        if step % 10 == 0:
            print(f"Step {step:3d} | Average risk preference: {avg_risk:.3f}")


if __name__ == "__main__":
    run_model()
