from __future__ import annotations

from mesa import Model
from mesa.examples.adaptive_risk_agents.agents import AdaptiveRiskAgent


class AdaptiveRiskModel(Model):
    """A simple model running adaptive risk-taking agents."""

    def __init__(self, n_agents: int = 50, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)

        # Create agents — Mesa will register them automatically
        for _ in range(n_agents):
            AdaptiveRiskAgent(self)

    def step(self) -> None:
        """Advance the model by one step."""
        self.agents.shuffle_do("step")
