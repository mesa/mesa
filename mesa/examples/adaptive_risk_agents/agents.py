"""Adaptive Risk Agent.

An agent that chooses between safe and risky actions and adapts its
risk preference based on past outcomes.

This example intentionally keeps all decision logic inside `step()`
to highlight current limitations in Mesa's behavior modeling.
"""

from __future__ import annotations

from collections import deque

from mesa import Agent


class AdaptiveRiskAgent(Agent):
    """An agent that adapts its risk-taking behavior over time.

    Attributes
    ----------
    risk_preference : float
        Probability (0-1) of choosing a risky action.
    memory : deque[int]
        Recent outcomes of risky actions (+1 reward, -1 loss).
    """

    def __init__(
        self,
        model,
        *,
        initial_risk_preference: float = 0.5,
        memory_size: int = 10,
    ) -> None:
        super().__init__(model)
        self.risk_preference = initial_risk_preference
        self.memory: deque[int] = deque(maxlen=memory_size)

    def choose_action(self) -> str:
        """Choose between a safe or risky action."""
        if self.model.random.random() < self.risk_preference:
            return "risky"
        return "safe"

    def risky_action(self) -> int:
        """Perform a risky action.

        Returns
        -------
        int
            Outcome of the action (+1 for reward, -1 for loss).
        """
        return 1 if self.model.random.random() < 0.5 else -1

    def safe_action(self) -> int:
        """Perform a safe action.Returns ------- int  Guaranteed neutral outcome."""
        return 0

    def update_risk_preference(self) -> None:
        """Update risk preference based on recent memory."""
        if not self.memory:
            return

        avg_outcome = sum(self.memory) / len(self.memory)

        if avg_outcome < 0:
            self.risk_preference = max(0.0, self.risk_preference - 0.05)
        else:
            self.risk_preference = min(1.0, self.risk_preference + 0.05)

    def step(self) -> None:
        """Execute one decision step.

        NOTE:
        This method intentionally mixes decision-making, action execution,
        memory updates, and learning to demonstrate how behavioral
        complexity accumulates in current Mesa models.
        """
        action = self.choose_action()

        if action == "risky":
            outcome = self.risky_action()
            self.memory.append(outcome)
            self.update_risk_preference()
        else:
            self.safe_action()
