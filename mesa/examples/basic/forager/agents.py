"""Forager example: agents and their actions."""

from mesa.experimental.actions import Action, ActionAgent


def _apply_food(agent, completion):
    """on_effect callback: award food proportional to how much foraging was done."""
    agent.food_collected += agent.food_per_forage * completion


class ForagerAgent(ActionAgent):
    """An agent that forages for food and flees from threats.

    Demonstrates:
    - reschedule_on_interrupt="remainder": interrupted foraging resumes from
      where it left off after the threat passes.
    - request_action with priority: a high-priority flee action preempts a
      low-priority forage action.

    Attributes:
        food_collected: Total food earned (scaled by partial completion).
        food_per_forage: Maximum food earned from one full forage cycle.
        forage_duration: Sim-time units needed to complete one forage.
        flee_duration: Sim-time units needed to flee a threat.
        threats_faced: Number of times this agent was threatened.
    """

    def __init__(
        self,
        model,
        food_per_forage: float = 1.0,
        forage_duration: float = 5.0,
        flee_duration: float = 1.0,
    ):
        """Create a ForagerAgent.

        Args:
            model: The model instance.
            food_per_forage: Maximum food earned per completed forage.
            forage_duration: Sim-time units to complete one forage.
            flee_duration: Sim-time units to flee a threat.
        """
        super().__init__(model)
        self.food_per_forage = food_per_forage
        self.forage_duration = forage_duration
        self.flee_duration = flee_duration

        self.food_collected: float = 0.0
        self.threats_faced: int = 0

    def _make_forage_action(self) -> Action:
        """Build a new forage Action for this agent."""
        return Action(
            name="forage",
            duration=self.forage_duration,
            priority=1.0,
            on_effect=_apply_food,
            reschedule_on_interrupt="remainder",  # resume from where we left off
        )

    def _make_flee_action(self) -> Action:
        """Build a flee Action for this agent."""
        return Action(
            name="flee",
            duration=self.flee_duration,
            priority=10.0,  # higher than forage → always preempts
        )

    def step(self) -> None:
        """Each step: react to threats or start foraging if free."""
        threatened = self.model.random.random() < self.model.threat_prob

        if threatened:
            self.threats_faced += 1
            # request_action: if flee.priority(10) > current.priority(1), preempt.
            # The interrupted forage is re-queued (reschedule_on_interrupt="remainder")
            # and will auto-resume once fleeing finishes.
            self.request_action(self._make_flee_action())

        elif not self.is_busy:
            # No threat and no current action: start foraging.
            self.start_action(self._make_forage_action())
