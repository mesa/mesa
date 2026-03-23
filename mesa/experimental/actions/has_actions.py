"""HasActions mixin for agent-level action lifecycle hooks.

This mixin provides agents with hooks that fire after action lifecycle events,
giving them a centralized place for "what to do next" decision logic.

Example::

    from mesa import Agent
    from mesa.experimental.actions import Action, HasActions

    class Forage(Action):
        def on_complete(self):
            self.agent.energy += 30  # Just the effect

    class Rest(Action):
        def on_complete(self):
            self.agent.energy = min(100, self.agent.energy + 20)

    class Flee(Action):
        def on_complete(self):
            self.agent.move_away_from_predator()

    class Sheep(Agent, HasActions):
        def on_action_complete(self, action):
            self.decide_next()

        def on_action_interrupt(self, action, progress):
            self.decide_next()

        def decide_next(self):
            if self.sees_predator():
                self.start_action(Flee(self))
            elif self.energy < 30:
                self.start_action(Rest(self))
            else:
                self.start_action(Forage(self))

This separates action effects (defined in Action subclasses) from agent
decision logic (centralized in the agent class).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mesa.experimental.actions import Action


class HasActions:
    """Mixin providing agent-level action lifecycle hooks.

    Agents that inherit from this mixin will have their hooks called
    after the corresponding Action hooks fire. This allows centralizing
    "what to do next" logic in the agent rather than duplicating it
    across every Action subclass.

    All hooks default to doing nothing (pass), so you only need to
    override the ones you care about.

    Notes:
        - Hooks fire AFTER the Action's own hooks (e.g., on_action_complete
          fires after Action.on_complete).
        - Zero-duration (instantaneous) actions skip on_action_start to
          avoid double-firing in the same instant.
        - These hooks are purely additive — you can still put all logic
          in Action.on_complete if you prefer.
    """

    def on_action_start(self, action: Action) -> None:
        """Called after an action starts or resumes.

        Fires after Action.on_start() (for first starts) or
        Action.on_resume() (for resumptions).

        Args:
            action: The action that just started or resumed.

        Notes:
            NOT called for zero-duration (instantaneous) actions, which
            go directly to on_action_complete instead.
        """

    def on_action_complete(self, action: Action) -> None:
        """Called after an action completes normally.

        Fires after Action.on_complete(). This is the ideal place to
        implement "what to do next" logic.

        Args:
            action: The action that just completed.
        """

    def on_action_interrupt(self, action: Action, progress: float) -> None:
        """Called after an action is interrupted or cancelled.

        Fires after Action.on_interrupt(progress). Use this to handle
        partial completion or decide on a new action.

        Args:
            action: The action that was interrupted.
            progress: Fraction of duration completed (0.0 to 1.0).

        Notes:
            Also fires when cancel() is called. If you need to distinguish
            interruption from cancellation, check action.interruptible.
        """
