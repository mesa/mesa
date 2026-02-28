"""Core classes for the ActionSpace framework.

This module implements the action validation pipeline:
1. Agent creates an Action (what it wants to do)
2. ActionSpace checks all hard constraints (clip/project violations)
3. ActionSpace computes soft constraint costs
4. Returns the (possibly modified) action + a report of what changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mesa.agent import Agent


@dataclass
class Action:
    """An intended action the agent wants to perform with typed parameters.

    Attributes:
        action_type: Name of the action (e.g., "move", "speak").
        params: Key-value parameters for the action.

    Example:
        action = Action("move", {"speed": 10, "direction": "north"})

    """

    action_type: str
    params: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Return a string representation of the action."""
        return f"Action({self.action_type!r}, {self.params})"

    def copy(self) -> Action:
        """Return a shallow copy with an independent params dict(Used for reporting)."""
        return Action(self.action_type, dict(self.params))


@dataclass
class ActionReport:
    """Report from action validation.

    Attributes:
        original: The action the agent intended.
        result: The action after validation (may be modified).
        was_modified: Whether any constraint changed the action.
        reasons: Human-readable explanations for each modification.
        costs: Total resource costs of the final action.
    """

    original: Action
    result: Action
    was_modified: bool
    reasons: list[str] = field(default_factory=list)
    costs: dict[str, float] = field(default_factory=dict)


class Constraint:
    """Base class for constraints. Subclass to define limits.

    Two kinds:
        Hard — Override ``check()`` + ``project()``. If check fails,
        project clips the action to the nearest valid point.
        Soft — Override ``cost()``. Returns resource costs. If the agent
        can't afford it, numeric params are scaled down proportionally.
    By default all methods are permissive (check passes, zero cost).

    Attributes:
        name: Human-readable name for reports and debugging.
    """

    def __init__(self, name: str = "") -> None:
        """Initialize constraint.

        Args:
            name: Display name. Defaults to class name.
        """
        self.name = name or self.__class__.__name__

    def check(self, action: Action, agent: Agent) -> bool:
        """Return True if action satisfies this constraint.

        The default implementation is permissive and always returns True.
        Subclasses should override this method only if they impose a
        Hard feasibility constraint. Constraints that only compute
        costs typically do not override this method.

        Args:
            action: The action to check.
            agent: The agent attempting it.
        """
        return True

    def project(self, action: Action, agent: Agent) -> Action:
        """Clip an invalid action to the nearest valid one.

        Only called when ``check()`` returns False in Hard constraints.

        Args:
            action: The invalid action.
            agent: The agent attempting it.
        """
        return action

    def cost(self, action: Action, agent: Agent) -> dict[str, float]:
        """Return resource costs for this action.

        Args:
            action: The action to cost.
            agent: The agent attempting it.

        Returns:
            Dict of resource name → amount consumed. Empty = free.
        """
        return {}

    def describe(self, agent: Agent) -> str:
        """Human-readable description of this constraint's current state.

        Args:
            agent: The agent to describe for.
        """
        return self.name

    def __repr__(self) -> str:
        """Return string representation."""
        return f"{self.__class__.__name__}(name={self.name!r})"


class ActionSpace:
    """Collection of constraints defining what an agent can do.

    Validates actions through a two-stage pipeline:
        1. Hard: ``check()`` each constraint → ``project()`` if violated.
        2. Soft: sum all ``cost()`` → scale if agent can't afford.
    """

    def __init__(self) -> None:
        """Initialize an empty ActionSpace."""
        self._constraints: list[Constraint] = []

    @property
    def constraints(self) -> list[Constraint]:
        """Return a copy of the constraint list."""
        return list(self._constraints)

    def add(self, constraint: Constraint) -> None:
        """Add a constraint.

        Args:
            constraint: Must be a Constraint instance.

        Raises:
            TypeError: If not a Constraint.
        """
        if not isinstance(constraint, Constraint):
            message = f"Expected Constraint, got {type(constraint).__name__}"
            raise TypeError(message)
        self._constraints.append(constraint)

    def remove(self, constraint: Constraint) -> None:
        """Remove a specific constraint instance.

        Raises:
            ValueError: If not found.
        """
        self._constraints.remove(constraint)

    def remove_by_type(self, constraint_type: type) -> int:
        """Remove all constraints of a given type. Returns count removed.

        Used to remove classes of a certain type quickly in a single command.

        Return: It returns the number of constraints that were removed
        """
        before = len(self._constraints)

        self._constraints = [
            c for c in self._constraints if not isinstance(c, constraint_type)
        ]
        return before - len(self._constraints)

    def clear(self) -> None:
        """Removes all constraints in a single command."""
        self._constraints.clear()

    def validate(self, action: Action, agent: Agent) -> tuple[Action, ActionReport]:
        """Validate an action through all constraints.

        Pipeline:
            1. Hard constraints: check → project if violated
            2. Soft constraints: compute costs → scale if over budget
        Args:
            action: The intended action.
            agent: The agent attempting the action.

        Returns:
            A tuple of (validated_action, report).
            The validated action may have been modified by constraints.
        """
        original = action.copy()
        reasons: list[str] = []

        # Hard constraints (check + project)
        for constraint in self._constraints:
            if not constraint.check(action, agent):
                action = constraint.project(action, agent)
                reasons.append(f"{constraint.name}: projected to feasible range")

        # Soft constraints (compute total cost)
        total_cost: dict[str, float] = {}
        for constraint in self._constraints:
            for resource, amount in constraint.cost(action, agent).items():
                total_cost[resource] = total_cost.get(resource, 0.0) + amount

        # Check affordability and scale if needed
        for resource, needed in total_cost.items():
            available = getattr(agent, resource, None)

            if available is not None and needed > 0 and needed > available:
                scale = available / needed

                for key, val in action.params.items():
                    if isinstance(val, int | float):
                        action.params[key] = val * scale
                reasons.append(
                    f"Scaled to {scale:.0%} — insufficient {resource} "
                    f"(had {available:.1f}, needed {needed:.1f})"
                )
                # Recalculate cost after scaling
                total_cost = self.get_cost(action, agent)
                break

        was_modified = len(reasons) > 0
        return action, ActionReport(
            original=original,
            result=action,
            was_modified=was_modified,
            reasons=reasons,
            costs=total_cost,
        )

    def is_feasible(self, action: Action, agent: Agent) -> bool:
        """Check if an action passes all hard constraints without modification.

        Args:
            action: The action to check.
            agent: The agent attempting the action.

        Returns:
            True if all hard constraints pass.
        """
        return all(c.check(action, agent) for c in self._constraints)

    def get_cost(self, action: Action, agent: Agent) -> dict[str, float]:
        """Compute total cost of an action across all constraints.

        Args:
            action: The action to compute costs for.
            agent: The agent attempting the action.

        Returns:
            Dict of total resource costs.
        """
        total: dict[str, float] = {}
        for constraint in self._constraints:
            for resource, amount in constraint.cost(action, agent).items():
                total[resource] = total.get(resource, 0.0) + amount
        return total

    def describe_all(self, agent: Agent) -> list[str]:
        """Get human-readable descriptions of all constraints.

        Useful for injecting into LLM prompts or debugging.

        Args:
            agent: The agent to describe constraints for.

        Returns:
            List of description strings, one per constraint.
        """
        return [c.describe(agent) for c in self._constraints]

    def __len__(self) -> int:
        """Return the number of constraints."""
        return len(self._constraints)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ActionSpace(constraints={len(self._constraints)})"
