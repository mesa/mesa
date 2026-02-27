"""Core classes for the ActionSpace framework.
This module implements the action validation pipeline:
1. Agent creates an Action (what it wants to do)
2. ActionSpace checks all hard constraints (clip/project violations)
3. ActionSpace computes soft constraint costs 
4. Returns the (possibly modified) action + a report of what changed
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from mesa.agent import Agent

@dataclass
class Action:
    """An intended action the agent wants to perfrom with typed parameters.
    Attributes:
        action_type: Name of the action (e.g., "move", "speak").
        params: Key-value parameters for the action.
    Example:
        action = Action("move", {"speed": 10, "direction": "north"})
    """
    action_type : str
    params: dict[str, Any] = field(default_factory=dict)

    def __repr__(self)->str:
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

    def __init__(self, name: str="") -> None:
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

    

    


    


