"""ActionSpace: experimental module for defining agent capabilities and constraints.

This module provides a framework for constraining what agents can do and at what
cost. Constraints are composable, queryable, and enforce physical/logical limits
on agent actions.

Classes:
    Action: Data object representing an intended agent action.
    Constraint: Abstract base class for user-defined constraints.
    ActionSpace: Container of constraints that validates and projects actions.
    ActionReport: Report of what happened during action validation.

Example:
    >>> from mesa.experimental.action_space import ActionSpace, Action, Constraint
    >>> class MaxSpeed(Constraint):
    ...     def __init__(self, limit):
    ...         super().__init__(name=f"MaxSpeed({limit})")
    ...         self.limit = limit
    ...     def check(self, action, agent):
    ...         if action.action_type != "move":
    ...             return True
    ...         return action.params.get("speed", 0) <= self.limit
    ...     def project(self, action, agent):
    ...         params = dict(action.params)
    ...         params["speed"] = min(params.get("speed", 0), self.limit)
    ...         return Action(action.action_type, params)
"""

from mesa.experimental.action_space.core import (
    Action,
    ActionReport,
    ActionSpace,
    Constraint,
)

__all__ = [
    "Action",
    "ActionReport",
    "ActionSpace",
    "Constraint",
]
