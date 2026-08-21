"""Meta-agents: agents composed of other agents."""

from .meta_agent import evaluate_combination, find_combinations
from .meta_agents_api import MembershipEdge, MembershipView, MetaAgents

__all__ = [
    "MembershipEdge",
    "MembershipView",
    "MetaAgents",
    "evaluate_combination",
    "find_combinations",
]
