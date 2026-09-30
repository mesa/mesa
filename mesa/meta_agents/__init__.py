"""Compatibility shim - meta-agents now live in mesa.experimental.meta_agents.

This module re-exports the public API so that existing ``from mesa.meta_agents``
imports continue to work.  New code should import from
``mesa.experimental.meta_agents`` instead.
"""

from mesa.experimental.meta_agents import MembershipEdge, MembershipView, MetaAgents

__all__ = [
    "MembershipEdge",
    "MembershipView",
    "MetaAgents",
]
