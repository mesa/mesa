"""This method is for dynamically creating new agents (meta-agents).

Meta-agents are defined as agents composed of existing agents.

Meta-agents are created dynamically with a pointer to the model, name of the meta-agent,,
iterable of agents to belong to the new meta-agents, any new functions for the meta-agent,
any new attributes for the meta-agent, whether to retain sub-agent functions,
whether to retain sub-agent attributes.

Examples of meta-agents:
- An autonomous car where the subagents are the wheels, sensors,
battery, computer etc. and the meta-agent is the car itself.
- A company where the subagents are employees, departments, buildings, etc.
- A city where the subagents are people, buildings, streets, etc.

Currently meta-agents are restricted to one parent agent for each subagent/
one meta-agent per subagent.

Goal is to assess usage and expand functionality.

"""

from .backend import MembershipBackend
from .meta_agent import MetaAgent

__all__ = ["MembershipBackend", "MetaAgent"]
