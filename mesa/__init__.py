"""Mesa Agent-Based Modeling Framework.

Core Objects: Model, and Agent.
"""

import datetime

import mesa.discrete_space as discrete_space
import mesa.experimental as experimental
import mesa.meta_agents as meta_agents
import mesa.time as time
from mesa.agent import Agent
from mesa.datacollection import DataCollector
from mesa.model import Model

__all__ = [
    "Agent",
    "DataCollector",
    "Model",
    "discrete_space",
    "experimental",
    "meta_agents",
    "time",
]

__title__ = "mesa"
__version__ = "4.0.0a0"
__license__ = "Apache 2.0"
_this_year = datetime.datetime.now(tz=datetime.UTC).date().year
__copyright__ = f"Copyright {_this_year} Mesa Team"
