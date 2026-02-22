"""Mesa Agent-Based Modeling Framework.

Core Objects: Model, and Agent.
"""

import datetime

import mesa.discrete_space as discrete_space
import mesa.experimental as experimental
import mesa.time as time
from mesa.agent import Agent
from mesa.datacollection import DataCollector
from mesa.model import Model
from mesa.protocols import HasPosition, Locatable, PositionLike

__all__ = [
    "Agent",
    "DataCollector",
    "HasPosition",
    "Locatable",
    "Model",
    "PositionLike",
    "discrete_space",
    "experimental",
    "time",
]

__title__ = "mesa"
__version__ = "4.0.0.dev0"
__license__ = "Apache 2.0"
_this_year = datetime.datetime.now(tz=datetime.UTC).date().year
__copyright__ = f"Copyright {_this_year} Mesa Team"
