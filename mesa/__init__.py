"""Mesa Agent-Based Modeling Framework.

Core Objects: Model, and Agent.
"""

import datetime

from mesa.agent import Agent
from mesa.datacollection import DataCollector
from mesa.model import Model

__all__ = [
    "Agent",
    "DataCollector",
    "Model",
    "discrete_space",
    "experimental",
    "time",
]


__title__ = "mesa"
__version__ = "4.0.0a0"
__license__ = "Apache 2.0"
_this_year = datetime.datetime.now(tz=datetime.UTC).date().year
__copyright__ = f"Copyright {_this_year} Mesa Team"

# Lazy import to avoid importing optional dependencies at package import time


def __getattr__(name):
    if name == "discrete_space":
        import mesa.discrete_space as discrete_space

        return discrete_space
    elif name == "experimental":
        import mesa.experimental as experimental

        return experimental
    elif name == "time":
        import mesa.time as time

        return time
    raise AttributeError(f"module 'mesa' has no attribute '{name}'")
