"""Core event management functionality for Mesa's discrete event simulation system.

This module provides the foundational data structures and classes needed for event-based
simulation in Mesa. The EventList class is a priority queue implementation that maintains
simulation events in chronological order while respecting event priorities. Key features:

- Priority-based event ordering
- Weak references to prevent memory leaks from canceled events
- Efficient event insertion and removal using a heap queue
- Support for event cancellation without breaking the heap structure

The module contains three main components:
- Priority: An enumeration defining event priority levels (HIGH, DEFAULT, LOW)
- SimulationEvent: A class representing individual events with timing and execution details
- EventList: A heap-based priority queue managing the chronological ordering of events

The implementation supports both pure discrete event simulation and hybrid approaches
combining agent-based modeling with event scheduling.
"""

from .eventlist import Priority, SimulationEvent
from .simulator import ABMSimulator, DEVSimulator, Simulator

__all__ = ["ABMSimulator", "DEVSimulator", "Priority", "SimulationEvent", "Simulator"]
