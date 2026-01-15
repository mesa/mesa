"""Utility functions and classes for Mesa's signals implementation.

This module provides helper functionality used by Mesa's reactive programming system:

- AttributeDict: A dictionary subclass that allows attribute-style access to its contents
- create_weakref: Helper function to properly create weak references to different types

These utilities support the core signals implementation by providing reference
management and convenient data structures used throughout the reactive system.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

__all__ = [
    "Message",
    "create_weakref",
]

if TYPE_CHECKING:
    from mesa.experimental.mesa_signals import ListSignalType, SignalType


@dataclass(frozen=True, slots=True)
class Message:
    """A message class containing information about a signal change."""

    name: str
    old: Any
    new: Any
    owner: Any
    signal_type: SignalType | ListSignalType
    additional_kwargs: dict


def create_weakref(item, callback=None):
    """Helper function to create a correct weakref for any item."""
    if hasattr(item, "__self__"):
        ref = weakref.WeakMethod(item, callback)
    else:
        ref = weakref.ref(item, callback)
    return ref
