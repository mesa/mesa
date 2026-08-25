"""Utility functions and classes for Mesa's signals implementation.

This module provides helper functionality used by Mesa's reactive programming system:

- Message: a dataclass containing information about a signal change
- SignalType: root enum defining the types of signals that can be emitted
- create_weakref: Helper function to properly create weak references to different types
- _AllSentinel class for subscribing to all signals or all observables

These utilities support the core signals implementation by providing reference
management and convenient data structures used throughout the reactive system.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

__all__ = ["ALL", "Message", "SignalType", "create_weakref"]


@dataclass(frozen=True, slots=True)
class Message:
    """A message class containing information about a signal change."""

    name: str
    owner: Any
    signal_type: SignalType
    additional_kwargs: dict


class SignalType(StrEnum):
    """Root class for all signal type enums."""


def create_weakref(item, callback=None):
    """Helper function to create a correct weakref for any item."""
    if hasattr(item, "__self__"):
        ref = weakref.WeakMethod(item, callback)
    else:
        ref = weakref.ref(item, callback)
    return ref


class _AllSentinel:
    """Sentinel for subscribing to all signals or all observables."""

    __slots__ = ()
    _instance: _AllSentinel | None = None

    def __new__(cls) -> _AllSentinel:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "ALL"

    def __str__(self) -> str:
        return "all"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _AllSentinel)

    def __hash__(self) -> int:
        return hash(_AllSentinel)

    def __reduce__(self) -> tuple:
        # Ensure unpickling returns the singleton
        return (_AllSentinel, ())


ALL = _AllSentinel()
