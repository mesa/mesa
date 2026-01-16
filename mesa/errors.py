"""
Mesa-specific exception hierarchy.

This module defines a minimal base exception class for Mesa along with a small
set of domain-specific subclasses. All Mesa exceptions inherit from MesaError,
allowing users to catch framework-specific errors explicitly.

This file is intentionally minimal and non-breaking. Future expansions (error
codes, metadata, or more subclasses) can be added incrementally.
"""

from __future__ import annotations

__all__ = ["MesaError", "ModelError", "AgentError", "SpaceError"]

class MesaError(Exception):
    """
    Base class for all Mesa-specific exceptions.

    Metadata such as Mesa version is stored for debugging purposes, without
    changing the exception message.
    """

    def __init__(self, message: str):
        super().__init__(message)
        try:
            from importlib.metadata import version, PackageNotFoundError
            self.mesa_version: str = version("mesa")
        except (ImportError, PackageNotFoundError):
            self.mesa_version: str = "unknown"


# --- Core domain exceptions -------------------------------------------------

class ModelError(MesaError):
    """Errors related to model configuration, initialization, or execution."""
    pass

class AgentError(MesaError):
    """Errors related to agent lifecycle or behavior."""
    pass

class SpaceError(MesaError):
    """Errors related to spaces, grids, movement, or spatial constraints."""
    pass
