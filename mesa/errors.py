"""Custom exceptions for Mesa."""


class MesaError(Exception):
    """Base class for all Mesa exceptions."""


class CellFullError(MesaError):
    """Raised when a cell is full."""

    def __init__(self):
        """Initialize the exception."""
        super().__init__("ERROR: Cell is full")
