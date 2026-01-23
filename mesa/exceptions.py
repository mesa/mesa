"""Mesa-specific exception hierarchy."""


class MesaException(Exception):  # noqa: N818
    """Base class for all Mesa-specific exceptions."""


class CellFullException(MesaException):
    """Raised when attempting to add an agent to a cell with no available capacity."""

    def __init__(self, coordinate):
        """Initialize the exception.

        Args:
            coordinate: The coordinate tuple of the full cell.
        """
        self.coordinate = coordinate
        super().__init__(f"Cell at coordinate {coordinate} is full.")
