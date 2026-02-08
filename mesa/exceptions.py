"""Mesa-specific exception hierarchy."""


class MesaException(Exception):  # noqa: N818
    """Base class for all Mesa-specific exceptions."""


class SpaceException(MesaException):
    """Base exception for errors in the discrete_space module."""


class CellFullException(SpaceException):
    """Raised when attempting to add an agent to a cell with no available capacity."""

    def __init__(self, coordinate):
        """Initialize the exception.

        Args:
            coordinate: The coordinate tuple of the full cell.
        """
        self.coordinate = coordinate
        super().__init__(f"Cell at coordinate {coordinate} is full.")


class AgentMissingException(MesaException):
    """Raised when attempting to remove an agent that is not in the cell."""

    def __init__(self, agent, coordinate):
        """Initialize the exception.

        Args:
            agent: The agent instance that was expected.
            coordinate: The coordinate tuple of the cell.
        """
        self.agent = agent
        self.coordinate = coordinate
        super().__init__(f"Agent {agent.unique_id} is not in cell {coordinate}.")


class CellMissingException(SpaceException):
    """Raised when attempting to access or remove a cell that does not exist."""

    def __init__(self, coordinate):
        """Initialize the exception.

        Args:
            coordinate: The coordinate tuple of the missing cell.
        """
        self.coordinate = coordinate
        super().__init__(f"Cell at coordinate {coordinate} does not exist.")


class ConnectionMissingException(SpaceException):
    """Raised when attempting to disconnect a cell that is not connected."""

    def __init__(self, cell, other):
        """Initialize the exception.

        Args:
            cell: The source cell instance.
            other: The target cell instance that was not connected.
        """
        self.cell = cell
        self.other = other
        super().__init__(
            f"Connection between {cell.coordinate} and {other.coordinate} does not exist."
        )


class DimensionException(MesaException, ValueError):  # noqa: N818
    """Raised when spatial dimensions do not match expectations or are invalid."""

    def __init__(self, message):
        """Initialize the exception.

        Args:
            message: The error message describing the dimension mismatch.
        """
        super().__init__(message)
