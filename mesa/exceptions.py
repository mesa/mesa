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


class ModelException(MesaException):
    """Base exception for errors in the Model class."""


class RNGMismatchException(ModelException, ValueError):  # noqa: N818
    """Raised when there is a mismatch between model and scenario RNGs."""


class TimeException(MesaException):
    """Base exception for errors in the time and scheduling modules."""


class PastEventException(TimeException, ValueError):  # noqa: N818
    """Raised when attempting to schedule an event in the past."""


class InvalidCallbackException(TimeException):
    """Base exception for invalid event callbacks."""


class CallbackTypeError(InvalidCallbackException, TypeError):
    """Raised when an event callback has the wrong type (e.g., not callable)."""


class CallbackValueError(InvalidCallbackException, ValueError):
    """Raised when an event callback has an inappropriate value (e.g., a lambda)."""


class InvalidScheduleException(TimeException, ValueError):  # noqa: N818
    """Raised when a schedule interval or start time is invalid."""


class EmptyEventListException(TimeException, IndexError):  # noqa: N818
    """Raised when attempting to pop or peek an empty event list."""


class AgentException(MesaException):
    """Base exception for errors related to agents."""


class AgentNotRegisteredException(AgentException, LookupError):  # noqa: N818
    """Raised when an operation is performed on an agent not registered with the model."""


class DuplicateAgentIDException(AgentException, KeyError):  # noqa: N818
    """Raised when attempting to register an agent with a duplicate ID."""


class AgentSetException(MesaException):
    """Base exception for errors in AgentSet operations."""


class InvalidOptionException(AgentSetException, ValueError):  # noqa: N818
    """Raised when an invalid option is provided to an AgentSet method."""


class VisualizationException(MesaException):
    """Base exception for visualization-related errors."""


class UnsupportedBackendException(VisualizationException, ValueError):  # noqa: N818
    """Raised when an unsupported visualization backend is specified."""


class UnsupportedSpaceException(VisualizationException, ValueError):  # noqa: N818
    """Raised when an unsupported space type is provided for visualization."""
