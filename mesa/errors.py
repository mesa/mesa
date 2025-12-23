import mesa


class MesaError(Exception):
    """Base class for all Mesa-specific exceptions.
    It automatically appends the Mesa version to help with debugging reports.
    """

    def __init__(self, message: str):
        self.mesa_version = getattr(mesa, "__version__", "unknown")
        # Store the original message cleanly for programmatic access
        self.original_message = message
        full_message = f"[Mesa {self.mesa_version}] {message}"
        super().__init__(full_message)


# Model Errors
class ModelError(MesaError):
    """Generic errors related to model initialization or execution."""


class ConfigurationError(ModelError):
    """Raised when model parameters are invalid or missing."""

    def __init__(self, param_name: str = None, reason: str = None):
        # Allow flexible usage: raise ConfigurationError("Generic message")
        # OR: raise ConfigurationError("width", "must be positive")
        if param_name and reason:
            message = f"Invalid configuration for '{param_name}': {reason}"
            self.param_name = param_name
        else:
            message = param_name if param_name else "Invalid configuration"
            self.param_name = None

        super().__init__(message)


class SeedError(ModelError):
    """Raised when there is a conflict in random number generation settings.
    Example: Providing both a fixed 'seed' and an external 'rng' object.
    """


# Agent Errors
class AgentError(MesaError):
    """Generic errors related to agent behavior or lifecycle."""


class AgentStateError(AgentError):
    """Raised when an agent performs an action invalid for its current state (e.g. moving when dead)."""


# Space Errors
class SpaceError(MesaError):
    """Generic errors related to space, grids, or movement."""


class GridDimensionError(SpaceError):
    """Raised when grid dimensions are invalid.
    Examples: Negative width/height, non-integer dimensions, or mismatching
    dimensions between a grid and a property layer.
    """


class OutOfBoundsError(SpaceError):
    """Raised when an agent attempts to move to a coordinate outside the grid."""

    def __init__(self, pos, dimensions):
        self.pos = pos
        self.dimensions = dimensions
        message = f"Position {pos} is out of bounds for grid dimensions {dimensions}."
        super().__init__(message)


class CellNotEmptyError(SpaceError):
    """Raised in SingleGrid when moving to an occupied cell."""

    def __init__(self, pos, content):
        self.pos = pos
        self.content = content
        message = f"Cell {pos} is already occupied by {content}."
        super().__init__(message)


class DuplicatePropertyLayerError(SpaceError):
    """Raised when attempting to add a property layer with a name that already exists."""

    def __init__(self, layer_name):
        self.layer_name = layer_name
        message = f"Property layer '{layer_name}' already exists in the grid."
        super().__init__(message)


class PropertyLayerNotFoundError(SpaceError):
    """Raised when attempting to access or remove a property layer that does not exist."""

    def __init__(self, layer_name):
        self.layer_name = layer_name
        message = f"Property layer '{layer_name}' does not exist."
        super().__init__(message)


# Scheduler Errors
class TimeError(MesaError):
    """Base class for errors related to time or scheduling."""


class ScheduleError(TimeError):
    """Errors related to the Time/Scheduler (e.g., removing an agent not in schedule)."""


class EventSchedulingError(TimeError):
    """Raised specifically in DEVS/Event simulators.
    Example: Attempting to schedule an event in the past.
    """


# Visualization Errors
class VisualizationError(MesaError):
    """Errors related to visualization backends (Solara, Canvas, etc.)."""


class UserInputError(VisualizationError):
    """Raised when user-provided parameters in the visualization are invalid."""

    def __init__(self, param_name, reason=None):
        if reason:
            message = f"Invalid visualization parameter '{param_name}': {reason}"
        else:
            message = param_name  # Fallback if someone raises just a string
        super().__init__(message)
