"""Mesa Space Module.

Objects used to add a spatial component to a model.

.. note::
    mesa.space now in maintenance-only mode. While these classes remain
    fully supported, new development occurs in the discrete space module
    (:mod:`mesa.discrete_space`) and the experimental ContinuousSpace module

Classes
-------
* PropertyLayer: A data layer that can be added to Grids to store cell properties
* SingleGrid: a Grid which strictly enforces one agent per cell.
* MultiGrid: a Grid where each cell can contain a set of agents.
* HexGrid: a Grid to handle hexagonal neighbors.
* ContinuousSpace: a two-dimensional space where each agent has an arbitrary position of `float`'s.
* NetworkGrid: a network where each node contains zero or more agents.
"""

# Postpone annotation evaluation to avoid NameError from forward references (PEP 563). Remove once Python 3.14+ is required.
from __future__ import annotations

import collections
import contextlib
import inspect
import itertools
import math
import warnings
from collections.abc import Callable, Iterable, Iterator, Sequence
from numbers import Real
from typing import Any, TypeVar, cast, overload
from warnings import warn

with contextlib.suppress(ImportError):
    import networkx as nx

import numpy as np
import numpy.typing as npt

# For Mypy
from .agent import Agent, AgentSet

# for better performance, we calculate the tuple to use in the is_integer function
_types_integer = (int, np.integer)

Coordinate = tuple[int, int]
# used in ContinuousSpace
FloatCoordinate = tuple[float, float] | npt.NDArray[float]
NetworkCoordinate = int

Position = Coordinate | FloatCoordinate | NetworkCoordinate

GridContent = Agent | None
MultiGridContent = list[Agent]

F = TypeVar("F", bound=Callable[..., Any])


def accept_tuple_argument[F: Callable[..., Any]](wrapped_function: F) -> F:
    """Decorator to allow grid methods that take a list of (x, y) coord tuples to also handle a single position.

    Tuples are wrapped in a single-item list rather than forcing user to do it.
    """

    def wrapper(grid_instance, positions) -> Any:
        if len(positions) == 2 and not isinstance(positions[0], tuple):
            positions = [positions]
        return wrapped_function(grid_instance, positions)

    return cast(F, wrapper)


def is_integer(x: Real) -> bool:
    """Check if x is either a CPython integer or Numpy integer."""
    return isinstance(x, _types_integer)


def warn_if_agent_has_position_already(placement_func):
    """Decorator to give warning if agent has position already set."""

    def wrapper(self, agent, *args, **kwargs):
        if agent.pos is not None:
            warnings.warn(
                f"""Agent {agent.unique_id} is being placed with
place_agent() despite already having the position {agent.pos}. In most
cases, you'd want to clear the current position with remove_agent()
before placing the agent again.""",
                stacklevel=2,
            )
        placement_func(self, agent, *args, **kwargs)

    return wrapper


class _Grid:
    """Base class for a rectangular grid.

    Grid cells are indexed by [x, y], where [0, 0] is assumed to be the
    bottom-left and [width-1, height-1] is the top-right. If a grid is
    toroidal, the top and bottom, and left and right, edges wrap to each other

    Properties:
        width, height: The grid's width and height.
        torus: Boolean which determines whether to treat the grid as a torus.
    """

    def __init__(self, width: int, height: int, torus: bool) -> None:
        """Create a new grid.

        Args:
            width: The grid's width.
            height: The grid's height.
            torus: Boolean whether the grid wraps or not.
        """
        self.height = height
        self.width = width
        self.torus = torus
        self.num_cells = height * width

        # Internal list-of-lists which holds the grid cells themselves
        self._grid: list[list[GridContent]]
        self._grid = [
            [self.default_val() for _ in range(self.height)] for _ in range(self.width)
        ]

        # Flag to check if the empties set has been created. Better than initializing
        # _empties as set() because in this case it would become impossible to discern
        # if the set hasn't still being built or if it has become empty after creation.
        self._empties_built = False

        # Neighborhood Cache
        self._neighborhood_cache: dict[Any, Sequence[Coordinate]] = {}

        # Cutoff used inside self.move_to_empty. The parameters are fitted on Python
        # 3.11 and it was verified that they are roughly the same for 3.10. Refer to
        # the code in PR#1565 to check for their stability when a new release gets out.
        self.cutoff_empties = 7.953 * self.num_cells**0.384

    @staticmethod
    def default_val() -> None:
        """Default value for new cell elements."""
        return None

    @property
    def empties(self) -> set:
        if not self._empties_built:
            self.build_empties()
        return self._empties

    def build_empties(self) -> None:
        self._empties = set(
            filter(
                self.is_cell_empty,
                itertools.product(range(self.width), range(self.height)),
            )
        )
        self._empties_built = True

    @overload
    def __getitem__(self, index: int | Sequence[Coordinate]) -> list[GridContent]: ...

    @property
    def agents(self) -> AgentSet:
        """Return an AgentSet with the agents in the space."""
        agents = []
        for entry in self:
            if not entry:
                continue
            if not isinstance(entry, list):
                entry = [entry]  # noqa PLW2901
            for agent in entry:
                agents.append(agent)

        # getting the rng is a bit hacky because old style spaces don't have the rng
        try:
            rng = agents[0].random
        except IndexError:
            # there are no agents in the space
            rng = None
        return AgentSet(agents, random=rng)

    @overload
    def __getitem__(
        self, index: tuple[int | slice, int | slice]
    ) -> GridContent | list[GridContent]: ...

    def __getitem__(self, index):
        """Access contents from the grid."""
        if isinstance(index, int):
            # grid[x]
            return self._grid[index]
        elif isinstance(index[0], tuple):
            # grid[(x1, y1), (x2, y2), ...]
            index = cast(Sequence[Coordinate], index)
            return [self._grid[x][y] for x, y in map(self.torus_adj, index)]

        x, y = index
        x_int, y_int = is_integer(x), is_integer(y)

        if x_int and y_int:
            # grid[x, y]
            index = cast(Coordinate, index)
            x, y = self.torus_adj(index)
            return self._grid[x][y]
        elif x_int:
            # grid[x, :]
            x, _ = self.torus_adj((x, 0))
            y = cast(slice, y)
            return self._grid[x][y]
        elif y_int:
            # grid[:, y]
            _, y = self.torus_adj((0, y))
            x = cast(slice, x)
            return [rows[y] for rows in self._grid[x]]
        else:
            # grid[:, :]
            x, y = (cast(slice, x), cast(slice, y))
            return [cell for rows in self._grid[x] for cell in rows[y]]

    def __iter__(self) -> Iterator[GridContent]:
        """Create an iterator that chains the rows of the grid together as if it is one list."""
        return itertools.chain(*self._grid)

    def coord_iter(self) -> Iterator[tuple[GridContent, Coordinate]]:
        """An iterator that returns positions as well as cell contents."""
        for row in range(self.width):
            for col in range(self.height):
                yield self._grid[row][col], (row, col)  # agent, position

    def iter_neighborhood(
        self,
        pos: Coordinate,
        moore: bool,
        include_center: bool = False,
        radius: int = 1,
    ) -> Iterator[Coordinate]:
        """Return an iterator over cell coordinates that are in the neighborhood of a certain point.

        Args:
            pos: Coordinate tuple for the neighborhood to get.
            moore: If True, return Moore neighborhood
                        (including diagonals)
                   If False, return Von Neumann neighborhood
                        (exclude diagonals)
            include_center: If True, return the (x, y) cell as well.
                            Otherwise, return surrounding cells only.
            radius: radius, in cells, of neighborhood to get.

        Returns:
            An iterator of coordinate tuples representing the neighborhood. For
            example with radius 1, it will return list with number of elements
            equals at most 9 (8) if Moore, 5 (4) if Von Neumann (if not
            including the center).
        """
        yield from self.get_neighborhood(pos, moore, include_center, radius)

    def get_neighborhood(
        self,
        pos: Coordinate,
        moore: bool,
        include_center: bool = False,
        radius: int = 1,
    ) -> Sequence[Coordinate]:
        """Return a list of cells that are in the neighborhood of a certain point.

        Args:
            pos: Coordinate tuple for the neighborhood to get.
            moore: If True, return Moore neighborhood
                   (including diagonals)
                   If False, return Von Neumann neighborhood
                   (exclude diagonals)
            include_center: If True, return the (x, y) cell as well.
                            Otherwise, return surrounding cells only.
            radius: radius, in cells, of neighborhood to get.

        Returns:
            A list of coordinate tuples representing the neighborhood;
            With radius 1, at most 9 if Moore, 5 if Von Neumann (8 and 4
            if not including the center).
        """
        cache_key = (pos, moore, include_center, radius)
        neighborhood = self._neighborhood_cache.get(cache_key, None)

        if neighborhood is not None:
            return neighborhood

        if self.out_of_bounds(pos):
            raise Exception("The `pos` tuple passed is out of bounds.")

        # we use a dict to keep insertion order
        neighborhood = {}

        x, y = pos

        # First we check if the neighborhood is inside the grid
        if (
            x >= radius
            and self.width - x > radius
            and y >= radius
            and self.height - y > radius
        ):
            # If the radius is smaller than the distance from the borders, we
            # can skip boundary checks.
            x_range = range(x - radius, x + radius + 1)
            y_range = range(y - radius, y + radius + 1)

            for new_x in x_range:
                for new_y in y_range:
                    if not moore and abs(new_x - x) + abs(new_y - y) > radius:
                        continue

                    neighborhood[(new_x, new_y)] = True

        else:
            # If the radius is larger than the distance from the borders, we
            # must use a slower method, that takes into account the borders
            # and the torus property.
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if not moore and abs(dx) + abs(dy) > radius:
                        continue

                    new_x = x + dx
                    new_y = y + dy

                    if self.torus:
                        new_x %= self.width
                        new_y %= self.height

                    if not self.out_of_bounds((new_x, new_y)):
                        neighborhood[(new_x, new_y)] = True

        if not include_center:
            neighborhood.pop(pos, None)

        self._neighborhood_cache[cache_key] = tuple(neighborhood.keys())

        return tuple(neighborhood.keys())

    def iter_neighbors(
        self,
        pos: Coordinate,
        moore: bool,
        include_center: bool = False,
        radius: int = 1,
    ) -> Iterator[Agent]:
        """Return an iterator over neighbors to a certain point.

        Args:
            pos: Coordinates for the neighborhood to get.
            moore: If True, return Moore neighborhood
                    (including diagonals)
                   If False, return Von Neumann neighborhood
                     (exclude diagonals)
            include_center: If True, return the (x, y) cell as well.
                            Otherwise,
                            return surrounding cells only.
            radius: radius, in cells, of neighborhood to get.

        Returns:
            An iterator of non-None objects in the given neighborhood;
            at most 9 if Moore, 5 if Von-Neumann
            (8 and 4 if not including the center).
        """
        default_val = self.default_val()
        for x, y in self.get_neighborhood(pos, moore, include_center, radius):
            if (cell := self._grid[x][y]) != default_val:
                yield cell

    def get_neighbors(
        self,
        pos: Coordinate,
        moore: bool,
        include_center: bool = False,
        radius: int = 1,
    ) -> list[Agent]:
        """Return a list of neighbors to a certain point.

        Args:
            pos: Coordinate tuple for the neighborhood to get.
            moore: If True, return Moore neighborhood
                    (including diagonals)
                   If False, return Von Neumann neighborhood
                     (exclude diagonals)
            include_center: If True, return the (x, y) cell as well.
                            Otherwise,
                            return surrounding cells only.
            radius: radius, in cells, of neighborhood to get.

        Returns:
            A list of non-None objects in the given neighborhood;
            at most 9 if Moore, 5 if Von-Neumann
            (8 and 4 if not including the center).
        """
        return list(self.iter_neighbors(pos, moore, include_center, radius))

    def torus_adj(self, pos: Coordinate) -> Coordinate:
        """Convert coordinate, handling torus looping."""
        if not self.out_of_bounds(pos):
            return pos
        elif not self.torus:
            raise Exception("Point out of bounds, and space non-toroidal.")
        else:
            return pos[0] % self.width, pos[1] % self.height

    def out_of_bounds(self, pos: Coordinate) -> bool:
        """Determines whether position is off the grid, returns the out of bounds coordinate."""
        x, y = pos
        return x < 0 or x >= self.width or y < 0 or y >= self.height

    @accept_tuple_argument
    def iter_cell_list_contents(
        self, cell_list: Iterable[Coordinate]
    ) -> Iterator[Agent]:
        """Returns an iterator of the agents contained in the cells identified in `cell_list`; cells with empty content are excluded.

        Args:
            cell_list: Array-like of (x, y) tuples, or single tuple.

        Returns:
            An iterator of the agents contained in the cells identified in `cell_list`.
        """
        # iter_cell_list_contents returns only non-empty contents.
        default_val = self.default_val()
        for x, y in cell_list:
            if (cell := self._grid[x][y]) != default_val:
                yield cell

    @accept_tuple_argument
    def get_cell_list_contents(self, cell_list: Iterable[Coordinate]) -> list[Agent]:
        """Returns an iterator of the agents contained in the cells identified in `cell_list`; cells with empty content are excluded.

        Args:
            cell_list: Array-like of (x, y) tuples, or single tuple.

        Returns:
            A list of the agents contained in the cells identified in `cell_list`.
        """
        return list(self.iter_cell_list_contents(cell_list))

    def place_agent(self, agent: Agent, pos: Coordinate) -> None: ...

    def remove_agent(self, agent: Agent) -> None: ...

    def move_agent(self, agent: Agent, pos: Coordinate) -> None:
        """Move an agent from its current position to a new position.

        Args:
            agent: Agent object to move. Assumed to have its current location
                   stored in a 'pos' tuple.
            pos: Tuple of new position to move the agent to.
        """
        pos = self.torus_adj(pos)
        self.remove_agent(agent)
        self.place_agent(agent, pos)

    def move_agent_to_one_of(
        self,
        agent: Agent,
        pos: list[Coordinate],
        selection: str = "random",
        handle_empty: str | None = None,
    ) -> None:
        """Move an agent to one of the given positions.

        Args:
            agent: Agent object to move. Assumed to have its current location stored in a 'pos' tuple.
            pos: List of possible positions.
            selection: String, either "random" (default) or "closest". If "closest" is selected and multiple
                       cells are the same distance, one is chosen randomly.
            handle_empty: String, either "warning", "error" or None (default). If "warning" or "error" is selected
                          and no positions are given (an empty list), a warning or error is raised respectively.
        """
        # Only move agent if there are positions given (non-empty list)
        if pos:
            if selection == "random":
                chosen_pos = agent.random.choice(pos)
            elif selection == "closest":
                current_pos = agent.pos
                # Find the closest position without sorting all positions
                # TODO: See if this method can be optimized further
                closest_pos = []
                min_distance = float("inf")
                agent.random.shuffle(pos)
                for p in pos:
                    distance = self._distance_squared(p, current_pos)
                    if distance < min_distance:
                        min_distance = distance
                        closest_pos.clear()
                        closest_pos.append(p)
                    elif distance == min_distance:
                        closest_pos.append(p)

                chosen_pos = agent.random.choice(closest_pos)

            else:
                raise ValueError(
                    f"Invalid selection method {selection}. Choose 'random' or 'closest'."
                )
            #  Move agent to chosen position
            self.move_agent(agent, chosen_pos)

        # If no positions are given, throw warning/error if selected
        elif handle_empty == "warning":
            warn(
                f"No positions given, could not move agent {agent.unique_id}.",
                RuntimeWarning,
                stacklevel=2,
            )
        elif handle_empty == "error":
            raise ValueError(
                f"No positions given, could not move agent {agent.unique_id}."
            )

    def _distance_squared(self, pos1: Coordinate, pos2: Coordinate) -> float:
        """Calculate the squared Euclidean distance between two points for performance."""
        # Use squared Euclidean distance to avoid sqrt operation
        dx, dy = abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1])
        if self.torus:
            dx = min(dx, self.width - dx)
            dy = min(dy, self.height - dy)
        return dx**2 + dy**2

    def swap_pos(self, agent_a: Agent, agent_b: Agent) -> None:
        """Swap agents positions."""
        agents_no_pos = []
        if (pos_a := agent_a.pos) is None:
            agents_no_pos.append(agent_a)
        if (pos_b := agent_b.pos) is None:
            agents_no_pos.append(agent_b)
        if agents_no_pos:
            agents_no_pos = [f"<Agent id: {a.unique_id}>" for a in agents_no_pos]
            raise Exception(f"{', '.join(agents_no_pos)} - not on the grid")

        if pos_a == pos_b:
            return

        self.remove_agent(agent_a)
        self.remove_agent(agent_b)

        self.place_agent(agent_a, pos_b)
        self.place_agent(agent_b, pos_a)

    def is_cell_empty(self, pos: Coordinate) -> bool:
        """Returns a bool of the contents of a cell."""
        x, y = pos
        return self._grid[x][y] == self.default_val()

    def move_to_empty(self, agent: Agent) -> None:
        """Moves agent to a random empty cell, vacating agent's old cell."""
        num_empty_cells = len(self.empties)
        if num_empty_cells == 0:
            raise Exception("ERROR: No empty cells")

        # This method is based on Agents.jl's random_empty() implementation. See
        # https://github.com/JuliaDynamics/Agents.jl/pull/541. For the discussion, see
        # https://github.com/mesa/mesa/issues/1052 and
        # https://github.com/mesa/mesa/pull/1565. The cutoff value provided
        # is the break-even comparison with the time taken in the else branching point.
        if num_empty_cells > self.cutoff_empties:
            while True:
                new_pos = (
                    agent.random.randrange(self.width),
                    agent.random.randrange(self.height),
                )
                if self.is_cell_empty(new_pos):
                    break
        else:
            new_pos = agent.random.choice(sorted(self.empties))
        self.remove_agent(agent)
        self.place_agent(agent, new_pos)

    def exists_empty_cells(self) -> bool:
        """Return True if any cells empty else False."""
        return len(self.empties) > 0


def is_single_argument_function(function):
    """Check if a function is a single argument function."""
    return (
        inspect.isfunction(function)
        and len(inspect.signature(function).parameters) == 1
    )


def ufunc_requires_additional_input(ufunc):  # noqa: D103
    # NumPy ufuncs have a 'nargs' attribute indicating the number of input arguments
    # For binary ufuncs (like np.add), nargs is 2
    return ufunc.nargs > 1


class PropertyLayer:
    """A class representing a layer of properties in a two-dimensional grid.

    Each cell in the grid can store a value of a specified data type.

    Attributes:
        name (str): The name of the property layer.
        width (int): The width of the grid (number of columns).
        height (int): The height of the grid (number of rows).
        data (numpy.ndarray): A NumPy array representing the grid data.

    """

    def __init__(
        self, name: str, width: int, height: int, default_value, dtype=np.float64
    ):
        """Initializes a new PropertyLayer instance.

        Args:
            name (str): The name of the property layer.
            width (int): The width of the grid (number of columns). Must be a positive integer.
            height (int): The height of the grid (number of rows). Must be a positive integer.
            default_value: The default value to initialize each cell in the grid. Should ideally
                           be of the same type as specified by the dtype parameter.
            dtype (data-type, optional): The desired data-type for the grid's elements. Default is np.float64.

        Raises:
            ValueError: If width or height is not a positive integer.

        Notes:
            A UserWarning is raised if the default_value is not of a type compatible with dtype.
            The dtype parameter can accept both Python data types (like bool, int or float) and NumPy data types
            (like np.int64 or np.float64). Using NumPy data types is recommended (except for bool) for better control
            over the precision and efficiency of data storage and computations, especially in cases of large data
            volumes or specialized numerical operations.
        """
        self.name = name
        self.width = width
        self.height = height

        # Check that width and height are positive integers
        if (not isinstance(width, int) or width < 1) or (
            not isinstance(height, int) or height < 1
        ):
            raise ValueError(
                f"Width and height must be positive integers, got {width} and {height}."
            )
        # Check if the dtype is suitable for the data
        try:
            if dtype(default_value) != default_value:
                warnings.warn(
                    f"Default value {default_value} will lose precision when converted to {dtype.__name__}.",
                    UserWarning,
                    stacklevel=2,
                )
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"Default value {default_value} is incompatible with dtype={dtype.__name__}."
            ) from e

        self.data = np.full((width, height), default_value, dtype=dtype)

    def set_cell(self, position: Coordinate, value):
        """Update a single cell's value in-place."""
        self.data[position] = value

    def set_cells(self, value, condition=None):
        """Perform a batch update either on the entire grid or conditionally, in-place.

        Args:
            value: The value to be used for the update.
            condition: (Optional) A callable (like a lambda function or a NumPy ufunc)
                       that returns a boolean array when applied to the data.
        """
        if condition is None:
            np.copyto(self.data, value)  # In-place update
        else:
            if isinstance(condition, np.ufunc):
                # Directly apply NumPy ufunc
                condition_result = condition(self.data)
            else:
                # Vectorize non-ufunc conditions
                vectorized_condition = np.vectorize(condition)
                condition_result = vectorized_condition(self.data)

            if (
                not isinstance(condition_result, np.ndarray)
                or condition_result.shape != self.data.shape
            ):
                raise ValueError(
                    "Result of condition must be a NumPy array with the same shape as the grid."
                )

            np.copyto(self.data, value, where=condition_result)

    def modify_cell(self, position: Coordinate, operation, value=None):
        """Modify a single cell using an operation, which can be a lambda function or a NumPy ufunc.

        If a NumPy ufunc is used, an additional value should be provided.

        Args:
            position: The grid coordinates of the cell to modify.
            operation: A function to apply. Can be a lambda function or a NumPy ufunc.
            value: The value to be used if the operation is a NumPy ufunc. Ignored for lambda functions.
        """
        current_value = self.data[position]

        # Determine if the operation is a lambda function or a NumPy ufunc
        if is_single_argument_function(operation):
            # Lambda function case
            self.data[position] = operation(current_value)
        elif value is not None:
            # NumPy ufunc case
            self.data[position] = operation(current_value, value)
        else:
            raise ValueError("Invalid operation or missing value for NumPy ufunc.")

    def modify_cells(self, operation, value=None, condition_function=None):
        """Modify cells using an operation, which can be a lambda function or a NumPy ufunc.

        If a NumPy ufunc is used, an additional value should be provided.

        Args:
            operation: A function to apply. Can be a lambda function or a NumPy ufunc.
            value: The value to be used if the operation is a NumPy ufunc. Ignored for lambda functions.
            condition_function: (Optional) A callable that returns a boolean array when applied to the data.
        """
        condition_array = np.ones_like(
            self.data, dtype=bool
        )  # Default condition (all cells)
        if condition_function is not None:
            if isinstance(condition_function, np.ufunc):
                condition_array = condition_function(self.data)
            else:
                vectorized_condition = np.vectorize(condition_function)
                condition_array = vectorized_condition(self.data)

        # Check if the operation is a lambda function or a NumPy ufunc
        if isinstance(operation, np.ufunc):
            if ufunc_requires_additional_input(operation):
                if value is None:
                    raise ValueError("This ufunc requires an additional input value.")
                modified_data = operation(self.data, value)
            else:
                modified_data = operation(self.data)
        else:
            # Vectorize non-ufunc operations
            vectorized_operation = np.vectorize(operation)
            modified_data = vectorized_operation(self.data)

        self.data = np.where(condition_array, modified_data, self.data)

    def select_cells(self, condition, return_list=True):
        """Find cells that meet a specified condition using NumPy's boolean indexing, in-place.

        Args:
            condition: A callable that returns a boolean array when applied to the data.
            return_list: (Optional) If True, return a list of (x, y) tuples. Otherwise, return a boolean array.

        Returns:
            A list of (x, y) tuples or a boolean array.
        """
        condition_array = condition(self.data)
        if return_list:
            return list(zip(*np.where(condition_array)))
        else:
            return condition_array

    def aggregate_property(self, operation):
        """Perform an aggregate operation (e.g., sum, mean) on a property across all cells.

        Args:
            operation: A function to apply. Can be a lambda function or a NumPy ufunc.
        """
        return operation(self.data)


class _PropertyGrid(_Grid):
    """A private subclass of _Grid that supports the addition of property layers.

    This enables the representation and manipulation of additional data layers on the grid. This class is
    intended for internal use within the Mesa framework and is currently utilized by SingleGrid
    and MultiGrid classes to provide enhanced grid functionality.

    The `_PropertyGrid` extends the capabilities of a basic grid by allowing each cell
    to have multiple properties, each represented by a separate PropertyLayer.
    These properties can be used to model complex environments where each cell
    has multiple attributes or states.

    Attributes:
        properties (dict): A dictionary mapping property layer names to PropertyLayer instances.
        empty_mask (np.ndarray): A boolean array indicating empty cells on the grid.


    Several methods in this class accept a mask as an input, which is a NumPy ndarray of boolean values. This mask
    specifies the cells to be considered (True) or ignored (False) in operations. Users can create custom masks,
    including neighborhood masks, to apply specific conditions or constraints. Additionally, methods that deal with
    cell selection or agent movement can return either a list of cell coordinates or a mask, based on the 'return_list'
    parameter. This flexibility allows for more nuanced control and customization of grid operations, catering to a wide
    range of modeling requirements and scenarios.

    Notes:
        This class is not intended for direct use in user models but is currently used by the SingleGrid and MultiGrid.
    """

    def __init__(
        self,
        width: int,
        height: int,
        torus: bool,
        property_layers: None | PropertyLayer | list[PropertyLayer] = None,
    ):
        """Initializes a new _PropertyGrid instance with specified dimensions and optional property layers.

        Args:
            width (int): The width of the grid (number of columns).
            height (int): The height of the grid (number of rows).
            torus (bool): A boolean indicating if the grid should behave like a torus.
            property_layers (None | PropertyLayer | list[PropertyLayer], optional): A single PropertyLayer instance,
                a list of PropertyLayer instances, or None to initialize without any property layers.

        Raises:
            ValueError: If a property layer's dimensions do not match the grid dimensions.
        """
        super().__init__(width, height, torus)
        self.properties = {}

        # Initialize an empty mask as a boolean NumPy array
        # True = cell is empty, False = cell is occupied
        self._empty_mask = np.ones((self.width, self.height), dtype=bool)

        # Handle both single PropertyLayer instance and list of PropertyLayer instances
        if property_layers:
            # If a single PropertyLayer is passed, convert it to a list
            if isinstance(property_layers, PropertyLayer):
                property_layers = [property_layers]

            for layer in property_layers:
                self.add_property_layer(layer)

    @property
    def empty_mask(self) -> np.ndarray:
        """Returns a boolean mask indicating empty cells on the grid."""
        return self._empty_mask

    # Add and remove properties to the grid
    def add_property_layer(self, property_layer: PropertyLayer):
        """Adds a new property layer to the grid.

        Args:
            property_layer (PropertyLayer): The PropertyLayer instance to be added to the grid.

        Raises:
            ValueError: If a property layer with the same name already exists in the grid.
            ValueError: If the dimensions of the property layer do not match the grid's dimensions.
        """
        if property_layer.name in self.properties:
            raise ValueError(f"Property layer {property_layer.name} already exists.")
        if property_layer.width != self.width or property_layer.height != self.height:
            raise ValueError(
                f"Property layer dimensions {property_layer.width}x{property_layer.height} do not match grid dimensions {self.width}x{self.height}."
            )
        self.properties[property_layer.name] = property_layer

    def remove_property_layer(self, property_name: str):
        """Removes a property layer from the grid by its name.

        Args:
            property_name (str): The name of the property layer to be removed.

        Raises:
            ValueError: If a property layer with the given name does not exist in the grid.
        """
        if property_name not in self.properties:
            raise ValueError(f"Property layer {property_name} does not exist.")
        del self.properties[property_name]

    def get_neighborhood_mask(
        self, pos: Coordinate, moore: bool, include_center: bool, radius: int
    ) -> np.ndarray:
        """Generate a boolean mask representing the neighborhood.

        Helper method for select_cells_multi_properties() and move_agent_to_random_cell()

        Args:
            pos (Coordinate): Center of the neighborhood.
            moore (bool): True for Moore neighborhood, False for Von Neumann.
            include_center (bool): Include the central cell in the neighborhood.
            radius (int): The radius of the neighborhood.

        Returns:
            np.ndarray: A boolean mask representing the neighborhood.
        """
        neighborhood = self.get_neighborhood(pos, moore, include_center, radius)
        mask = np.zeros((self.width, self.height), dtype=bool)

        # Convert the neighborhood list to a NumPy array and use advanced indexing
        coords = np.array(neighborhood)
        mask[coords[:, 0], coords[:, 1]] = True
        return mask

    def select_cells(
        self,
        conditions: dict | None = None,
        extreme_values: dict | None = None,
        masks: np.ndarray | list[np.ndarray] = None,
        only_empty: bool = False,
        return_list: bool = True,
    ) -> list[Coordinate] | np.ndarray:
        """Select cells based on property conditions, extreme values, and/or masks, with an option to only select empty cells.

        Args:
            conditions (dict): A dictionary where keys are property names and values are callables that return a boolean when applied.
            extreme_values (dict): A dictionary where keys are property names and values are either 'highest' or 'lowest'.
            masks (np.ndarray | list[np.ndarray], optional): A mask or list of masks to restrict the selection.
            only_empty (bool, optional): If True, only select cells that are empty. Default is False.
            return_list (bool, optional): If True, return a list of coordinates, otherwise return a mask.

        Returns:
            Union[list[Coordinate], np.ndarray]: Coordinates where conditions are satisfied or the combined mask.
        """
        # Initialize the combined mask
        combined_mask = np.ones((self.width, self.height), dtype=bool)

        # Apply the masks
        if masks is not None:
            if isinstance(masks, list):
                for mask in masks:
                    combined_mask = np.logical_and(combined_mask, mask)
            else:
                combined_mask = np.logical_and(combined_mask, masks)

        # Apply the empty mask if only_empty is True
        if only_empty:
            combined_mask = np.logical_and(combined_mask, self.empty_mask)

        # Apply conditions
        if conditions:
            for prop_name, condition in conditions.items():
                prop_layer = self.properties[prop_name].data
                prop_mask = condition(prop_layer)
                combined_mask = np.logical_and(combined_mask, prop_mask)

        # Apply extreme values
        if extreme_values:
            for property_name, mode in extreme_values.items():
                prop_values = self.properties[property_name].data

                # Create a masked array using the combined_mask
                masked_values = np.ma.masked_array(prop_values, mask=~combined_mask)

                if mode == "highest":
                    target_value = masked_values.max()
                elif mode == "lowest":
                    target_value = masked_values.min()
                else:
                    raise ValueError(
                        f"Invalid mode {mode}. Choose from 'highest' or 'lowest'."
                    )

                extreme_value_mask = prop_values == target_value
                combined_mask = np.logical_and(combined_mask, extreme_value_mask)

        # Generate output
        if return_list:
            selected_cells = list(zip(*np.where(combined_mask)))
            return selected_cells
        else:
            return combined_mask


class SingleGrid(_PropertyGrid):
    """Rectangular grid where each cell contains exactly at most one agent.

    Grid cells are indexed by [x, y], where [0, 0] is assumed to be the
    bottom-left and [width-1, height-1] is the top-right. If a grid is
    toroidal, the top and bottom, and left and right, edges wrap to each other.

    This class provides a property `empties` that returns a set of coordinates
    for all empty cells in the grid. It is automatically updated whenever
    agents are added or removed from the grid. The `empties` property should be
    used for efficient access to current empty cells rather than manually
    iterating over the grid to check for emptiness.

    Properties:
        width, height: The grid's width and height.
        torus: Boolean which determines whether to treat the grid as a torus.
        empties: Returns a set of (x, y) tuples for all empty cells. This set is
                 maintained internally and provides a performant way to query
                 the grid for empty spaces.
    """

    @warn_if_agent_has_position_already
    def place_agent(self, agent: Agent, pos: Coordinate) -> None:
        """Place the agent at the specified location, and set its pos variable."""
        if self.is_cell_empty(pos):
            x, y = pos
            self._grid[x][y] = agent
            if self._empties_built:
                self._empties.discard(pos)
            self._empty_mask[pos] = False
            agent.pos = pos
        else:
            raise Exception("Cell not empty")

    def remove_agent(self, agent: Agent) -> None:
        """Remove the agent from the grid and set its pos attribute to None."""
        if (pos := agent.pos) is None:
            return
        x, y = pos
        self._grid[x][y] = self.default_val()
        if self._empties_built:
            self._empties.add(pos)
        self._empty_mask[agent.pos] = True
        agent.pos = None


class MultiGrid(_PropertyGrid):
    """Rectangular grid where each cell can contain more than one agent.

    Grid cells are indexed by [x, y], where [0, 0] is assumed to be at
    bottom-left and [width-1, height-1] is the top-right. If a grid is
    toroidal, the top and bottom, and left and right, edges wrap to each other.

    This class maintains an `empties` property, which is a set of coordinates
    for all cells that currently contain no agents. This property is updated
    automatically as agents are added to or removed from the grid.

    Properties:
        width, height: The grid's width and height.
        torus: Boolean which determines whether to treat the grid as a torus.
        empties: Returns a set of (x, y) tuples for all empty cells.
    """

    grid: list[list[MultiGridContent]]

    @staticmethod
    def default_val() -> MultiGridContent:
        """Default value for new cell elements."""
        return []

    @warn_if_agent_has_position_already
    def place_agent(self, agent: Agent, pos: Coordinate) -> None:
        """Place the agent at the specified location, and set its pos variable."""
        x, y = pos
        if agent.pos is None or agent not in self._grid[x][y]:
            self._grid[x][y].append(agent)
            agent.pos = pos
            if self._empties_built:
                self._empties.discard(pos)
            self._empty_mask[pos] = False

    def remove_agent(self, agent: Agent) -> None:
        """Remove the agent from the given location and set its pos attribute to None."""
        pos = agent.pos
        x, y = pos
        self._grid[x][y].remove(agent)
        if self._empties_built and self.is_cell_empty(pos):
            self._empties.add(pos)
        if self.is_cell_empty(pos):
            self._empty_mask[pos] = True
        agent.pos = None

    def iter_neighbors(  # noqa: D102
        self,
        pos: Coordinate,
        moore: bool,
        include_center: bool = False,
        radius: int = 1,
    ) -> Iterator[Agent]:
        return itertools.chain.from_iterable(
            super().iter_neighbors(pos, moore, include_center, radius)
        )

    @accept_tuple_argument
    def iter_cell_list_contents(
        self, cell_list: Iterable[Coordinate]
    ) -> Iterator[Agent]:
        """Returns an iterator of the agents contained in the cells identified in `cell_list`.

        Cells with empty content are excluded.

        Args:
            cell_list: Array-like of (x, y) tuples, or single tuple.

        Returns:
            An iterator of the agents contained in the cells identified in `cell_list`.
        """
        default_val = self.default_val()
        return itertools.chain.from_iterable(
            cell for x, y in cell_list if (cell := self._grid[x][y]) != default_val
        )


class _HexGrid:
    """Hexagonal Grid which handles hexagonal neighbors.

    Functions according to odd-q rules.
    See http://www.redblobgames.com/grids/hexagons/#coordinates for more.

    Properties:
        width, height: The grid's width and height.
        torus: Boolean which determines whether to treat the grid as a torus.

    Methods:
        get_neighbors: Returns the objects surrounding a given cell.
        get_neighborhood: Returns the cells surrounding a given cell.
        iter_neighbors: Iterates over position neighbors.
        iter_neighborhood: Returns an iterator over cell coordinates that are
            in the neighborhood of a certain point.
    """

    def torus_adj_2d(self, pos: Coordinate) -> Coordinate:
        return pos[0] % self.width, pos[1] % self.height

    def get_neighborhood(
        self, pos: Coordinate, include_center: bool = False, radius: int = 1
    ) -> list[Coordinate]:
        """Return a list of coordinates that are in the neighborhood of a certain point.

        To calculate the neighborhood for a HexGrid the parity of the x coordinate of the point is
        important, the neighborhood can be sketched as:

            Always: (0,-), (0,+)
            When x is even: (-,+), (-,0), (+,+), (+,0)
            When x is odd:  (-,0), (-,-), (+,0), (+,-)

        Args:
            pos: Coordinate tuple for the neighborhood to get.
            include_center: If True, return the (x, y) cell as well.
                            Otherwise, return surrounding cells only.
            radius: radius, in cells, of neighborhood to get.

        Returns:
            A list of coordinate tuples representing the neighborhood. For
            example with radius 1, it will return list with number of elements
            equals at most 9 (8) if Moore, 5 (4) if Von Neumann (if not
            including the center).
        """
        cache_key = (pos, include_center, radius)
        neighborhood = self._neighborhood_cache.get(cache_key, None)

        if neighborhood is not None:
            return neighborhood

        queue = collections.deque()
        queue.append(pos)
        coordinates = set()

        while radius > 0:
            level_size = len(queue)
            radius -= 1

            for _i in range(level_size):
                x, y = queue.pop()

                if x % 2 == 0:
                    adjacent = [
                        (x, y - 1),
                        (x, y + 1),
                        (x - 1, y + 1),
                        (x - 1, y),
                        (x + 1, y + 1),
                        (x + 1, y),
                    ]
                else:
                    adjacent = [
                        (x, y - 1),
                        (x, y + 1),
                        (x - 1, y),
                        (x - 1, y - 1),
                        (x + 1, y),
                        (x + 1, y - 1),
                    ]

                if self.torus:
                    adjacent = [
                        coord
                        for coord in map(self.torus_adj_2d, adjacent)
                        if coord not in coordinates
                    ]
                else:
                    adjacent = [
                        coord
                        for coord in adjacent
                        if not self.out_of_bounds(coord) and coord not in coordinates
                    ]

                coordinates.update(adjacent)

                if radius > 0:
                    queue.extendleft(adjacent)

        if include_center:
            coordinates.add(pos)
        else:
            coordinates.discard(pos)

        neighborhood = tuple(sorted(coordinates))
        self._neighborhood_cache[cache_key] = neighborhood

        return neighborhood

    def iter_neighborhood(
        self, pos: Coordinate, include_center: bool = False, radius: int = 1
    ) -> Iterator[Coordinate]:
        """Return an iterator over cell coordinates that are in the neighborhood of a certain point.

        Args:
            pos: Coordinate tuple for the neighborhood to get.
            include_center: If True, return the (x, y) cell as well.
                            Otherwise, return surrounding cells only.
            radius: radius, in cells, of neighborhood to get.

        Returns:
            An iterator of coordinate tuples representing the neighborhood.
        """
        yield from self.get_neighborhood(pos, include_center, radius)

    def iter_neighbors(
        self, pos: Coordinate, include_center: bool = False, radius: int = 1
    ) -> Iterator[Agent]:
        """Return an iterator over neighbors to a certain point.

        Args:
            pos: Coordinates for the neighborhood to get.
            include_center: If True, return the (x, y) cell as well.
                            Otherwise,
                            return surrounding cells only.
            radius: radius, in cells, of neighborhood to get.

        Returns:
            An iterator of non-None objects in the given neighborhood
        """
        neighborhood = self.get_neighborhood(pos, include_center, radius)
        return self.iter_cell_list_contents(neighborhood)

    def get_neighbors(
        self, pos: Coordinate, include_center: bool = False, radius: int = 1
    ) -> list[Agent]:
        """Return a list of neighbors to a certain point.

        Args:
            pos: Coordinate tuple for the neighborhood to get.
            include_center: If True, return the (x, y) cell as well.
                            Otherwise,
                            return surrounding cells only.
            radius: radius, in cells, of neighborhood to get.

        Returns:
            A list of non-None objects in the given neighborhood
        """
        return list(self.iter_neighbors(pos, include_center, radius))


class HexSingleGrid(_HexGrid, SingleGrid):
    """Hexagonal SingleGrid: a SingleGrid where neighbors are computed according to a hexagonal tiling of the grid.

    Functions according to odd-q rules.
    See http://www.redblobgames.com/grids/hexagons/#coordinates for more.

    This class also maintains an `empties` property, similar to SingleGrid,
    which provides a set of coordinates for all empty hexagonal cells.

    Properties:
        width, height: The grid's width and height.
        torus: Boolean which determines whether to treat the grid as a torus.
        empties: Returns a set of hexagonal coordinates for all empty cells.
    """


class HexMultiGrid(_HexGrid, MultiGrid):
    """Hexagonal MultiGrid: a MultiGrid where neighbors are computed according to a hexagonal tiling of the grid.

    Functions according to odd-q rules.
    See http://www.redblobgames.com/grids/hexagons/#coordinates for more.

    Similar to the standard MultiGrid, this class maintains an `empties` property,
    which is a set of coordinates for all hexagonal cells that currently contain
    no agents. This property is updated automatically as agents are added to or
    removed from the grid.

    Properties:
        width, height: The grid's width and height.
        torus: Boolean which determines whether to treat the grid as a torus.
        empties: Returns a set of hexagonal coordinates for all empty cells.
    """


class ContinuousSpace:
    """Continuous space where each agent can have an arbitrary position.

    Assumes that all agents have a pos property storing their position as
    an (x, y) tuple.

    This class uses a numpy array internally to store agents in order to speed
    up neighborhood lookups. This array is calculated on the first neighborhood
    lookup, and is updated if agents are added or removed.

    The concept of 'empty cells' is not directly applicable in continuous space,
    as positions are not discretized.
    """

    def __init__(
        self,
        x_max: float,
        y_max: float,
        torus: bool,
        x_min: float = 0,
        y_min: float = 0,
    ) -> None:
        """Create a new continuous space.

        Args:
            x_max: the maximum x-coordinate
            y_max: the maximum y-coordinate.
            torus: Boolean for whether the edges loop around.
            x_min: (default 0) If provided, set the minimum x -coordinate for the space. Below them, values loop to
                  the other edge (if torus=True) or raise an exception.
            y_min: (default 0) If provided, set the minimum y -coordinate for the space. Below them, values loop to
                  the other edge (if torus=True) or raise an exception.
        """
        self.x_min = x_min
        self.x_max = x_max
        self.width = x_max - x_min
        self.y_min = y_min
        self.y_max = y_max
        self.height = y_max - y_min
        self.center = np.array(((x_max + x_min) / 2, (y_max + y_min) / 2))
        self.size = np.array((self.width, self.height))
        self.torus = torus

        self._agent_points: npt.NDArray[FloatCoordinate] | None = None
        self._index_to_agent: dict[int, Agent] = {}
        self._agent_to_index: dict[Agent, int | None] = {}

    @property
    def agents(self) -> AgentSet:
        """Return an AgentSet with the agents in the space."""
        agents = list(self._agent_to_index)

        # getting the rng is a bit hacky because old style spaces don't have the rng
        try:
            rng = agents[0].random
        except IndexError:
            # there are no agents in the space
            rng = None
        return AgentSet(agents, random=rng)

    def _build_agent_cache(self):
        """Cache agents positions to speed up neighbors calculations."""
        self._index_to_agent = {}
        for idx, agent in enumerate(self._agent_to_index):
            self._agent_to_index[agent] = idx
            self._index_to_agent[idx] = agent
        # Since dicts are ordered by insertion, we can iterate through agents keys
        self._agent_points = np.array([agent.pos for agent in self._agent_to_index])

    def _invalidate_agent_cache(self):
        """Clear cached data of agents and positions in the space."""
        self._agent_points = None
        self._index_to_agent = {}

    @warn_if_agent_has_position_already
    def place_agent(self, agent: Agent, pos: FloatCoordinate) -> None:
        """Place a new agent in the space.

        Args:
            agent: Agent object to place.
            pos: Coordinate tuple for where to place the agent.
        """
        self._invalidate_agent_cache()
        self._agent_to_index[agent] = None
        pos = self.torus_adj(pos)
        agent.pos = pos

    def move_agent(self, agent: Agent, pos: FloatCoordinate) -> None:
        """Move an agent from its current position to a new position.

        Args:
            agent: The agent object to move.
            pos: Coordinate tuple to move the agent to.
        """
        pos = self.torus_adj(pos)
        agent.pos = pos

        if self._agent_points is not None:
            # instead of invalidating the full cache,
            # apply the move to the cached values
            idx = self._agent_to_index[agent]
            self._agent_points[idx] = pos

    def remove_agent(self, agent: Agent) -> None:
        """Remove an agent from the space.

        Args:
            agent: The agent object to remove
        """
        if agent not in self._agent_to_index:
            raise Exception("Agent does not exist in the space")
        del self._agent_to_index[agent]

        self._invalidate_agent_cache()
        agent.pos = None

    def get_neighbors(
        self, pos: FloatCoordinate, radius: float, include_center: bool = True
    ) -> list[Agent]:
        """Get all agents within a certain radius.

        Args:
            pos: (x,y) coordinate tuple to center the search at.
            radius: Get all the objects within this distance of the center.
            include_center: If True, include an object at the *exact* provided
                            coordinates. i.e. if you are searching for the
                            neighbors of a given agent, True will include that
                            agent in the results.

        Notes:
            If 1 or more agents are located on pos, include_center=False will remove all these agents
            from the results. So, if you really want to get the neighbors of a given agent,
            you should set include_center=True, and then filter the list of agents to remove
            the given agent (i.e., self when calling it from an agent).

        """
        if self._agent_points is None:
            self._build_agent_cache()

        deltas = np.abs(self._agent_points - np.array(pos))
        if self.torus:
            deltas = np.minimum(deltas, self.size - deltas)
        dists = deltas[:, 0] ** 2 + deltas[:, 1] ** 2

        (idxs,) = np.where(dists <= radius**2)
        neighbors = [
            self._index_to_agent[x] for x in idxs if include_center or dists[x] > 0
        ]
        return neighbors

    def get_heading(
        self, pos_1: FloatCoordinate, pos_2: FloatCoordinate
    ) -> FloatCoordinate:
        """Get the heading vector between two points, accounting for toroidal space.

        It is possible to calculate the heading angle by applying the atan2 function to the
        result.

        Args:
            pos_1: Coordinate tuples for both points.
            pos_2: Coordinate tuples for both points.
        """
        one = np.array(pos_1)
        two = np.array(pos_2)
        heading = two - one
        if self.torus:
            inverse_heading = heading - np.sign(heading) * self.size

            def get_min_abs(x, y):
                return x if abs(x) < abs(y) else y

            # Choose the smaller heading based on their absolute value for
            # each dimension independently.
            heading = tuple(
                get_min_abs(heading[i], inverse_heading[i]) for i in range(2)
            )
        if isinstance(pos_1, np.ndarray):
            heading = np.asarray(heading)
        else:
            heading = tuple(heading)
        return heading

    def get_distance(self, pos_1: FloatCoordinate, pos_2: FloatCoordinate) -> float:
        """Get the distance between two point, accounting for toroidal space.

        Args:
            pos_1: Coordinate tuples for point1.
            pos_2: Coordinate tuples for point2.
        """
        x1, y1 = pos_1
        x2, y2 = pos_2

        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        if self.torus:
            dx = min(dx, self.width - dx)
            dy = min(dy, self.height - dy)
        return math.sqrt(dx * dx + dy * dy)

    def torus_adj(self, pos: FloatCoordinate) -> FloatCoordinate:
        """Adjust coordinates to handle torus looping.

        If the coordinate is out-of-bounds and the space is toroidal, return
        the corresponding point within the space. If the space is not toroidal,
        raise an exception.

        Args:
            pos: Coordinate tuple to convert.
        """
        if not self.out_of_bounds(pos):
            return pos
        elif not self.torus:
            raise Exception("Point out of bounds, and space non-toroidal.")
        else:
            x = self.x_min + ((pos[0] - self.x_min) % self.width)
            y = self.y_min + ((pos[1] - self.y_min) % self.height)
            if isinstance(pos, tuple):
                return (x, y)
            else:
                return np.array((x, y))

    def out_of_bounds(self, pos: FloatCoordinate) -> bool:
        """Check if a point is out of bounds."""
        x, y = pos
        return x < self.x_min or x >= self.x_max or y < self.y_min or y >= self.y_max


class NetworkGrid:
    """Network Grid where each node contains zero or more agents."""

    def __init__(self, g: Any) -> None:
        """Create a new network.

        Args:
            g: a NetworkX graph instance.
        """
        self.G = g
        for node_id in self.G.nodes:
            g.nodes[node_id]["agent"] = self.default_val()

    @property
    def agents(self) -> AgentSet:
        """Return an AgentSet with the agents in the space."""
        agents = []
        for node_id in self.G.nodes:
            entry = self.G.nodes[node_id]["agent"]
            if not entry:
                continue
            if not isinstance(entry, list):
                entry = [entry]
            for agent in entry:
                agents.append(agent)

        # getting the rng is a bit hacky because old style spaces don't have the rng
        try:
            rng = agents[0].random
        except IndexError:
            # there are no agents in the space
            rng = None
        return AgentSet(agents, random=rng)

    @staticmethod
    def default_val() -> list:
        """Default value for a new node."""
        return []

    @warn_if_agent_has_position_already
    def place_agent(self, agent: Agent, node_id: int) -> None:
        """Place an agent in a node."""
        self.G.nodes[node_id]["agent"].append(agent)
        agent.pos = node_id

    def get_neighborhood(
        self, node_id: int, include_center: bool = False, radius: int = 1
    ) -> list[int]:
        """Get all adjacent nodes within a certain radius.

        Args:
            node_id: node id for which to get neighborhood
            include_center: boolean to include node itself or not
            radius: size of neighborhood

        Returns:
            a list
        """
        if radius == 1:
            neighborhood = list(self.G.neighbors(node_id))
            if include_center:
                neighborhood.append(node_id)
        else:
            neighbors_with_distance = nx.single_source_shortest_path_length(
                self.G, node_id, radius
            )
            if not include_center:
                del neighbors_with_distance[node_id]
            neighbors_with_distance = sorted(
                neighbors_with_distance.items(), key=lambda item: item[1]
            )
            neighborhood = [node_id for node_id, _ in neighbors_with_distance]
        return neighborhood

    def get_neighbors(
        self, node_id: int, include_center: bool = False, radius: int = 1
    ) -> list[Agent]:
        """Get all agents in adjacent nodes (within a certain radius).

        Args:
            node_id: node id for which to get neighbors
            include_center: whether to include node itself or not
            radius: size of neighborhood in which to find neighbors

        Returns:
            list of agents in neighborhood.
        """
        neighborhood = self.get_neighborhood(node_id, include_center, radius)
        return self.get_cell_list_contents(neighborhood)

    def move_agent(self, agent: Agent, node_id: int) -> None:
        """Move an agent from its current node to a new node.

        Args:
            agent: agent instance
            node_id: id of node

        """
        self.remove_agent(agent)
        self.place_agent(agent, node_id)

    def remove_agent(self, agent: Agent) -> None:
        """Remove the agent from the network and set its pos attribute to None.

        Args:
            agent: agent instance

        """
        node_id = agent.pos
        self.G.nodes[node_id]["agent"].remove(agent)
        agent.pos = None

    def is_cell_empty(self, node_id: int) -> bool:
        """Returns a bool of the contents of a cell.

        Args:
            node_id: id of node

        """
        return self.G.nodes[node_id]["agent"] == self.default_val()

    def get_cell_list_contents(self, cell_list: list[int]) -> list[Agent]:
        """Returns a list of the agents contained in the nodes identified in `cell_list`.

        Nodes with empty content are excluded.

        Args:
            cell_list: list of cell ids.

        Returns:
            list of the agents contained in the nodes identified in `cell_list`.

        """
        return list(self.iter_cell_list_contents(cell_list))

    def get_all_cell_contents(self) -> list[Agent]:
        """Returns a list of all the agents in the network."""
        return self.get_cell_list_contents(self.G)

    def iter_cell_list_contents(self, cell_list: list[int]) -> Iterator[Agent]:
        """Returns an iterator of the agents contained in the nodes identified in `cell_list`.

        Nodes with empty content are excluded.

        Args:
            cell_list: list of cell ids.

        Returns:
            iterator of the agents contained in the nodes identified in `cell_list`.

        """
        return itertools.chain.from_iterable(
            self.G.nodes[node_id]["agent"]
            for node_id in itertools.filterfalse(self.is_cell_empty, cell_list)
        )
