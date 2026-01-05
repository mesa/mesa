"""Efficient storage and manipulation of cell properties across spaces.

PropertyLayers provide a way to associate properties with cells in a space efficiently.
The module includes:
- PropertyLayer class for managing grid-wide properties
- Property access descriptors for cells
- Batch operations for property modification
- Property-based cell selection
- Integration with numpy for efficient operations

This system separates property storage from cells themselves, enabling
fast bulk operations and sophisticated property-based behaviors while
maintaining an intuitive interface through cell attributes. Properties
can represent environmental factors, cell states, or any other grid-wide
attributes.
"""

import warnings
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import numpy as np

from mesa.discrete_space import Cell

Coordinate = Sequence[int]
T = TypeVar("T", bound=Cell)


class PropertyLayer:
    """A class representing a layer of properties in a two-dimensional grid.

    Each cell in the grid can store a value of a specified data type.

    Attributes:
        name: The name of the property layer.
        dimensions: The width of the grid (number of columns).
        data: A NumPy array representing the grid data.
    """

    propertylayer_experimental_warning_given = False

    def __init__(
        self, name: str, dimensions: Sequence[int], default_value=0.0, dtype=float
    ):
        """Initializes a new PropertyLayer instance.

        Args:
            name: The name of the property layer.
            dimensions: the dimensions of the property layer.
            default_value: The default value to initialize each cell in the grid. Should ideally
                           be of the same type as specified by the dtype parameter.
            dtype (data-type, optional): The desired data-type for the grid's elements. Default is float.

        Notes:
            An exception is raised if the default_value is not of a type compatible with dtype.
            A UserWarning is raised if the conversion would results in a loss of precision.
            The dtype parameter can accept both Python data types (like bool, int or float) and NumPy data types
            (like np.int64 or np.float64).
        """
        self.name = name
        self.dimensions = dimensions

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

        # Public attribute exposing the raw data
        self.data = np.full(self.dimensions, default_value, dtype=dtype)

    @classmethod
    def from_data(cls, name: str, data: np.ndarray):
        """Create a property layer directly from an existing NumPy array.

        Args:
            name: The name of the property layer.
            data: A NumPy array representing the grid data.
        """
        layer = cls(
            name,
            data.shape,
            default_value=data.flat[0],
            dtype=data.dtype.type,
        )
        layer.data = data
        return layer

    # NumPy Array Interface

    def __array__(self, dtype=None):
        """Allow the layer to be passed directly to NumPy functions."""
        return np.asarray(self.data, dtype=dtype)

    def __getitem__(self, key):
        """Allow direct indexing (e.g., layer[0, 0])."""
        return self.data[key]

    def __setitem__(self, key, value):
        """Allow direct item assignment (e.g., layer[0, 0] = 5)."""
        self.data[key] = value

    def __iter__(self):
        """Allow iteration over the data."""
        return iter(self.data)

    def __len__(self):
        """Return the length of the data."""
        return len(self.data)

    # In-place Arithmetic Operations

    def __iadd__(self, other):
        """In-place addition."""
        self.data += other
        return self

    def __isub__(self, other):
        """In-place subtraction."""
        self.data -= other
        return self

    def __imul__(self, other):
        """In-place multiplication."""
        self.data *= other
        return self

    def __itruediv__(self, other):
        """In-place true division."""
        self.data /= other
        return self

    def __ifloordiv__(self, other):
        """In-place floor division."""
        self.data //= other
        return self

    def __ipow__(self, other):
        """In-place power."""
        self.data **= other
        return self

    # Deprecated Methods

    def set_cells(self, value, condition: Callable | None = None):
        """Perform a batch update either on the entire grid or conditionally, in-place.

        Args:
            value: The value to be used for the update.
            condition: (Optional) A callable that returns a boolean array when applied to the data.
        """
        warnings.warn(
            "set_cells is deprecated and will be removed in a future version. "
            "Use direct NumPy indexing on the layer object instead (e.g. layer[:] = value or layer[mask] = value).",
            DeprecationWarning,
            stacklevel=2,
        )
        if condition is None:
            self.data[:] = value
        else:
            mask = condition(self.data)
            self.data[mask] = value

    def modify_cells(
        self,
        operation: Callable,
        value=None,
        condition: Callable | None = None,
    ):
        """Modify cells using an operation, which can be a lambda function or a NumPy ufunc.

        Args:
            operation: A function to apply. Can be a lambda function or a NumPy ufunc.
            value: The value to be used if the operation is a NumPy ufunc. Ignored for lambda functions.
            condition: (Optional) A callable that returns a boolean array when applied to the data.
        """
        warnings.warn(
            "modify_cells is deprecated and will be removed in a future version. "
            "Use direct NumPy operations on the layer object instead (e.g. layer += 1).",
            DeprecationWarning,
            stacklevel=2,
        )
        mask = condition(self.data) if condition is not None else slice(None)

        target_data = self.data[mask]

        if isinstance(operation, np.ufunc) and operation.nargs > 1:
            if value is None:
                raise ValueError("This ufunc requires an additional input value.")
            self.data[mask] = operation(target_data, value)
        else:
            vectorized_op = np.vectorize(operation)
            self.data[mask] = vectorized_op(target_data)

    def select_cells(self, condition: Callable, return_list=True):
        """Find cells that meet a specified condition using NumPy's boolean indexing.

        Args:
            condition: A callable that returns a boolean array when applied to the data.
            return_list: (Optional) If True, return a list of (x, y) tuples. Otherwise, return a boolean array.

        Returns:
            A list of (x, y) tuples or a boolean array.
        """
        warnings.warn(
            "select_cells is deprecated and will be removed in a future version. "
            "Use np.argwhere(condition(layer)) or boolean masks instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        mask = condition(self.data)
        if return_list:
            return list(zip(*np.where(mask)))
        else:
            return mask

    def aggregate(self, operation: Callable):
        """Perform an aggregate operation (e.g., sum, mean) on a property across all cells.

        Args:
            operation: A function to apply. Can be a lambda function or a NumPy ufunc.
        """
        warnings.warn(
            "aggregate is deprecated and will be removed in a future version. "
            "Use NumPy aggregate functions directly on the layer object (e.g. np.mean(layer)).",
            DeprecationWarning,
            stacklevel=2,
        )
        return operation(self.data)


class HasPropertyLayers:
    """Mixin class to add property layer functionality to Grids.

    Property layers can be added to a grid using create_property_layer or add_property_layer.
    Once created, property layers can be accessed as attributes.
    """

    def __init__(self, *args, **kwargs):
        """Initialize a HasPropertyLayers instance."""
        super().__init__(*args, **kwargs)
        self._mesa_property_layers = {}

    def create_property_layer(
        self,
        name: str,
        default_value=0.0,
        dtype=float,
    ):
        """Add a property layer to the grid.

        Args:
            name: The name of the property layer.
            default_value: The default value of the property layer.
            dtype: The data type of the property layer.

        Returns:
              The created PropertyLayer instance.
        """
        layer = PropertyLayer(
            name, self.dimensions, default_value=default_value, dtype=dtype
        )
        self.add_property_layer(layer)
        return layer

    def add_property_layer(self, layer: PropertyLayer):
        """Add a predefined property layer to the grid.

        Args:
            layer: The property layer to add.

        Raises:
            ValueError: If dimensions do not match or if the layer name already exists.
        """
        if layer.dimensions != self.dimensions:
            raise ValueError(
                "Dimensions of property layer do not match the dimensions of the grid"
            )
        if layer.name in self._mesa_property_layers:
            raise ValueError(f"Property layer {layer.name} already exists.")
        if (
            layer.name in self.cell_klass.__slots__
            or layer.name in self.cell_klass.__dict__
        ):
            raise ValueError(
                f"Property layer {layer.name} clashes with existing attribute in {self.cell_klass.__name__}"
            )

        self._mesa_property_layers[layer.name] = layer
        setattr(self.cell_klass, layer.name, PropertyDescriptor(layer))
        self.cell_klass._mesa_properties.add(layer.name)

    def remove_property_layer(self, property_name: str):
        """Remove a property layer from the grid.

        Args:
            property_name: The name of the property layer to remove.
        """
        del self._mesa_property_layers[property_name]
        delattr(self.cell_klass, property_name)
        self.cell_klass._mesa_properties.remove(property_name)

    def set_property(
        self, property_name: str, value, condition: Callable[[T], bool] | None = None
    ):
        """Set the value of a property for all cells in the grid.

        Args:
            property_name: The name of the property to set.
            value: The value to set.
            condition: A function that takes a cell and returns a boolean.
        """
        # Refactored to bypass deprecated set_cells
        layer = self._mesa_property_layers[property_name]
        if condition is None:
            layer.data[:] = value
        else:
            mask = np.vectorize(condition)(layer.data)
            layer.data[mask] = value

    def modify_properties(
        self,
        property_name: str,
        operation: Callable,
        value: Any = None,
        condition: Callable[[T], bool] | None = None,
    ):
        """Modify the values of a specific property for all cells in the grid.

        Args:
            property_name: The name of the property to modify.
            operation: The operation to perform.
            value: The value to use in the operation.
            condition: A function that takes a cell and returns a boolean.
        """
        # Refactored to bypass deprecated modify_cells
        layer = self._mesa_property_layers[property_name]

        mask = slice(None) if condition is None else np.vectorize(condition)(layer.data)

        if isinstance(operation, np.ufunc) and operation.nargs > 1:
            if value is None:
                raise ValueError("This ufunc requires an additional input value.")
            layer.data[mask] = operation(layer.data[mask], value)
        else:
            vectorized_op = np.vectorize(operation)
            layer.data[mask] = vectorized_op(layer.data[mask])

    def get_neighborhood_mask(
        self, coordinate: Coordinate, include_center: bool = True, radius: int = 1
    ) -> np.ndarray:
        """Generate a boolean mask representing the neighborhood.

        Args:
            coordinate: Center of the neighborhood.
            include_center: Include the central cell in the neighborhood.
            radius: The radius of the neighborhood.

        Returns:
            np.ndarray: A boolean mask representing the neighborhood.
        """
        cell = self._cells[coordinate]
        neighborhood = cell.get_neighborhood(
            include_center=include_center, radius=radius
        )
        mask = np.zeros(self.dimensions, dtype=bool)

        coords = np.array([c.coordinate for c in neighborhood])
        indices = [coords[:, i] for i in range(coords.shape[1])]
        mask[*indices] = True
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
        combined_mask = np.ones(self.dimensions, dtype=bool)

        # Apply the masks
        if masks is not None:
            if isinstance(masks, list):
                for mask in masks:
                    combined_mask = np.logical_and(combined_mask, mask)
            else:
                combined_mask = np.logical_and(combined_mask, masks)

        # Apply the empty mask if only_empty is True      
        if only_empty:
            combined_mask = np.logical_and(
                combined_mask, self._mesa_property_layers["empty"]
            )

        # Apply conditions
        if conditions:
            for prop_name, condition in conditions.items():
                prop_layer = self._mesa_property_layers[prop_name].data
                prop_mask = condition(prop_layer)
                combined_mask = np.logical_and(combined_mask, prop_mask)

        # Apply extreme values
        if extreme_values:
            for property_name, mode in extreme_values.items():
                prop_values = self._mesa_property_layers[property_name].data
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

        if return_list:
            selected_cells = list(zip(*np.where(combined_mask)))
            return selected_cells
        else:
            return combined_mask

    def __getattr__(self, name: str) -> Any:  # noqa: D105
        try:
            return self._mesa_property_layers[name]
        except KeyError as e:
            raise AttributeError(
                f"'{type(self).__name__}' object has no property layer called '{name}'"
            ) from e

    def __setattr__(self, key, value):  # noqa: D105
        # We must carefully check if _mesa_property_layers exists to avoid recursion
        try:
            layers = self.__dict__.get("_mesa_property_layers")
        except KeyError:
            super().__setattr__(key, value)
            return

        if layers and key in layers:
            raise AttributeError(
                f"'{type(self).__name__}' object already has a property layer with name '{key}'"
            )
        else:
            super().__setattr__(key, value)


class PropertyDescriptor:
    """Descriptor for giving cells attribute-like access to values defined in property layers."""

    def __init__(self, property_layer: PropertyLayer):  # noqa: D107
        self.layer: PropertyLayer = property_layer

    def __get__(self, instance: Cell, owner):  # noqa: D105
        return self.layer.data[instance.coordinate]

    def __set__(self, instance: Cell, value):  # noqa: D105
        self.layer.data[instance.coordinate] = value