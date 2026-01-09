"""Tests for the PropertyLayer module in discrete_space."""

import random
import warnings

import numpy as np
import pytest

from mesa.discrete_space.cell import Cell
from mesa.discrete_space.grid import OrthogonalMooreGrid
from mesa.discrete_space.property_layer import (
    HasPropertyLayers,
    PropertyDescriptor,
    PropertyLayer,
    ufunc_requires_additional_input,
)


class TestPropertyLayer:
    """Tests for the PropertyLayer class."""

    def test_property_layer_initialization(self):
        """Test PropertyLayer basic initialization."""
        layer = PropertyLayer("test", (10, 10))
        
        assert layer.name == "test"
        assert layer.dimensions == (10, 10)
        assert layer.data.shape == (10, 10)

    def test_property_layer_default_value(self):
        """Test PropertyLayer with default value."""
        layer = PropertyLayer("test", (5, 5), default_value=42.0)
        
        assert np.all(layer.data == 42.0)

    def test_property_layer_custom_dtype(self):
        """Test PropertyLayer with custom dtype."""
        layer = PropertyLayer("test", (5, 5), default_value=1, dtype=int)
        
        assert layer.data.dtype == int

    def test_property_layer_bool_dtype(self):
        """Test PropertyLayer with boolean dtype."""
        layer = PropertyLayer("test", (5, 5), default_value=True, dtype=bool)
        
        assert layer.data.dtype == bool
        assert np.all(layer.data == True)

    def test_property_layer_precision_warning(self):
        """Test PropertyLayer warns on precision loss."""
        with pytest.warns(UserWarning, match="lose precision"):
            PropertyLayer("test", (5, 5), default_value=0.5, dtype=int)

    def test_property_layer_incompatible_type_raises_error(self):
        """Test PropertyLayer raises error on incompatible types."""
        with pytest.raises(TypeError, match="incompatible"):
            PropertyLayer("test", (5, 5), default_value="invalid", dtype=int)

    def test_property_layer_from_data(self):
        """Test PropertyLayer.from_data class method."""
        data = np.array([[1, 2], [3, 4]])
        layer = PropertyLayer.from_data("test", data)
        
        assert layer.name == "test"
        assert np.array_equal(layer.data, data)
        assert layer.dimensions == (2, 2)

    def test_property_layer_set_cells_all(self):
        """Test setting all cells in PropertyLayer."""
        layer = PropertyLayer("test", (5, 5), default_value=0.0)
        layer.set_cells(10.0)
        
        assert np.all(layer.data == 10.0)

    def test_property_layer_set_cells_conditional(self):
        """Test setting cells conditionally in PropertyLayer."""
        layer = PropertyLayer("test", (5, 5), default_value=0.0)
        layer.data[2, 2] = 5.0
        
        layer.set_cells(100.0, condition=lambda x: x == 5.0)
        
        assert layer.data[2, 2] == 100.0
        assert layer.data[0, 0] == 0.0

    def test_property_layer_modify_cells_lambda(self):
        """Test modifying cells with lambda function."""
        layer = PropertyLayer("test", (3, 3), default_value=2.0)
        layer.modify_cells(lambda x: x * 2)
        
        assert np.all(layer.data == 4.0)

    def test_property_layer_modify_cells_ufunc(self):
        """Test modifying cells with numpy ufunc."""
        layer = PropertyLayer("test", (3, 3), default_value=2.0)
        layer.modify_cells(np.add, value=3.0)
        
        assert np.all(layer.data == 5.0)

    def test_property_layer_modify_cells_ufunc_no_value_raises(self):
        """Test modifying cells with binary ufunc without value raises error."""
        layer = PropertyLayer("test", (3, 3), default_value=2.0)
        
        with pytest.raises(ValueError, match="requires an additional input"):
            layer.modify_cells(np.add)

    def test_property_layer_modify_cells_unary_ufunc(self):
        """Test modifying cells with unary ufunc."""
        layer = PropertyLayer("test", (3, 3), default_value=-2.0)
        layer.modify_cells(np.abs)
        
        assert np.all(layer.data == 2.0)

    def test_property_layer_modify_cells_conditional(self):
        """Test modifying cells conditionally."""
        layer = PropertyLayer("test", (3, 3), default_value=1.0)
        layer.data[1, 1] = 5.0
        
        layer.modify_cells(lambda x: x * 10, condition=lambda x: x > 3)
        
        assert layer.data[1, 1] == 50.0
        assert layer.data[0, 0] == 1.0

    def test_property_layer_select_cells_return_list(self):
        """Test select_cells returning list of coordinates."""
        layer = PropertyLayer("test", (3, 3), default_value=0.0)
        layer.data[1, 1] = 1.0
        layer.data[2, 0] = 1.0
        
        selected = layer.select_cells(lambda x: x == 1.0, return_list=True)
        
        assert (1, 1) in selected
        assert (2, 0) in selected
        assert len(selected) == 2

    def test_property_layer_select_cells_return_array(self):
        """Test select_cells returning boolean array."""
        layer = PropertyLayer("test", (3, 3), default_value=0.0)
        layer.data[1, 1] = 1.0
        
        mask = layer.select_cells(lambda x: x == 1.0, return_list=False)
        
        assert mask.dtype == bool
        assert mask[1, 1] == True
        assert mask[0, 0] == False

    def test_property_layer_aggregate_sum(self):
        """Test aggregate operation with sum."""
        layer = PropertyLayer("test", (3, 3), default_value=1.0)
        
        result = layer.aggregate(np.sum)
        
        assert result == 9.0

    def test_property_layer_aggregate_mean(self):
        """Test aggregate operation with mean."""
        layer = PropertyLayer("test", (3, 3), default_value=3.0)
        
        result = layer.aggregate(np.mean)
        
        assert result == 3.0

    def test_property_layer_3d_dimensions(self):
        """Test PropertyLayer with 3D dimensions."""
        layer = PropertyLayer("test", (2, 3, 4), default_value=1.0)
        
        assert layer.data.shape == (2, 3, 4)


class TestHasPropertyLayers:
    """Tests for the HasPropertyLayers mixin."""

    def test_create_property_layer(self):
        """Test creating a property layer on grid."""
        rng = random.Random(42)
        grid = OrthogonalMooreGrid((5, 5), random=rng)
        
        layer = grid.create_property_layer("test_layer", default_value=0.0)
        
        assert "test_layer" in grid._mesa_property_layers
        assert isinstance(layer, PropertyLayer)

    def test_add_property_layer(self):
        """Test adding a property layer to grid."""
        rng = random.Random(42)
        grid = OrthogonalMooreGrid((5, 5), random=rng)
        layer = PropertyLayer("custom_layer", (5, 5), default_value=1.0)
        
        grid.add_property_layer(layer)
        
        assert "custom_layer" in grid._mesa_property_layers

    def test_add_property_layer_dimension_mismatch_raises(self):
        """Test adding property layer with wrong dimensions raises error."""
        rng = random.Random(42)
        grid = OrthogonalMooreGrid((5, 5), random=rng)
        layer = PropertyLayer("wrong_dims", (3, 3), default_value=1.0)
        
        with pytest.raises(ValueError, match="Dimensions"):
            grid.add_property_layer(layer)

    def test_add_property_layer_duplicate_name_raises(self):
        """Test adding property layer with duplicate name raises error."""
        rng = random.Random(42)
        grid = OrthogonalMooreGrid((5, 5), random=rng)
        grid.create_property_layer("duplicate", default_value=0.0)
        
        layer = PropertyLayer("duplicate", (5, 5), default_value=1.0)
        
        with pytest.raises(ValueError, match="already exists"):
            grid.add_property_layer(layer)

    def test_remove_property_layer(self):
        """Test removing a property layer."""
        rng = random.Random(42)
        grid = OrthogonalMooreGrid((5, 5), random=rng)
        grid.create_property_layer("to_remove", default_value=0.0)
        
        grid.remove_property_layer("to_remove")
        
        assert "to_remove" not in grid._mesa_property_layers

    def test_set_property(self):
        """Test setting property values."""
        rng = random.Random(42)
        grid = OrthogonalMooreGrid((5, 5), random=rng)
        grid.create_property_layer("test_prop", default_value=0.0)
        
        grid.set_property("test_prop", 5.0)
        
        assert np.all(grid._mesa_property_layers["test_prop"].data == 5.0)

    def test_modify_properties(self):
        """Test modifying property values."""
        rng = random.Random(42)
        grid = OrthogonalMooreGrid((5, 5), random=rng)
        grid.create_property_layer("test_prop", default_value=2.0)
        
        grid.modify_properties("test_prop", lambda x: x * 3)
        
        assert np.all(grid._mesa_property_layers["test_prop"].data == 6.0)

    def test_get_neighborhood_mask(self):
        """Test getting neighborhood mask."""
        rng = random.Random(42)
        grid = OrthogonalMooreGrid((5, 5), random=rng)
        
        mask = grid.get_neighborhood_mask((2, 2), include_center=True, radius=1)
        
        assert mask.shape == (5, 5)
        assert mask[2, 2] == True  # Center
        assert mask[1, 1] == True  # Neighbor
        assert mask[0, 0] == False  # Not in neighborhood

    def test_select_cells_with_conditions(self):
        """Test selecting cells based on conditions."""
        rng = random.Random(42)
        grid = OrthogonalMooreGrid((5, 5), random=rng)
        grid.create_property_layer("test_prop", default_value=0.0)
        grid._mesa_property_layers["test_prop"].data[2, 2] = 10.0
        
        selected = grid.select_cells(conditions={"test_prop": lambda x: x > 5})
        
        assert (2, 2) in selected

    def test_select_cells_with_extreme_values_highest(self):
        """Test selecting cells with highest value."""
        rng = random.Random(42)
        grid = OrthogonalMooreGrid((5, 5), random=rng)
        grid.create_property_layer("test_prop", default_value=0.0)
        grid._mesa_property_layers["test_prop"].data[3, 3] = 100.0
        
        selected = grid.select_cells(extreme_values={"test_prop": "highest"})
        
        assert (3, 3) in selected

    def test_select_cells_with_extreme_values_lowest(self):
        """Test selecting cells with lowest value."""
        rng = random.Random(42)
        grid = OrthogonalMooreGrid((5, 5), random=rng)
        grid.create_property_layer("test_prop", default_value=10.0)
        grid._mesa_property_layers["test_prop"].data[1, 1] = 0.0
        
        selected = grid.select_cells(extreme_values={"test_prop": "lowest"})
        
        assert (1, 1) in selected

    def test_select_cells_invalid_extreme_mode_raises(self):
        """Test selecting cells with invalid extreme mode raises error."""
        rng = random.Random(42)
        grid = OrthogonalMooreGrid((5, 5), random=rng)
        grid.create_property_layer("test_prop", default_value=0.0)
        
        with pytest.raises(ValueError, match="Invalid mode"):
            grid.select_cells(extreme_values={"test_prop": "invalid"})

    def test_select_cells_with_mask(self):
        """Test selecting cells with a mask."""
        rng = random.Random(42)
        grid = OrthogonalMooreGrid((5, 5), random=rng)
        
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        mask[3, 3] = True
        
        selected = grid.select_cells(masks=mask)
        
        assert len(selected) == 2

    def test_select_cells_return_mask(self):
        """Test selecting cells returning mask."""
        rng = random.Random(42)
        grid = OrthogonalMooreGrid((5, 5), random=rng)
        
        result = grid.select_cells(return_list=False)
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == bool

    def test_attribute_access_property_layer(self):
        """Test accessing property layer as attribute."""
        rng = random.Random(42)
        grid = OrthogonalMooreGrid((5, 5), random=rng)
        grid.create_property_layer("my_layer", default_value=0.0)
        
        layer = grid.my_layer
        
        assert isinstance(layer, PropertyLayer)
        assert layer.name == "my_layer"

    def test_attribute_access_nonexistent_raises(self):
        """Test accessing non-existent property layer raises error."""
        rng = random.Random(42)
        grid = OrthogonalMooreGrid((5, 5), random=rng)
        
        with pytest.raises(AttributeError, match="no property layer"):
            _ = grid.nonexistent_layer

    def test_setattr_conflict_with_layer_raises(self):
        """Test setting attribute that conflicts with layer raises error."""
        rng = random.Random(42)
        grid = OrthogonalMooreGrid((5, 5), random=rng)
        grid.create_property_layer("conflict", default_value=0.0)
        
        with pytest.raises(AttributeError, match="already has a property layer"):
            grid.conflict = "some_value"


class TestPropertyDescriptor:
    """Tests for the PropertyDescriptor class."""

    def test_property_descriptor_get(self):
        """Test PropertyDescriptor get value."""
        rng = random.Random(42)
        grid = OrthogonalMooreGrid((5, 5), random=rng)
        grid.create_property_layer("test", default_value=0.0)
        grid._mesa_property_layers["test"].data[2, 2] = 42.0
        
        cell = grid._cells[(2, 2)]
        value = cell.test
        
        assert value == 42.0

    def test_property_descriptor_set(self):
        """Test PropertyDescriptor set value."""
        rng = random.Random(42)
        grid = OrthogonalMooreGrid((5, 5), random=rng)
        grid.create_property_layer("test", default_value=0.0)
        
        cell = grid._cells[(2, 2)]
        cell.test = 99.0
        
        assert grid._mesa_property_layers["test"].data[2, 2] == 99.0


class TestUfuncRequiresAdditionalInput:
    """Tests for the ufunc_requires_additional_input function."""

    def test_binary_ufunc(self):
        """Test binary ufunc detection."""
        assert ufunc_requires_additional_input(np.add) == True
        assert ufunc_requires_additional_input(np.multiply) == True

    def test_unary_ufunc(self):
        """Test unary ufunc detection."""
        assert ufunc_requires_additional_input(np.abs) == False
        assert ufunc_requires_additional_input(np.sqrt) == False
