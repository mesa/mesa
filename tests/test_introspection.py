"""Tests for mesa.visualization.introspection."""

import unittest

import mesa
from mesa.visualization.introspection import ModelSchema, describe_model


class MinimalModel(mesa.Model):
    """Simplest possible model, used to test the empty schema shape."""


class TestDescribeModel(unittest.TestCase):
    """Tests for describe_model()'s empty-shape skeleton behavior."""

    def test_returns_model_schema(self):
        """describe_model() should return a ModelSchema instance."""
        schema = describe_model(MinimalModel())
        self.assertIsInstance(schema, ModelSchema)

    def test_empty_shape_defaults(self):
        """With no population logic yet, all fields should be empty."""
        schema = describe_model(MinimalModel())
        self.assertEqual(schema.agent_types, [])
        self.assertIsNone(schema.space)
        self.assertEqual(schema.params, {})
        self.assertEqual(schema.reporters, {})
