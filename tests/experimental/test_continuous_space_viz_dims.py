import numpy as np
import pytest
from random import Random

from mesa.experimental.continuous_space import ContinuousSpace


def test_continuous_space_requires_at_least_2_dimensions():
    dims = np.array([[0.0, 1.0]])
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        ContinuousSpace(dims, random=Random(1))


@pytest.mark.parametrize("viz_dims", [(0,), (0, 1, 2)])
def test_viz_dims_must_be_length_2(viz_dims):
    dims = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="viz_dims must contain exactly two distinct dimensions"):
        ContinuousSpace(dims, viz_dims=viz_dims, random=Random(1))


def test_viz_dims_must_be_distinct():
    dims = np.array([[0.0, 1.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="viz_dims must contain exactly two distinct dimensions"):
        ContinuousSpace(dims, viz_dims=(0, 0), random=Random(1))


@pytest.mark.parametrize("viz_dims", [(0, 3), (-1, 1)])
def test_viz_dims_must_be_in_range(viz_dims):
    dims = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match=r"viz_dims must be within"):
        ContinuousSpace(dims, viz_dims=viz_dims, random=Random(1))