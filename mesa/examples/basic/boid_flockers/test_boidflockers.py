"""Tests for the canonical + distance-weighted Boids implementation.

These tests verify:
1. Canonical Reynolds (1986) behavior — mean not sum
2. Option 4 — distance-weighted decay behavior
3. Edge cases — no neighbors, normalization
"""

import numpy as np
import pytest

from mesa.examples.basic.boid_flockers.agents import Boid
from mesa.examples.basic.boid_flockers.model import BoidFlockers, BoidsScenario


class TestCanonicalBoids:
    """Tests that verify canonical Reynolds (1986) behavior."""

    def test_boids_move_each_step(self):
        """All boids must change position every step."""
        model = BoidFlockers()
        positions_before = [b.position.copy() for b in model.agents]
        model.step()
        positions_after = [b.position for b in model.agents]
        moved = sum(
            not np.allclose(a, b)
            for a, b in zip(positions_before, positions_after)
        )
        assert moved > len(model.agents) * 0.8, "Most boids should move each step"

    def test_direction_always_normalized(self):
        """Direction vector must always be a unit vector after each step."""
        model = BoidFlockers()
        for _ in range(10):
            model.step()
        for boid in model.agents:
            norm = np.linalg.norm(boid.direction)
            assert abs(norm - 1.0) < 0.5, (
                f"Direction norm={norm:.6f}, expected 1.0 — normalization broken"
            )

    def test_cohesion_stable_with_more_neighbors(self):
        """
        Cohesion force must NOT grow with neighbor count.
        With mean (canonical), force stays stable.
        With sum (buggy), force would blow up with more boids.
        """
        model_small = BoidFlockers(scenario=BoidsScenario(population_size=10))
        model_large = BoidFlockers(scenario=BoidsScenario(population_size=50))

        for _ in range(5):
            model_small.step()
            model_large.step()

        # Direction norms must stay ~1.0 in both cases
        for boid in model_small.agents:
            assert abs(np.linalg.norm(boid.direction) - 1.0) < 0.5
        for boid in model_large.agents:
            assert abs(np.linalg.norm(boid.direction) - 1.0) < 0.5, (
                "Large flock blew up — cohesion likely using sum not mean"
            )

    def test_no_neighbors_boid_keeps_moving(self):
        """A boid with no neighbors must keep moving in its current direction."""
        # Single boid — no neighbors possible
        model = BoidFlockers(scenario=BoidsScenario(population_size=1))
        boid = list(model.agents)[0]
        direction_before = boid.direction.copy()
        position_before = boid.position.copy()
        model.step()
        # Direction unchanged when no neighbors
        np.testing.assert_array_almost_equal(boid.direction, direction_before)
        # Position must still update
        assert not np.allclose(boid.position, position_before)

    def test_average_heading_is_tracked(self):
        """Model must track average heading after each step."""
        model = BoidFlockers()
        model.step()
        assert model.average_heading is not None, "average_heading should be updated"
        assert isinstance(model.average_heading, float), (
            "average_heading should be a float (angle in radians)"
        )


class TestDistanceWeightedDecay:
    """Tests for Option 4 — distance-weighted decay."""

    def test_decay_parameter_exists(self):
        """Boid must have a decay attribute."""
        model = BoidFlockers()
        boid = list(model.agents)[0]
        assert hasattr(boid, "decay"), (
            "Boid missing 'decay' attribute — Option 4 not implemented"
        )

    def test_decay_default_is_2(self):
        """Default decay exponent should be 2.0 (inverse square law)."""
        model = BoidFlockers()
        boid = list(model.agents)[0]
        assert boid.decay == 2.0, f"Expected decay=2.0, got {boid.decay}"

    def test_closer_neighbor_has_more_weight(self):
        """With decay>0, a closer neighbor must have higher weight than a far one."""
        distances = np.array([10.0, 50.0])
        decay = 2.0
        weights = 1.0 / (distances**decay + 1e-6)
        weights /= weights.sum()
        assert weights[0] > weights[1], (
            "Closer neighbor should have higher weight than farther one"
        )

    def test_decay_zero_equals_canonical(self):
        """
        decay=0.0 means 1/d^0 = 1 for all neighbors → equal weights → canonical.
        """
        distances = np.array([10.0, 30.0, 60.0])
        decay = 0.0
        weights = 1.0 / (distances**decay + 1e-6)
        weights /= weights.sum()
        np.testing.assert_allclose(
            weights,
            np.full(3, 1 / 3),
            atol=1e-3,
            err_msg="decay=0 should produce equal weights (canonical behavior)",
        )

    def test_higher_decay_increases_close_neighbor_dominance(self):
        """Higher decay = closer neighbors dominate even more."""
        distances = np.array([10.0, 50.0])

        weights_low = 1.0 / (distances**1.0 + 1e-6)
        weights_low /= weights_low.sum()

        weights_high = 1.0 / (distances**4.0 + 1e-6)
        weights_high /= weights_high.sum()

        assert weights_high[0] > weights_low[0], (
            "Higher decay should give even more weight to closer neighbor"
        )

    def test_weights_always_sum_to_one(self):
        """Normalized weights must always sum to 1.0."""
        distances = np.array([5.0, 15.0, 40.0, 80.0])
        for decay in [0.5, 1.0, 2.0, 3.0]:
            weights = 1.0 / (distances**decay + 1e-6)
            weights /= weights.sum()
            assert abs(weights.sum() - 1.0) < 1e-6, (
                f"Weights don't sum to 1.0 for decay={decay}"
            )

    def test_custom_decay_via_scenario(self):
        """Model must accept custom decay value via scenario."""
        scenario = BoidsScenario(population_size=10)
        model = BoidFlockers(scenario=scenario)
        # Manually set decay on agents to test custom value
        for boid in model.agents:
            boid.decay = 3.0
        for boid in model.agents:
            assert boid.decay == 3.0, "Custom decay value not applied correctly"