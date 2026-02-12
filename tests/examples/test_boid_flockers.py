import numpy as np

from mesa.examples.basic.boid_flockers.model import BoidFlockers


def test_boid_cohesion_steers_to_local_centroid():
    model = BoidFlockers(
        population_size=3,
        width=100,
        height=100,
        speed=0,
        vision=30,
        separation=1,
        cohere=1.0,
        separate=0.0,
        match=0.0,
        rng=42,
    )
    boid, neighbor_a, neighbor_b = list(model.agents)

    boid.position = np.array([10.0, 10.0])
    neighbor_a.position = np.array([20.0, 10.0])
    neighbor_b.position = np.array([10.0, 20.0])

    boid.direction = np.array([0.0, 0.0])
    neighbor_a.direction = np.array([1.0, 0.0])
    neighbor_b.direction = np.array([1.0, 0.0])

    boid.step()

    expected_direction = np.array([1.0, 1.0]) / np.sqrt(2.0)
    np.testing.assert_allclose(boid.direction, expected_direction, atol=1e-8)
    assert len(boid.neighbors) == 2
