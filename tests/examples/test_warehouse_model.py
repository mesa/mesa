"""Tests for the warehouse meta-agent example."""

from __future__ import annotations

import gc
import weakref

from mesa.examples import WarehouseModel
from mesa.examples.advanced.warehouse.agents import (
    InventoryAgent,
    RouteAgent,
    SensorAgent,
    WorkerAgent,
)
from mesa.examples.advanced.warehouse.make_warehouse import LOADING_DOCK_COORDS


def test_warehouse_model_uses_membership_backend():
    """Robot memberships should be mirrored into the backend and cleaned up."""
    model = WarehouseModel(rng=42)
    backend = model.membership_backend
    robot_type = model.robot_agent_type

    assert robot_type is not None

    robots = sorted(model.agents_by_type[robot_type], key=lambda agent: agent.unique_id)
    assert len(robots) == len(LOADING_DOCK_COORDS)

    expected_triplets = set()
    for robot in robots:
        route = robot.get_constituting_agent_instance(RouteAgent)
        sensor = robot.get_constituting_agent_instance(SensorAgent)
        worker = robot.get_constituting_agent_instance(WorkerAgent)

        expected_triplets.update(
            {
                (route.unique_id, robot.unique_id, "router"),
                (sensor.unique_id, robot.unique_id, "sensor"),
                (worker.unique_id, robot.unique_id, "worker"),
            }
        )

        assert backend.groups_of(route) == {robot.unique_id}
        assert backend.groups_of(sensor) == {robot.unique_id}
        assert backend.groups_of(worker) == {robot.unique_id}
        assert backend.relations_between(route, robot) == {"router"}
        assert backend.relations_between(sensor, robot) == {"sensor"}
        assert backend.relations_between(worker, robot) == {"worker"}

    assert backend.as_triplets() == expected_triplets
    backend.assert_invariants()

    before_step = backend.as_triplets()
    model.step()
    assert backend.as_triplets() == before_step
    backend.assert_invariants()

    ref = weakref.ref(model)
    model.remove_all_agents()

    assert backend.as_triplets() == set()
    backend.assert_invariants()

    del robots, robot, robot_type, expected_triplets, route, sensor, worker, before_step
    del model
    gc.collect()
    assert ref() is None


def test_warehouse_robot_completes_inventory_cycle():
    """A robot should complete the full inventory and loading workflow."""
    model = WarehouseModel(rng=42)
    backend = model.membership_backend
    robot_type = model.robot_agent_type

    assert robot_type is not None

    robot = sorted(model.agents_by_type[robot_type], key=lambda agent: agent.unique_id)[
        0
    ]
    item = min(
        model.agents_by_type[InventoryAgent],
        key=lambda agent: (
            abs(agent.cell.coordinate[0] - robot.cell.coordinate[0])
            + abs(agent.cell.coordinate[1] - robot.cell.coordinate[1]),
            agent.unique_id,
        ),
    )

    start_triplets = backend.as_triplets()
    start_quantity = item.quantity

    robot.initiate_task(item)
    robot.status = "inventory"

    for _ in range(200):
        if robot.status == "open":
            break
        robot.continue_task()
    else:
        raise AssertionError("Robot did not complete the inventory cycle")

    assert robot.status == "open"
    assert robot.carrying is None
    assert robot.item is None
    assert robot.path is None
    assert item.quantity == start_quantity - 1
    assert backend.as_triplets() == start_triplets
    backend.assert_invariants()
