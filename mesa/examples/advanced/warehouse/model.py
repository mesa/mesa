"""Warehouse meta-agent example built on the membership backend."""

from __future__ import annotations

import mesa
from mesa.discrete_space import OrthogonalMooreGrid
from mesa.discrete_space.cell_agent import CellAgent
from mesa.examples.advanced.warehouse.agents import (
    InventoryAgent,
    RouteAgent,
    SensorAgent,
    WorkerAgent,
)
from mesa.examples.advanced.warehouse.make_warehouse import (
    CHARGING_STATION_COORDS,
    LOADING_DOCK_COORDS,
    make_warehouse,
)
from mesa.experimental.meta_agents.backend import MembershipBackend
from mesa.experimental.meta_agents.meta_agent import MetaAgent, create_meta_agent


class WarehouseModel(mesa.Model):
    """Model for simulating warehouse robots assembled from sub-agents."""

    def __init__(self, rng=42):
        """Create the warehouse, inventory, and robot meta-agents."""
        super().__init__(rng=rng)
        self.inventory = {}
        self.membership_backend = MembershipBackend()

        layout = make_warehouse(rng=self.random)
        self.warehouse = OrthogonalMooreGrid(
            (layout.shape[0], layout.shape[1], layout.shape[2]),
            torus=False,
            capacity=1,
            random=self.random,
        )

        # Inventory agents live in the storage rows of the warehouse.
        for row in range(2, layout.shape[0] - 1, 3):
            for col in range(layout.shape[1]):
                for height in range(layout.shape[2]):
                    item = layout[row][col][height]
                    if item.strip():
                        InventoryAgent(self, self.warehouse[row, col, height], item)

        self.robot_agent_type: type | None = None
        self.RobotAgent = None

        # One robot is created per loading dock / charging station pair.
        for loading_dock, charging_station in zip(
            LOADING_DOCK_COORDS, CHARGING_STATION_COORDS, strict=True
        ):
            router = RouteAgent(self)
            sensor = SensorAgent(self)
            worker = WorkerAgent(
                self,
                self.warehouse[loading_dock],
                self.warehouse[charging_station],
            )

            def remove_robot(robot):
                """Remove robot memberships even if the meta-agent teardown fails."""
                try:
                    MetaAgent.remove(robot)
                finally:
                    robot.model.membership_backend.remove_group(robot)

            meta = create_meta_agent(
                self,
                "RobotAgent",
                [router, sensor, worker],
                CellAgent,
                meta_attributes={
                    "cell": self.warehouse[charging_station],
                    "status": "open",
                },
                meta_methods={"remove": remove_robot},
                assume_constituting_agent_attributes=True,
                assume_constituting_agent_methods=True,
            )

            if meta is None:
                continue

            if self.robot_agent_type is None:
                self.robot_agent_type = type(meta)

            self.RobotAgent = meta
            self._record_robot_memberships(meta)

    def _record_robot_memberships(self, robot) -> None:
        """Mirror a robot's constituting relationships into the backend."""
        self.membership_backend.bulk_add(
            [
                (
                    robot.get_constituting_agent_instance(RouteAgent),
                    robot,
                    "router",
                ),
                (
                    robot.get_constituting_agent_instance(SensorAgent),
                    robot,
                    "sensor",
                ),
                (
                    robot.get_constituting_agent_instance(WorkerAgent),
                    robot,
                    "worker",
                ),
            ]
        )

    def central_move(self, robot):
        """Delegate path execution to the robot's worker role."""
        robot.move(robot.cell.coordinate, robot.path)

    def step(self):
        """Advance the model by one step."""
        if self.robot_agent_type is None:
            return

        for robot in self.agents_by_type[self.robot_agent_type]:
            agent_list = self.agents_by_type[InventoryAgent].to_list()

            if robot.status == "open":
                item = self.random.choice(agent_list)
                if item.quantity > 0:
                    robot.initiate_task(item)
                    robot.status = "inventory"
                    self.central_move(robot)

            else:
                robot.continue_task()
