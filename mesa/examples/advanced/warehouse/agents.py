"""Agents used by warehouse meta-agent example."""

from __future__ import annotations

from queue import PriorityQueue

import mesa
from mesa.discrete_space import FixedAgent


class InventoryAgent(FixedAgent):
    """Represents an inventory item in the warehouse."""

    def __init__(self, model, cell, item: str):
        super().__init__(model)
        self.cell = cell
        self.item = item
        self.quantity = 1000


class RouteAgent(mesa.Agent):
    """Handle path finding for the warehouse robots."""

    def __init__(self, model):
        super().__init__(model)

    def find_path(self, start, goal) -> list[tuple[int, int, int]] | None:
        """Find a path from ``start`` to ``goal`` using A* search."""

        def heuristic(a, b) -> int:
            dx = abs(a[0] - b[0])
            dy = abs(a[1] - b[1])
            return dx + dy

        open_set = PriorityQueue()
        open_set.put((0, start.coordinate))
        came_from = {}
        g_score = {start.coordinate: 0}

        while not open_set.empty():
            _, current = open_set.get()

            if current[:2] == goal.coordinate[:2]:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                path.insert(0, start.coordinate)
                path.pop()
                return path

            for n_cell in self.model.warehouse[current].neighborhood:
                coord = n_cell.coordinate

                # Only consider orthoginal neighbors in x/y plane.
                if abs(coord[0] - current[0]) + abs(coord[1] - current[1]) != 1:
                    continue

                tentative_g_score = g_score[current] + 1
                if not n_cell.is_empty:
                    tentative_g_score += 50

                if coord not in g_score or tentative_g_score < g_score[coord]:
                    g_score[coord] = tentative_g_score
                    f_score = tentative_g_score + heuristic(coord, goal.coordinate)
                    open_set.put((f_score, coord))
                    came_from[coord] = current

        return None


class SensorAgent(mesa.Agent):
    """Detect obstacles and move the robot along a computed path."""

    def __init__(self, model):
        super().__init__(model)

    def move(
        self, coord: tuple[int, int, int], path: list[tuple[int, int, int]]
    ) -> str:
        """Move one step along the current path."""
        robot = getattr(self, "meta_agent", self)

        if coord not in path:
            raise ValueError("Current coordinate not in path.")

        idx = path.index(coord)
        if idx + 1 >= len(path):
            return "movement complete"

        next_cell = self.model.warehouse[path[idx + 1]]
        if next_cell.is_empty:
            robot.cell = next_cell
            return "moving"

        neighbors = self.model.warehouse[robot.cell.coordinate].neighborhood
        empty_neighbors = [n for n in neighbors if n.is_empty]
        if empty_neighbors:
            robot.cell = self.random.choice(empty_neighbors)

        new_path = robot.get_constituting_agent_instance(RouteAgent).find_path(
            robot.cell, robot.item.cell
        )
        robot.path = new_path
        return "recalculating"


class WorkerAgent(mesa.Agent):
    """Handle inverntory pickup and delivery to the loading dock."""

    def __init__(self, model, ld, cs):
        super().__init__(model)
        self.loading_dock = ld
        self.charging_station = cs
        self.path: list[tuple[int, int, int]] | None = None
        self.carrying: str | None = None
        self.item: InventoryAgent | None = None

    def initiate_task(self, item: InventoryAgent):
        """Start a new inventory task."""
        robot = getattr(self, "meta_agent", self)
        robot.item = item
        robot.path = robot.find_path(robot.cell, item.cell)

    def continue_task(self):
        """Continue the current task if the robot has one."""
        robot = getattr(self, "meta_agent", self)
        if robot.path is None or robot.item is None:
            return

        status = robot.get_constituting_agent_instance(SensorAgent).move(
            robot.cell.coordinate, robot.path
        )

        if status == "movement complete" and robot.status == "inventory":
            source_coordinate = robot.cell.coordinate
            target_level = robot.item.cell.coordinate[2]
            robot.cell = self.model.warehouse[
                (source_coordinate[0], source_coordinate[1], target_level)
            ]
            robot.status = "loading"
            robot.carrying = robot.item.item
            robot.item.quantity -= 1

            loading_coordinate = robot.cell.coordinate
            robot.cell = self.model.warehouse[
                (loading_coordinate[0], loading_coordinate[1], 0)
            ]
            robot.path = robot.find_path(robot.cell, robot.loading_dock)

        if status == "movement complete" and robot.status == "loading":
            robot.carrying = None
            robot.status = "open"
            robot.path = None
            robot.item = None
