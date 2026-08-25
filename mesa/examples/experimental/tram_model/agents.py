"""Agents for the Tram Route Model."""

import math

from mesa import Agent, Model
from mesa.experimental.mesa_signals import HasEmitters, Observable
from mesa.experimental.states import ContinuousState, Threshold


class Tram(Agent, HasEmitters):
    """A tram travelling an ordered route of station positions.

    Attributes:
        acceleration (Observable): Current acceleration, m/s^2.
        speed (ContinuousState): Current speed, m/s.
        position (ContinuousState): Current position along the route, m.
            Chained off speed (position' = speed).
        brake_point (Observable): Position at which braking begins for the
            current segment, m. Updated at the start of each departure;
            the _brake_point threshold reacts to changes automatically.
        cruise_speed (Observable): Speed at which the tram stops accelerating,
            m/s. An Observable rather than a plain attribute so the _cruise
            threshold tracks it instead of a fixed limit.
    """

    acceleration = Observable(fallback_value=0.0)
    speed = ContinuousState(fallback_value=0.0, rate=lambda a: a.acceleration)
    position = ContinuousState(fallback_value=0.0, rate=lambda a: a.speed)
    brake_point = Observable(fallback_value=float("inf"))
    cruise_speed = Observable(fallback_value=15.0)

    # Threshold names must not collide with the private backing attribute of any
    # Observable they read: a Threshold called _brake_point would be found by the
    # lookup of brake_point's own "_brake_point" store and silently shadow it.
    _cruise_threshold = Threshold(
        state=speed, limit=cruise_speed, callback="start_coasting", direction="rising"
    )
    _braking_threshold = Threshold(
        state=position, limit=brake_point, callback="brake", direction="rising"
    )
    _stop_threshold = Threshold(
        state=speed, limit=0.0, callback="arrive_at_station", direction="falling"
    )

    def __init__(
        self,
        model: Model,
        route: list[float],
        cruise_speed: float = 15.0,
        acceleration_rate: float = 2.0,
        deceleration_rate: float = 3.0,
        dwell_time: float = 5.0,
    ) -> None:
        """Create a new tram.

        Args:
            model (Model): The model instance that contains the tram.
            route (list[float]): Ordered station positions, in meters, to visit.
                Must contain at least a starting position and one destination.
            cruise_speed (float, optional): Target speed once accelerating
                finishes, m/s. Defaults to 15.0.
            acceleration_rate (float, optional): Acceleration while departing
                a station, m/s^2. Defaults to 2.0.
            deceleration_rate (float, optional): Deceleration while braking
                into a station, m/s^2. Defaults to 3.0.
            dwell_time (float, optional): Time spent stopped at each station
                before departing for the next one, seconds. Defaults to 5.0.

        Raises:
            ValueError: If route has fewer than two stops.
        """
        super().__init__(model)

        if len(route) < 2:
            raise ValueError("route must contain at least a start and one destination")

        self.route = route
        self.acceleration_rate = acceleration_rate
        self.deceleration_rate = deceleration_rate
        self.dwell_time = dwell_time
        self._segment_index = 1

        # The limits the thresholds read must exist before the first
        # ContinuousState assignment, which is what binds the thresholds.
        self.brake_point = float("inf")
        self.cruise_speed = cruise_speed

        self.acceleration = 0.0
        self.speed = 0.0
        self.position = route[0]

    @property
    def next_station(self) -> float:
        """Position of the station this tram is currently travelling towards.

        Returns:
            float: The target station's position, in meters.
        """
        return self.route[self._segment_index]

    @property
    def route_complete(self) -> bool:
        """Whether the tram has visited every station on its route.

        Returns:
            bool: True once the final station has been reached.
        """
        return self._segment_index >= len(self.route)

    def peak_speed(self) -> float:
        """Highest speed the tram can reach before it must brake for the next stop.

        On a long segment this is the cruise speed. On a short one the tram runs
        out of room first and the profile is triangular: it accelerates to
        v = sqrt(2d / (1/a + 1/b)) and brakes from there, never coasting.

        Returns:
            float: The peak speed for this segment, in meters per second.
        """
        distance = self.next_station - self.position
        triangular = math.sqrt(
            2.0
            * distance
            / (1.0 / self.acceleration_rate + 1.0 / self.deceleration_rate)
        )
        return min(self.cruise_speed, triangular)

    def braking_distance(self) -> float:
        """Distance needed to decelerate to a stop from this segment's peak speed.

        Returns:
            float: The braking distance, in meters.
        """
        return self.peak_speed() ** 2 / (2.0 * self.deceleration_rate)

    def depart(self) -> None:
        """Accelerate towards the next station."""
        self.brake_point = self.next_station - self.braking_distance()
        self.acceleration = self.acceleration_rate

    def start_coasting(self) -> None:
        """Stop accelerating once cruise speed is reached."""
        self.acceleration = 0.0

    def brake(self) -> None:
        """Decelerate once the brake point for this segment is reached."""
        self.brake_point = float("inf")
        self.acceleration = -self.deceleration_rate

    def arrive_at_station(self) -> None:
        """Come to a stop and schedule departure for the next station, if any."""
        self.acceleration = 0.0
        self._segment_index += 1

        if self.route_complete:
            return

        self.model.schedule_event(self.depart, after=self.dwell_time)
