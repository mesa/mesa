"""Tram Route Model

A model of a tram running a multi-station route using continuous-time
kinematics layer. The tram accelerates to a cruise speed, coasts, brakes at an
analytically-computed point before each station, and dwells before departing
for the next one -- all without the model needing to step on every tick to
check whether a threshold has been crossed.
"""

from mesa import Agent, Model
from mesa.experimental.mesa_signals import HasEmitters, Observable
from mesa.experimental.states import ContinuousScheduler, ContinuousState, Threshold


class Tram(Agent, HasEmitters):
    """A tram travelling an ordered route of station positions.

    Args:
        acceleration (Observable): Current acceleration, m/s^2. The control
            input driving speed; positive while departing, negative while
            braking, zero while coasting or stopped.
        speed (ContinuousState): Current speed, m/s.
        position (ContinuousState): Current position along the route, m.
            Chained off speed (position' = speed).
    """

    acceleration = Observable(fallback_value=0.0)
    speed = ContinuousState(fallback_value=0.0, rate=lambda a: a.acceleration)
    position = ContinuousState(fallback_value=0.0, rate=lambda a: a.speed)

    _cruise = Threshold(
        state=speed, limit=15.0, callback="start_coasting", direction="rising"
    )
    _brake_point = Threshold(
        state=position, limit=float("inf"), callback="brake", direction="rising"
    )
    _stop = Threshold(
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
        self.cruise_speed = cruise_speed
        self.acceleration_rate = acceleration_rate
        self.deceleration_rate = deceleration_rate
        self.dwell_time = dwell_time
        self._segment_index = 1

        self.acceleration = 0.0
        self.speed = 0.0
        self.position = route[0]

        Tram._cruise.set_limit(self, self.cruise_speed)

    @property
    def next_station(self) -> float:
        """Position of the station this tram is currently travelling towards.

        Returns:
            float: The target station's position, in meters.
        """
        return self.route[self._segment_index]

    def braking_distance(self) -> float:
        """Distance needed to decelerate from cruise_speed to a stop.

        Returns:
            float: The braking distance, in meters.
        """
        return self.cruise_speed**2 / (2.0 * self.deceleration_rate)

    def depart(self) -> None:
        """Accelerate towards the next station.

        Arms the brake-point threshold for this segment and re-arms the
        cruise and stop thresholds, since each already fired once for the
        previous segment.
        """
        target = self.next_station
        print(
            f"[t={self.model.time:.2f}] Tram {self.unique_id} departing "
            f"(pos={self.position:.2f}) towards station at {target:.2f}m. "
            f"Accelerating at {self.acceleration_rate:.1f} m/s^2."
        )

        brake_at = target - self.braking_distance()
        Tram._brake_point.set_limit(self, brake_at)
        Tram._cruise.rearm(self)
        Tram._stop.rearm(self)

        self.acceleration = self.acceleration_rate

    def start_coasting(self) -> None:
        """Stop accelerating once cruise speed is reached."""
        print(
            f"[t={self.model.time:.2f}] Tram {self.unique_id} reached cruise "
            f"speed {self.cruise_speed:.1f} m/s. Coasting."
        )
        self.acceleration = 0.0

    def brake(self) -> None:
        """Decelerate once the brake point for this segment is reached."""
        print(
            f"[t={self.model.time:.2f}] Tram {self.unique_id} at brake point "
            f"(pos={self.position:.2f}). Applying brakes at "
            f"-{self.deceleration_rate:.1f} m/s^2."
        )
        self.acceleration = -self.deceleration_rate

    def arrive_at_station(self) -> None:
        """Come to a stop and schedule departure for the next station, if any."""
        print(
            f"[t={self.model.time:.2f}] Tram {self.unique_id} arrived at "
            f"station (pos={self.position:.2f})."
        )
        self.acceleration = 0.0
        self._segment_index += 1

        if self._segment_index >= len(self.route):
            print(
                f"[t={self.model.time:.2f}] Tram {self.unique_id} has completed its route."
            )
            return

        self.model.schedule_event(self.depart, after=self.dwell_time)


class TransitSystem(Model):
    """A minimal model containing a single tram running a multi-station route.

    Args:
        continuous_scheduler (ContinuousScheduler): Batches threshold-crossing
            events for all continuous-state agents in the model.
        tram (Tram): The tram running the route.
        event_count_log (list[int]): Size of the native event queue, recorded
            on each call to log_queue_health(); a diagnostic showing that the
            queue stays small regardless of how many crossings occur.
    """

    def __init__(self, route: list[float]) -> None:
        """Create the model and its tram.

        Args:
            route (list[float]): Ordered station positions, in meters, for the tram to visit.
        """
        super().__init__()
        self.continuous_scheduler = ContinuousScheduler(self)
        self.tram = Tram(self, route=route)


if __name__ == "__main__":
    route = [200 * i for i in range(20)]
    model = TransitSystem(route=route)

    print(f"Route: {route}")
    print(
        f"cruise_speed={model.tram.cruise_speed} m/s, "
        f"acceleration={model.tram.acceleration_rate} m/s^2, "
        f"deceleration={model.tram.deceleration_rate} m/s^2, "
        f"braking_distance={model.tram.braking_distance():.2f}m\n"
    )

    model.tram.depart()

    model.run_until(2000.0)

    print(f"\nFinal time:     {model.time:.2f}")
    print(f"Final position: {model.tram.position:.2f} m")
    print(f"Final speed:    {model.tram.speed:.2f} m/s")
