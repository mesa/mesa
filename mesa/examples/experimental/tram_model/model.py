"""Tram Route Model.

A model of a tram running a multi-station route using continuous-time
kinematics. The tram accelerates to a cruise speed, coasts, brakes at an
analytically-computed point before each station, and dwells before departing
for the next one -- all without the model needing to step on every tick to
check whether a threshold has been crossed.
"""

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.examples.experimental.tram_model.agents import Tram
from mesa.experimental.data_collection import DataRecorder
from mesa.experimental.scenarios import Scenario


class TramScenario(Scenario):
    """Scenario for the Tram Route model.

    Args:
        n_stations: Number of stations on the route, including the origin.
        station_spacing: Distance between consecutive stations, in meters.
        cruise_speed: Target speed once accelerating finishes, m/s.
        acceleration_rate: Acceleration while departing a station, m/s^2.
        deceleration_rate: Deceleration while braking into a station, m/s^2.
        dwell_time: Time spent stopped at each station, in seconds.
        rng: Seed for reproducibility.
    """

    n_stations: int = 20
    station_spacing: float = 200.0
    cruise_speed: float = 15.0
    acceleration_rate: float = 2.0
    deceleration_rate: float = 3.0
    dwell_time: float = 5.0


class TransitSystem(Model):
    """A minimal model containing a single tram running a multi-station route.

    Attributes:
        route (list[float]): Ordered station positions, in meters.
        tram (Tram): The tram running the route.
    """

    def __init__(self, scenario: TramScenario = TramScenario):
        """Create the model and its tram.

        Args:
            scenario: TramScenario containing model parameters.
        """
        super().__init__(scenario=scenario)

        self.route = [
            scenario.station_spacing * i for i in range(int(scenario.n_stations))
        ]

        self.tram = Tram(
            self,
            route=self.route,
            cruise_speed=scenario.cruise_speed,
            acceleration_rate=scenario.acceleration_rate,
            deceleration_rate=scenario.deceleration_rate,
            dwell_time=scenario.dwell_time,
        )

        self.recorder = DataRecorder(self)
        self.data_registry.track_agents(
            self.agents, "tram_data", ["position", "speed", "acceleration"]
        ).record(self.recorder)

        # The DataCollector is kept alongside only to feed the plots in
        # app.py, since make_plot_component reads model.datacollector.
        self.datacollector = DataCollector(
            model_reporters={
                "Position": lambda m: m.tram.position,
                "Speed": lambda m: m.tram.speed,
                "Acceleration": lambda m: m.tram.acceleration,
            }
        )
        self.datacollector.collect(self)

        # The tram is idle until it is told to leave the first station. Scheduling
        # the departure rather than calling it directly keeps every state change
        # on the event queue, so model.time is meaningful from the very first entry.
        self.schedule_event(self.tram.depart, at=0.0)

    def step(self) -> None:
        """Sample the tram's continuous states once per time unit.

        The tram itself needs no per-step logic: its speed and position are
        extrapolated analytically and its transitions fire from the event
        queue. This step exists purely to feed the DataCollector behind the
        plots; the recorder collects without it.
        """
        self.datacollector.collect(self)


if __name__ == "__main__":
    model = TransitSystem(scenario=TramScenario())

    print(f"Route: {model.route}")
    print(
        f"cruise_speed={model.tram.cruise_speed} m/s, "
        f"acceleration={model.tram.acceleration_rate} m/s^2, "
        f"deceleration={model.tram.deceleration_rate} m/s^2, "
        f"braking_distance={model.tram.braking_distance():.2f}m\n"
    )

    model.run_until(2000.0)

    print(model.recorder.get_table_dataframe("tram_data").to_string())

    print(f"\nFinal time:     {model.time:.2f}")
    print(f"Final position: {model.tram.position:.2f} m")
    print(f"Final speed:    {model.tram.speed:.2f} m/s")
