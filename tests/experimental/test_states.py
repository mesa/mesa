"""Tests for mesa.experimental.states.state.

Validates piecewise-linear extrapolation, piecewise-quadratic kinematics,
analytical threshold crossing detection, and initialization race condition safety.
"""

from __future__ import annotations

import pytest

from mesa import Agent, Model
from mesa.experimental.mesa_signals import HasEmitters, Observable
from mesa.experimental.states.state import (
    ContinuousScheduler,
    ContinuousState,
    Threshold,
)


class Tram(Agent, HasEmitters):
    """Mirrors the Tram from demo.py for testing.

    Validates chained continuous states (position' = speed, speed' = acceleration)
    and dynamic threshold limit overrides.
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

    def __init__(self, model: Model, initial_acceleration: float = 0.0) -> None:
        """Initialize the Tram agent.

        Args:
            model: The Mesa model instance.
            initial_acceleration: The starting acceleration rate.
        """
        super().__init__(model)
        self.acceleration = initial_acceleration
        self.speed = 0.0
        self.position = 0.0
        self.coasting_events: list[float] = []
        self.brake_events: list[float] = []
        self.stop_events: list[float] = []

    def depart(self) -> None:
        """Set acceleration to simulate departing a station."""
        self.acceleration = 2.0

    def brake(self) -> None:
        """Set deceleration to simulate applying brakes."""
        self.brake_events.append(self.model.time)
        self.acceleration = -3.0

    def start_coasting(self) -> None:
        """Set acceleration to zero to coast at a constant speed."""
        self.coasting_events.append(self.model.time)
        self.acceleration = 0.0

    def arrive_at_station(self) -> None:
        """Halt the tram upon reaching zero speed."""
        self.stop_events.append(self.model.time)
        self.acceleration = 0.0


class Wolf(Agent, HasEmitters):
    """A simple agent with a constant rate continuous state."""

    energy = ContinuousState(fallback_value=100.0, rate=-1.0)
    starvation = Threshold(state=energy, limit=0.0, callback="die", direction="falling")

    def __init__(self, model: Model, initial_energy: float = 100.0) -> None:
        """Initialize the Wolf agent.

        Args:
            model: The Mesa model instance.
            initial_energy: The starting energy level.
        """
        super().__init__(model)
        self.energy = initial_energy
        self.died_at: float | None = None

    def die(self) -> None:
        """Record death time and remove agent from the model."""
        self.died_at = self.model.time
        self.remove()


class BadRateAgent(Agent, HasEmitters):
    """An agent designed to trigger a genuine AttributeError in a rate lambda."""

    acceleration = Observable(fallback_value=0.0)
    speed = ContinuousState(fallback_value=0.0, rate=lambda a: a.a_cceleration)
    threshold = Threshold(state=speed, limit=10.0, callback="noop")

    def __init__(self, model: Model) -> None:
        """Initialize the BadRateAgent."""
        super().__init__(model)
        self.acceleration = 2.0
        self.speed = 0.0

    def noop(self) -> None:
        """Empty callback."""


class BadRateAgentNoThreshold(Agent, HasEmitters):
    """An agent with a rate bug, but no threshold to force eager evaluation."""

    acceleration = Observable(fallback_value=0.0)
    speed = ContinuousState(fallback_value=0.0, rate=lambda a: a.a_cceleration)

    def __init__(self, model: Model) -> None:
        """Initialize the BadRateAgentNoThreshold."""
        super().__init__(model)
        self.acceleration = 2.0
        self.speed = 0.0


class TramModel(Model):
    """Minimal model for continuous state testing."""

    def __init__(self) -> None:
        """Initialize the model and master clock."""
        super().__init__()
        self.continuous_scheduler = ContinuousScheduler(self)


@pytest.fixture
def model() -> TramModel:
    """Provide a fresh TramModel instance for tests."""
    return TramModel()


class TestInitOrderingRace:
    """Validates the handling of observable initialization race conditions."""

    def test_agent_construction_does_not_raise(self, model: TramModel) -> None:
        """Ensure eager rate evaluation during __init__ absorbs missing attributes."""
        tram = Tram(model)
        assert tram.speed == 0.0

    def test_rate_recovers_after_init_completes(self, model: TramModel) -> None:
        """Ensure rate evaluates correctly once initialization is complete."""
        tram = Tram(model, initial_acceleration=1.0)
        assert tram.__class__.speed.get_rate(tram) == 1.0

    def test_threshold_projects_correctly_with_nonzero_initial_rate(
        self, model: TramModel
    ) -> None:
        """Ensure crossing time is correct despite the initialization race absorption."""
        tram = Tram(model, initial_acceleration=1.0)
        assert tram._cruise == pytest.approx(15.0)

    def test_multiple_agents_each_survive_construction(self, model: TramModel) -> None:
        """Ensure racing condition absorption works cleanly across multiple agent instantiations."""
        trams = [Tram(model, initial_acceleration=float(i)) for i in range(4)]
        for i, tram in enumerate(trams):
            assert tram.speed == 0.0
            rate = tram.__class__.speed.get_rate(tram)
            assert rate == float(i)


class TestGenuineRateBugsStillRaise:
    """Validates that real user errors in rate functions are not swallowed."""

    def test_typo_raises_during_eager_bind(self, model: TramModel) -> None:
        """Ensure real attribute errors crash eager evaluations correctly."""
        with pytest.raises(AttributeError):
            BadRateAgent(model)

    def test_typo_error_message_mentions_missing_attribute(
        self, model: TramModel
    ) -> None:
        """Ensure error messages surface the correct misspelled attribute."""
        with pytest.raises(AttributeError) as exc_info:
            BadRateAgent(model)
        assert "a_cceleration" in str(exc_info.value)

    def test_typo_raises_on_lazy_first_access(self, model: TramModel) -> None:
        """Ensure lazy evaluation still raises errors accurately."""
        agent = BadRateAgentNoThreshold(model)
        with pytest.raises(AttributeError):
            _ = agent.speed


class TestContinuousStateExtrapolation:
    """Validates piecewise-linear and piecewise-quadratic trajectory logic."""

    def test_constant_rate_extrapolates_linearly(self, model: TramModel) -> None:
        """Ensure unchained states use 1st order extrapolation."""
        wolf = Wolf(model)
        assert wolf.energy == pytest.approx(100.0)
        model.run_until(10.0)
        assert wolf.energy == pytest.approx(90.0)
        model.run_until(30.0)
        assert wolf.energy == pytest.approx(70.0)

    def test_chained_rate_extrapolates_quadratically(self, model: TramModel) -> None:
        """Ensure chained states use 2nd order extrapolation (position = 0.5 * a * t^2)."""
        tram = Tram(model)
        tram.depart()  # acceleration = 2.0
        model.run_until(5.0)
        assert tram.speed == pytest.approx(10.0)  # v = 2 * 5
        assert tram.position == pytest.approx(25.0)  # p = 0.5 * 2 * 25

    def test_trajectory_continuous_across_rate_change(self, model: TramModel) -> None:
        """Ensure snapping the baseline prevents discontinuity in mathematical values."""
        tram = Tram(model)
        tram.depart()
        model.run_until(3.0)
        pos_before = tram.position
        tram.acceleration = 0.0  # rate changes here
        pos_immediately_after = tram.position
        assert pos_immediately_after == pytest.approx(pos_before)


class TestThresholdCrossing:
    """Validates analytical intersection solving for thresholds."""

    def test_constant_rate_threshold_fires_at_correct_time(
        self, model: TramModel
    ) -> None:
        """Ensure pure linear intersections are calculated accurately."""
        wolf = Wolf(model)
        model.run_until(150.0)
        assert wolf.died_at == pytest.approx(100.0)
        assert wolf not in model.agents

    def test_quadratic_threshold_fires_at_correct_time(self, model: TramModel) -> None:
        """Ensure parabolic intersections (acceleration) are solved exactly."""
        tram = Tram(model)
        type(tram)._brake_point.set_limit(tram, 25.0)
        tram.depart()  # acc = 2.0
        model.run_until(10.0)

        # 0.5 * 2 * t^2 = 25 -> t = 5.0
        assert tram.brake_events == [pytest.approx(5.0)]

    def test_rising_threshold_ignored_while_rate_is_negative(
        self, model: TramModel
    ) -> None:
        """Ensure rising thresholds are blind to decreasing states."""
        tram = Tram(model)
        tram.brake()
        model.run_until(10.0)
        assert tram.coasting_events == []

    def test_tangent_touch_rejection(self, model: TramModel) -> None:
        """Ensure tangent trajectories that don't pass the boundary are rejected."""
        tram = Tram(model)
        tram.speed = 10.0
        tram.position = 0.0
        type(tram)._brake_point.set_limit(tram, 25.0)

        # Distance to stop: v^2 / 2a = 100 / 4 = 25.
        # Tram will exactly touch 25.0 as speed hits 0.0, then reverse.
        tram.acceleration = -2.0
        model.run_until(10.0)

        # Because direction='rising' requires strictly positive velocity
        # crossing the threshold, the v_cross == 0 tangent touch must not fire.
        assert tram.brake_events == []

    def test_dynamic_limit_rearming(self, model: TramModel) -> None:
        """Ensure set_limit properly recalculates and rearms a previously fired threshold."""
        tram = Tram(model)
        type(tram)._brake_point.set_limit(tram, 25.0)
        tram.depart()
        model.run_until(10.0)
        assert len(tram.brake_events) == 1

        # Change limit to 100.0 and re-accelerate
        type(tram)._brake_point.set_limit(tram, 100.0)
        tram.speed = 0.0
        tram.position = 25.0
        tram.depart()
        model.run_until(20.0)

        assert len(tram.brake_events) == 2

        # The tram accelerates for 7.5s (hitting 15 m/s cruise speed at position 81.25),
        # then coasts the remaining 18.75m at 15 m/s, taking exactly 1.25s.
        # Total crossing time = 10.0 (start) + 7.5 (accelerating) + 1.25 (coasting) = 18.75
        assert tram.brake_events[1] == pytest.approx(18.75)


class TestContinuousScheduler:
    """Validates centralized master clock bookkeeping and sorting."""

    def test_threshold_is_tracked_after_bind(self, model: TramModel) -> None:
        """Ensure bound thresholds are registered with the active set."""
        tram = Tram(model)
        tram.depart()
        active_instances = {
            inst for inst, _ in model.continuous_scheduler._active_thresholds
        }
        assert tram in active_instances

    def test_threshold_untracked_after_firing(self, model: TramModel) -> None:
        """Ensure thresholds are removed from tracking once executed."""
        tram = Tram(model)
        tram.depart()
        model.run_until(10.0)
        active = model.continuous_scheduler._active_thresholds
        assert (tram, Tram._cruise) not in active

    def test_threshold_untracked_when_rate_becomes_zero(self, model: TramModel) -> None:
        """Ensure zero-rate trajectories safely remove thresholds from the queue."""
        tram = Tram(model)
        tram.depart()
        tram.acceleration = 0.0
        active = model.continuous_scheduler._active_thresholds
        assert (tram, Tram._cruise) not in active

    def test_multiple_agents_batch_independently(self, model: TramModel) -> None:
        """Ensure multiple agents yield independently correct threshold crossings."""
        slow = Tram(model, initial_acceleration=1.0)
        fast = Tram(model, initial_acceleration=3.0)
        model.run_until(20.0)
        assert fast.coasting_events == [pytest.approx(5.0)]
        assert slow.coasting_events == [pytest.approx(15.0)]

    def test_master_clock_reschedules_to_earliest_active_crossing(
        self, model: TramModel
    ) -> None:
        """Ensure the heapq always prioritizes the soonest crossing event."""
        Tram(model, initial_acceleration=1.0)
        Tram(model, initial_acceleration=3.0)
        scheduler = model.continuous_scheduler
        assert scheduler._master_event is not None
        assert scheduler._master_event.time == pytest.approx(5.0)


class TestDemoIntegration:
    """Validates the full piecewise hybrid kinematics system end-to-end."""

    def test_demo_scenario_end_to_end(self, model: TramModel) -> None:
        """Recreates the exact demo.py kinematics route and checks timing."""
        tram = Tram(model)

        # Route logic
        type(tram)._brake_point.set_limit(tram, 162.50)  # 200 - 37.5 braking distance
        tram.depart()

        model.run_until(20.0)

        # 1. Cruise limit reached at 7.5s (15.0 / 2.0)
        assert tram.coasting_events == [pytest.approx(7.5)]

        # 2. Brake limit (162.50) reached.
        # Accel phase pos = 56.25. Coasting distance = 162.50 - 56.25 = 106.25.
        # Coasting time = 106.25 / 15.0 = 7.0833s.
        # Total time = 7.5 + 7.0833 = 14.5833s.
        assert tram.brake_events == [pytest.approx(14.583333333333334)]

        # 3. Stop limit (0.0) reached.
        # Braking time = 15.0 / 3.0 = 5.0s.
        # Total time = 14.5833 + 5.0 = 19.5833s.
        assert tram.stop_events == [pytest.approx(19.583333333333334)]

        # Ensure final mathematical state is accurate
        assert tram.speed == pytest.approx(0.0)
        assert tram.position == pytest.approx(200.0)
