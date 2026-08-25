"""Tests for mesa.experimental.states.state.

Validates piecewise-linear extrapolation, piecewise-quadratic kinematics,
analytical threshold crossing detection, and initialization race condition safety.
"""

from __future__ import annotations

import pytest

import mesa.experimental.mesa_signals.core as signals_core
from mesa import Agent, Model
from mesa.experimental.mesa_signals import HasEmitters, Observable
from mesa.experimental.mesa_signals.core import _hashable_signal
from mesa.experimental.states import (
    ContinuousState,
    Threshold,
)


class Tram(Agent, HasEmitters):
    """Mirrors the Tram from demo.py for testing.

    Validates chained continuous states (position' = speed, speed' = acceleration)
    and dynamic threshold limit overrides natively using Observables.
    """

    acceleration = Observable(fallback_value=0.0)
    speed = ContinuousState(fallback_value=0.0, rate=lambda a: a.acceleration)
    position = ContinuousState(fallback_value=0.0, rate=lambda a: a.speed)
    brake_point = Observable(fallback_value=float("inf"))

    _cruise = Threshold(
        state=speed, limit=15.0, callback="start_coasting", direction="rising"
    )
    _brake_point = Threshold(
        state=position, limit=brake_point, callback="brake", direction="rising"
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
        self.brake_point = float("inf")
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
        tram.brake_point = 25.0
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
        tram.brake_point = 25.0

        # Distance to stop: v^2 / 2a = 100 / 4 = 25.
        # Tram will exactly touch 25.0 as speed hits 0.0, then reverse.
        tram.acceleration = -2.0
        model.run_until(10.0)

        # Because direction='rising' requires strictly positive velocity
        # crossing the threshold, the v_cross == 0 tangent touch must not fire.
        assert tram.brake_events == []

    def test_dynamic_limit_rearming(self, model: TramModel) -> None:
        """Ensure setting a declarative limit properly recalculates and arms the threshold."""
        tram = Tram(model)
        tram.brake_point = 25.0
        tram.depart()
        model.run_until(10.0)
        assert len(tram.brake_events) == 1

        # Change limit to 100.0 and re-accelerate
        tram.brake_point = 100.0
        tram.speed = 0.0
        tram.position = 25.0
        tram.depart()
        model.run_until(20.0)

        assert len(tram.brake_events) == 2

        # The tram accelerates for 7.5s (hitting 15 m/s cruise speed at position 81.25),
        # then coasts the remaining 18.75m at 15 m/s, taking exactly 1.25s.
        # Total crossing time = 10.0 (start) + 7.5 (accelerating) + 1.25 (coasting) = 18.75
        assert tram.brake_events[1] == pytest.approx(18.75)


class TestDemoIntegration:
    """Validates the full piecewise hybrid kinematics system end-to-end."""

    def test_demo_scenario_end_to_end(self, model: TramModel) -> None:
        """Recreates the exact demo.py kinematics route and checks timing."""
        tram = Tram(model)

        # Route logic
        tram.brake_point = 162.50  # 200 - 37.5 braking distance
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


class TestNamespaceCollision:
    """Validates the shadow rate attribute guard in ContinuousState.__set_name__."""

    def test_colliding_class_attribute_raises(self) -> None:
        """Ensure a pre-existing `<name>_rate` attribute is rejected."""
        with pytest.raises(AttributeError, match="Namespace collision"):

            class Colliding(Agent, HasEmitters):
                speed_rate = 1.0
                speed = ContinuousState(fallback_value=0.0, rate=1.0)

    def test_colliding_slot_raises(self) -> None:
        """Ensure the guard inspects __slots__ as well as the class dict."""
        with pytest.raises(AttributeError, match="Namespace collision"):

            class SlottedColliding(Agent, HasEmitters):
                __slots__ = ("speed_rate",)

                speed = ContinuousState(fallback_value=0.0, rate=1.0)


class TestRateResolutionFailures:
    """Validates which AttributeErrors are absorbed as init races and which propagate."""

    def test_unassigned_observable_is_absorbed_as_at_rest(
        self, model: TramModel
    ) -> None:
        """Ensure a rate reading a not-yet-assigned Observable resolves to 0.0."""
        tram = Tram(model, initial_acceleration=2.0)
        descriptor = type(tram).speed

        # Reproduce the window during __init__ where the backing store is absent.
        del tram._acceleration
        descriptor._get_state(tram)["rate_computed"].is_dirty = True

        assert descriptor.get_rate(tram) == 0.0

    def test_missing_private_non_observable_is_not_a_race(
        self, model: TramModel
    ) -> None:
        """Ensure a private name that is not an Observable is not absorbed."""
        tram = Tram(model)
        descriptor = type(tram).speed

        assert not descriptor._is_uninitialized_observable_race(
            tram, AttributeError("no attribute '_ghost'", name="_ghost")
        )


class TestCyclicalDependency:
    """Validates cycle detection when a computation writes the state it reads."""

    def test_write_from_dependent_computation_raises(
        self, model: TramModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ensure assigning a state that the active computation depends on raises."""
        tram = Tram(model)
        computed = type(tram).position._get_state(tram)["rate_computed"]

        # Stand in for being inside an evaluation that has already read position.
        monkeypatch.setattr(signals_core, "CURRENT_COMPUTED", computed)
        signals_core.PROCESSING_SIGNALS.add(_hashable_signal(tram, "position"))
        try:
            with pytest.raises(ValueError, match="Cyclical dependency"):
                tram.position = 1.0
        finally:
            signals_core.PROCESSING_SIGNALS.clear()


class TestThresholdBinding:
    """Validates the threshold binding lifecycle and class-level registration."""

    def test_bind_is_idempotent(self, model: TramModel) -> None:
        """Ensure re-binding an already-bound threshold leaves the projection intact."""
        tram = Tram(model)
        tram.brake_point = 50.0
        tram.depart()
        projected = tram._brake_point

        type(tram)._brake_point.bind(tram)

        assert tram._brake_point == projected

    def test_registration_is_idempotent(self) -> None:
        """Ensure re-running __set_name__ does not duplicate the registry entry."""
        Tram.__dict__["_cruise"].__set_name__(Tram, "_cruise")

        assert Tram._continuous_thresholds.count("_cruise") == 1


class TestApexRejection:
    """Validates that a tangent apex is rejected under the 'crossing' direction."""

    def test_crossing_direction_rejects_tangent(
        self, model: TramModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ensure a trajectory that only touches the limit never fires."""
        monkeypatch.setattr(Tram.__dict__["_brake_point"], "direction", "crossing")
        tram = Tram(model)

        # Arms a linear crossing at t=2.5, then re-solves to a tangent touch:
        # position peaks at exactly 25.0 at t=5.0 with zero velocity.
        tram.brake_point = 25.0
        tram.speed = 10.0
        tram.acceleration = -2.0

        model.run_until(5.0)
        assert tram.position == pytest.approx(25.0)

        model.run_until(10.0)
        assert tram.brake_events == []
