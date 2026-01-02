"""Tests for continuous observables in mesa_signals."""

from unittest.mock import Mock

import numpy as np

from mesa import Agent, Model
from mesa.experimental.devs import ABMSimulator, DEVSimulator
from mesa.experimental.mesa_signals import (
    Computable,
    Computed,
    ContinuousObservable,
    HasObservables,
)


class SimpleModel(Model):
    """Simple model with time tracking for testing."""

    def __init__(self, seed=None):
        """Initialize the model."""
        super().__init__(seed=seed)
        self.simulator = DEVSimulator()
        self.simulator.setup(self)


def test_continuous_observable_basic():
    """Test basic ContinuousObservable functionality."""

    class MyAgent(Agent, HasObservables):
        energy = ContinuousObservable(
            initial_value=100.0,
            rate_func=lambda value, elapsed, agent: -1.0,  # Constant depletion
        )

        def __init__(self, model):
            super().__init__(model)
            self.energy = 100.0

    model = SimpleModel()
    agent = MyAgent(model)

    # Initial value
    assert agent.energy == 100.0

    # Schedule an event to check energy later
    def check_energy():
        assert agent.energy == 90.0  # 100 - (1.0 * 10)

    model.simulator.schedule_event_absolute(check_energy, 10.0)
    model.simulator.run_until(10.0)


def test_continuous_observable_variable_rate():
    """Test ContinuousObservable with variable rate function."""

    class MyAgent(Agent, HasObservables):
        energy = ContinuousObservable(
            initial_value=100.0,
            rate_func=lambda value, elapsed, agent: -agent.metabolic_rate,
        )

        def __init__(self, model):
            super().__init__(model)
            self.metabolic_rate = 1.0
            self.energy = 100.0

    model = SimpleModel()
    agent = MyAgent(model)

    def check_first():
        assert agent.energy == 90.0
        # Change metabolic rate
        agent.metabolic_rate = 2.0

    def check_second():
        assert agent.energy == 80.0  # 90 - (2.0 * 5)

    model.simulator.schedule_event_absolute(check_first, 10.0)
    model.simulator.schedule_event_absolute(check_second, 15.0)
    model.simulator.run_until(15.0)


def test_continuous_observable_manual_set():
    """Test manually setting a ContinuousObservable value."""

    class MyAgent(Agent, HasObservables):
        energy = ContinuousObservable(
            initial_value=100.0, rate_func=lambda value, elapsed, agent: -1.0
        )

        def __init__(self, model):
            super().__init__(model)
            self.energy = 100.0

    model = SimpleModel()
    agent = MyAgent(model)

    def check_and_eat():
        assert agent.energy == 90.0
        # Manually increase energy (e.g., eating)
        agent.energy = 120.0
        assert agent.energy == 120.0

    def check_after_eating():
        assert agent.energy == 110.0  # 120 - (1.0 * 10)

    model.simulator.schedule_event_absolute(check_and_eat, 10.0)
    model.simulator.schedule_event_absolute(check_after_eating, 20.0)
    model.simulator.run_until(20.0)


def test_continuous_observable_change_signal():
    """Test that change signals are emitted correctly."""

    class MyAgent(Agent, HasObservables):
        energy = ContinuousObservable(
            initial_value=100.0, rate_func=lambda value, elapsed, agent: -1.0
        )

        def __init__(self, model):
            super().__init__(model)
            self.energy = 100.0

    model = SimpleModel()
    agent = MyAgent(model)

    handler = Mock()
    agent.observe("energy", "change", handler)

    def check_signal():
        _ = agent.energy  # Access triggers recalculation

        handler.assert_called_once()
        call_args = handler.call_args[0][0]
        assert call_args.name == "energy"
        assert call_args.old == 100.0
        assert call_args.new == 90.0
        assert call_args.type == "change"

    model.simulator.schedule_event_absolute(check_signal, 10.0)
    model.simulator.run_until(10.0)


def test_continuous_observable_no_change_no_signal():
    """Test that no signal is emitted when value doesn't change."""

    class MyAgent(Agent, HasObservables):
        energy = ContinuousObservable(
            initial_value=100.0,
            rate_func=lambda value, elapsed, agent: 0.0,  # No change
        )

        def __init__(self, model):
            super().__init__(model)
            self.energy = 100.0

    model = SimpleModel()
    agent = MyAgent(model)

    handler = Mock()
    agent.observe("energy", "change", handler)

    def check_no_signal():
        _ = agent.energy
        # Should not call handler since value didn't change
        handler.assert_not_called()

    model.simulator.schedule_event_absolute(check_no_signal, 10.0)
    model.simulator.run_until(10.0)


def test_continuous_observable_threshold_crossing_down():
    """Test threshold crossing detection (downward)."""

    class MyAgent(Agent, HasObservables):
        energy = ContinuousObservable(
            initial_value=100.0, rate_func=lambda value, elapsed, agent: -1.0
        )

        def __init__(self, model):
            super().__init__(model)
            self.energy = 100.0
            self.low_energy_triggered = False

        def on_low_energy(self, signal):
            if signal.direction == "down":
                self.low_energy_triggered = True

    model = SimpleModel()
    agent = MyAgent(model)

    # Register threshold at 50
    agent.add_threshold("energy", 50.0, agent.on_low_energy)

    def check_threshold():
        _ = agent.energy  # Should be 40.0, crossed 50.0
        assert agent.low_energy_triggered

    model.simulator.schedule_event_absolute(check_threshold, 60.0)
    model.simulator.run_until(60.0)


def test_continuous_observable_threshold_crossing_up():
    """Test threshold crossing detection (upward)."""

    class MyAgent(Agent, HasObservables):
        energy = ContinuousObservable(
            initial_value=40.0, rate_func=lambda value, elapsed, agent: 1.0
        )

        def __init__(self, model):
            super().__init__(model)
            self.energy = 40.0
            self.recharged = False

        def on_recharged(self, signal):
            if signal.direction == "up":
                self.recharged = True

    model = SimpleModel()
    agent = MyAgent(model)

    # Register threshold at 50
    agent.add_threshold("energy", 50.0, agent.on_recharged)

    def check_threshold():
        _ = agent.energy  # Should be 60.0, crossed 50.0
        assert agent.recharged

    model.simulator.schedule_event_absolute(check_threshold, 20.0)
    model.simulator.run_until(20.0)


def test_continuous_observable_multiple_thresholds():
    """Test multiple threshold crossings."""

    class MyAgent(Agent, HasObservables):
        energy = ContinuousObservable(
            initial_value=100.0, rate_func=lambda value, elapsed, agent: -1.0
        )

        def __init__(self, model):
            super().__init__(model)
            self.energy = 100.0
            self.crossings = []

        def on_threshold(self, signal):
            self.crossings.append((signal.threshold, signal.direction))

    model = SimpleModel()
    agent = MyAgent(model)

    # Register multiple thresholds
    for threshold in [75.0, 50.0, 25.0]:
        agent.add_threshold("energy", threshold, agent.on_threshold)

    def check_thresholds():
        _ = agent.energy  # Should be 20.0, crossed all three

        assert len(agent.crossings) == 3
        assert (75.0, "down") in agent.crossings
        assert (50.0, "down") in agent.crossings
        assert (25.0, "down") in agent.crossings

    model.simulator.schedule_event_absolute(check_thresholds, 80.0)
    model.simulator.run_until(80.0)


def test_continuous_observable_threshold_on_manual_set():
    """Test that thresholds are checked when manually setting values."""

    class MyAgent(Agent, HasObservables):
        energy = ContinuousObservable(
            initial_value=100.0, rate_func=lambda value, elapsed, agent: -1.0
        )

        def __init__(self, model):
            super().__init__(model)
            self.energy = 100.0
            self.triggered = False

        def on_threshold(self, signal):
            if signal.direction == "down":
                self.triggered = True

    model = SimpleModel()
    agent = MyAgent(model)

    agent.add_threshold("energy", 50.0, agent.on_threshold)

    # Manually set below threshold
    agent.energy = 30.0

    assert agent.triggered


def test_continuous_observable_no_threshold_cross_same_side():
    """Test that thresholds aren't triggered when staying on same side."""

    class MyAgent(Agent, HasObservables):
        energy = ContinuousObservable(
            initial_value=60.0, rate_func=lambda value, elapsed, agent: -1.0
        )

        def __init__(self, model):
            super().__init__(model)
            self.energy = 60.0
            self.triggered = False

        def on_threshold(self, signal):
            self.triggered = True

    model = SimpleModel()
    agent = MyAgent(model)

    agent.add_threshold("energy", 50.0, agent.on_threshold)

    def check_no_trigger():
        _ = agent.energy  # Move from 60 to 55 - doesn't cross threshold
        assert not agent.triggered

    model.simulator.schedule_event_absolute(check_no_trigger, 5.0)
    model.simulator.run_until(5.0)


def test_continuous_observable_exact_threshold_value():
    """Test behavior when value equals threshold exactly."""

    class MyAgent(Agent, HasObservables):
        energy = ContinuousObservable(
            initial_value=100.0, rate_func=lambda value, elapsed, agent: -1.0
        )

        def __init__(self, model):
            super().__init__(model)
            self.energy = 100.0
            self.crossings = []

        def on_threshold(self, signal):
            self.crossings.append(signal.direction)

    model = SimpleModel()
    agent = MyAgent(model)

    agent.add_threshold("energy", 50.0, agent.on_threshold)

    # Set exactly to threshold
    agent.energy = 50.0

    # Should trigger downward crossing (100 -> 50)
    assert len(agent.crossings) == 1
    assert agent.crossings[0] == "down"


def test_continuous_observable_with_computed():
    """Test ContinuousObservable working with Computed properties."""

    class MyAgent(Agent, HasObservables):
        energy = ContinuousObservable(
            initial_value=100.0, rate_func=lambda value, elapsed, agent: -1.0
        )
        is_hungry = Computable()

        def __init__(self, model):
            super().__init__(model)
            self.energy = 100.0
            # Pass self as an argument to Computed
            self.is_hungry = Computed(lambda agent: agent.energy < 50.0, self)

    model = SimpleModel()
    agent = MyAgent(model)

    # Not hungry initially
    assert not agent.is_hungry

    def check_hungry():
        # The Computed will access agent.energy, which will trigger
        # the ContinuousObservable to recalculate based on current time
        print(f"Energy at t=60: {agent.energy}")  # Debug
        print(f"Is hungry: {agent.is_hungry}")  # Debug
        assert agent.is_hungry  # Energy is now 40.0

    model.simulator.schedule_event_absolute(check_hungry, 60.0)
    model.simulator.run_until(60.0)


def test_continuous_observable_multiple_accesses_same_time():
    """Test that multiple accesses at same time don't recalculate."""

    class MyAgent(Agent, HasObservables):
        energy = ContinuousObservable(
            initial_value=100.0, rate_func=lambda value, elapsed, agent: -1.0
        )

        def __init__(self, model):
            super().__init__(model)
            self.energy = 100.0

    model = SimpleModel()
    agent = MyAgent(model)

    handler = Mock()
    agent.observe("energy", "change", handler)

    def check_multiple_access():
        # Multiple accesses at same time
        value1 = agent.energy
        value2 = agent.energy
        value3 = agent.energy

        # All should be same
        assert value1 == value2 == value3 == 90.0

        # Should only notify once
        handler.assert_called_once()

    model.simulator.schedule_event_absolute(check_multiple_access, 10.0)
    model.simulator.run_until(10.0)


def test_continuous_observable_zero_elapsed_time():
    """Test behavior when no time has elapsed."""

    class MyAgent(Agent, HasObservables):
        energy = ContinuousObservable(
            initial_value=100.0, rate_func=lambda value, elapsed, agent: -1.0
        )

        def __init__(self, model):
            super().__init__(model)
            self.energy = 100.0

    model = SimpleModel()
    agent = MyAgent(model)

    handler = Mock()
    agent.observe("energy", "change", handler)

    # Access without time passing (at t=0)
    value = agent.energy

    assert value == 100.0
    # Should not emit signal since nothing changed
    handler.assert_not_called()


def test_continuous_observable_numpy_float_compatibility():
    """Test compatibility with numpy float values (like from random)."""

    class MyAgent(Agent, HasObservables):
        energy = ContinuousObservable(
            initial_value=100.0, rate_func=lambda value, elapsed, agent: -1.0
        )

        def __init__(self, model, initial_energy):
            super().__init__(model)
            self.energy = initial_energy  # numpy float

    model = SimpleModel()

    # Create with numpy float
    numpy_value = np.float64(85.5)
    agent = MyAgent(model, numpy_value)

    # Should work without AttributeError
    assert agent.energy == 85.5

    def check_after_time():
        assert agent.energy == 75.5  # 85.5 - (1.0 * 10)

    model.simulator.schedule_event_absolute(check_after_time, 10.0)
    model.simulator.run_until(10.0)


def test_continuous_observable_with_create_agents():
    """Test ContinuousObservable with batch agent creation."""

    class MyAgent(Agent, HasObservables):
        energy = ContinuousObservable(
            initial_value=100.0, rate_func=lambda value, elapsed, agent: -1.0
        )

        def __init__(self, model, energy=100.0):
            super().__init__(model)
            self.energy = energy

    model = SimpleModel()

    # Create multiple agents with numpy array of energies
    initial_energies = np.random.random(10) * 100

    agents = MyAgent.create_agents(model, 10, energy=initial_energies)

    # All should be created successfully
    assert len(agents) == 10

    # Each should have correct energy
    for agent, expected_energy in zip(agents, initial_energies):
        assert abs(agent.energy - expected_energy) < 1e-10


def test_continuous_observable_with_abm_simulator():
    """Test ContinuousObservable with ABMSimulator (integer time steps)."""

    class StepModel(Model):
        def __init__(self):
            super().__init__()
            self.simulator = ABMSimulator()
            self.simulator.setup(self)

        def step(self):
            pass  # Model step gets called automatically

    class MyAgent(Agent, HasObservables):
        energy = ContinuousObservable(
            initial_value=100.0, rate_func=lambda value, elapsed, agent: -1.0
        )

        def __init__(self, model):
            super().__init__(model)
            self.energy = 100.0

    model = StepModel()
    agent = MyAgent(model)

    # Run for 10 steps (integer time)
    model.simulator.run_for(10)

    # Energy should have depleted
    assert agent.energy == 90.0


def test_continuous_observable_negative_values():
    """Test ContinuousObservable can go negative."""

    class MyAgent(Agent, HasObservables):
        energy = ContinuousObservable(
            initial_value=10.0, rate_func=lambda value, elapsed, agent: -1.0
        )

        def __init__(self, model):
            super().__init__(model)
            self.energy = 10.0

    model = SimpleModel()
    agent = MyAgent(model)

    def check_negative():
        assert agent.energy == -10.0

    model.simulator.schedule_event_absolute(check_negative, 20.0)
    model.simulator.run_until(20.0)


def test_continuous_observable_integration_with_wolf_sheep():
    """Integration test simulating wolf-sheep scenario."""

    class Animal(Agent, HasObservables):
        energy = ContinuousObservable(
            initial_value=100.0,
            rate_func=lambda value, elapsed, agent: -agent.metabolic_rate,
        )

        def __init__(self, model):
            super().__init__(model)
            self.metabolic_rate = 0.5
            self.energy = 100.0
            self.died = False

            # Death threshold - use add_threshold helper
            self.add_threshold("energy", 0.0, self._on_death)

        def _on_death(self, signal):
            if signal.direction == "down":
                self.died = True

        def eat(self):
            """Boost energy when eating."""
            self.energy += 20

    model = SimpleModel()
    agent = Animal(model)

    # Check survival at t=50
    def check_survival():
        assert agent.energy == 75.0  # 100 - (0.5 * 50)
        assert not agent.died
        # Eat
        agent.eat()
        assert agent.energy == 95.0

    # Check continued depletion at t=100
    def check_continued():
        assert agent.energy == 70.0  # 95 - (0.5 * 50)
        assert not agent.died

    # Check death at t=300
    def check_death():
        _ = agent.energy  # Trigger check
        assert agent.died

    model.simulator.schedule_event_absolute(check_survival, 50.0)
    model.simulator.schedule_event_absolute(check_continued, 100.0)
    model.simulator.schedule_event_absolute(check_death, 300.0)
    model.simulator.run_until(300.0)


def test_continuous_observable_multiple_agents_independent_values():
    """Test that multiple agents maintain independent continuous values."""

    class MyAgent(Agent, HasObservables):
        energy = ContinuousObservable(
            initial_value=100.0,
            rate_func=lambda value, elapsed, agent: -agent.metabolic_rate,
        )

        def __init__(self, model, metabolic_rate):
            super().__init__(model)
            self.metabolic_rate = metabolic_rate
            self.energy = 100.0

    model = SimpleModel()

    # Create agents with different metabolic rates
    agent1 = MyAgent(model, metabolic_rate=1.0)
    agent2 = MyAgent(model, metabolic_rate=2.0)
    agent3 = MyAgent(model, metabolic_rate=0.5)

    def check_values():
        # Each agent should deplete at their own rate
        assert agent1.energy == 90.0  # 100 - (1.0 * 10)
        assert agent2.energy == 80.0  # 100 - (2.0 * 10)
        assert agent3.energy == 95.0  # 100 - (0.5 * 10)

    model.simulator.schedule_event_absolute(check_values, 10.0)
    model.simulator.run_until(10.0)


def test_continuous_observable_multiple_agents_independent_thresholds():
    """Test that different agents can have different thresholds."""

    class MyAgent(Agent, HasObservables):
        energy = ContinuousObservable(
            initial_value=100.0, rate_func=lambda value, elapsed, agent: -1.0
        )

        def __init__(self, model, name):
            super().__init__(model)
            self.name = name
            self.energy = 100.0
            self.threshold_crossed = False

        def on_threshold(self, signal):
            if signal.direction == "down":
                self.threshold_crossed = True

    model = SimpleModel()

    # Create agents with different thresholds
    agent1 = MyAgent(model, "agent1")
    agent1.add_threshold("energy", 75.0, agent1.on_threshold)

    agent2 = MyAgent(model, "agent2")
    agent2.add_threshold("energy", 25.0, agent2.on_threshold)

    agent3 = MyAgent(model, "agent3")
    agent3.add_threshold("energy", 50.0, agent3.on_threshold)

    def check_at_30():
        # At t=30, all agents at energy=70
        _ = agent1.energy
        _ = agent2.energy
        _ = agent3.energy

        # Only agent1 should have crossed their threshold (75)
        assert agent1.threshold_crossed
        assert not agent2.threshold_crossed  # Hasn't reached 25 yet
        assert not agent3.threshold_crossed  # Hasn't reached 50 yet

    def check_at_55():
        # At t=55, all agents at energy=45
        _ = agent1.energy
        _ = agent2.energy
        _ = agent3.energy

        # agent1 and agent3 should have crossed
        assert agent1.threshold_crossed
        assert not agent2.threshold_crossed  # Still hasn't reached 25
        assert agent3.threshold_crossed  # Crossed 50

    def check_at_80():
        # At t=80, all agents at energy=20
        _ = agent1.energy
        _ = agent2.energy
        _ = agent3.energy

        # All should have crossed now
        assert agent1.threshold_crossed
        assert agent2.threshold_crossed  # Finally crossed 25
        assert agent3.threshold_crossed

    model.simulator.schedule_event_absolute(check_at_30, 30.0)
    model.simulator.schedule_event_absolute(check_at_55, 55.0)
    model.simulator.schedule_event_absolute(check_at_80, 80.0)
    model.simulator.run_until(80.0)


def test_continuous_observable_multiple_agents_same_threshold_different_callbacks():
    """Test that multiple agents can watch the same threshold value with different callbacks."""

    class MyAgent(Agent, HasObservables):
        energy = ContinuousObservable(
            initial_value=100.0, rate_func=lambda value, elapsed, agent: -1.0
        )

        def __init__(self, model, name):
            super().__init__(model)
            self.name = name
            self.energy = 100.0
            self.crossed_count = 0

        def on_threshold(self, signal):
            if signal.direction == "down":
                self.crossed_count += 1

    model = SimpleModel()

    # Create multiple agents, all watching threshold at 50
    agents = [MyAgent(model, f"agent{i}") for i in range(5)]

    for agent in agents:
        agent.add_threshold("energy", 50.0, agent.on_threshold)

    def check_crossings():
        # Access all agents' energy
        for agent in agents:
            _ = agent.energy

        # Each should have crossed independently
        for agent in agents:
            assert agent.crossed_count == 1

    model.simulator.schedule_event_absolute(check_crossings, 60.0)
    model.simulator.run_until(60.0)


def test_continuous_observable_agents_with_different_initial_values():
    """Test agents starting with different energy values."""

    class MyAgent(Agent, HasObservables):
        energy = ContinuousObservable(
            initial_value=100.0, rate_func=lambda value, elapsed, agent: -1.0
        )

        def __init__(self, model, initial_energy):
            super().__init__(model)
            self.energy = initial_energy

    model = SimpleModel()

    # Create agents with different starting energies
    agent1 = MyAgent(model, initial_energy=100.0)
    agent2 = MyAgent(model, initial_energy=50.0)
    agent3 = MyAgent(model, initial_energy=150.0)

    def check_values():
        # Each should deplete from their starting value
        assert agent1.energy == 90.0  # 100 - 10
        assert agent2.energy == 40.0  # 50 - 10
        assert agent3.energy == 140.0  # 150 - 10

    model.simulator.schedule_event_absolute(check_values, 10.0)
    model.simulator.run_until(10.0)


def test_continuous_observable_agent_interactions():
    """Test agents affecting each other's continuous observables."""

    class Predator(Agent, HasObservables):
        energy = ContinuousObservable(
            initial_value=50.0, rate_func=lambda value, elapsed, agent: -0.5
        )

        def __init__(self, model):
            super().__init__(model)
            self.energy = 50.0
            self.kills = 0

        def eat(self, prey):
            """Eat prey and gain energy."""
            self.energy += 20
            self.kills += 1
            prey.die()

    class Prey(Agent, HasObservables):
        energy = ContinuousObservable(
            initial_value=100.0, rate_func=lambda value, elapsed, agent: -1.0
        )

        def __init__(self, model):
            super().__init__(model)
            self.energy = 100.0
            self.alive = True

        def die(self):
            self.alive = False

    model = SimpleModel()

    predator = Predator(model)
    prey1 = Prey(model)
    prey2 = Prey(model)

    def predator_hunts():
        # Predator energy should have depleted
        assert predator.energy == 45.0  # 50 - (0.5 * 10)

        # Predator eats prey1
        predator.eat(prey1)

        # Predator gains energy
        assert predator.energy == 65.0  # 45 + 20
        assert not prey1.alive
        assert prey2.alive

    def check_final():
        # Predator continues depleting from boosted energy
        assert predator.energy == 60.0  # 65 - (0.5 * 10)

        # prey2 continues depleting
        assert prey2.energy == 80.0  # 100 - (1.0 * 20)
        assert prey2.alive

    model.simulator.schedule_event_absolute(predator_hunts, 10.0)
    model.simulator.schedule_event_absolute(check_final, 20.0)
    model.simulator.run_until(20.0)


def test_continuous_observable_batch_creation_with_thresholds():
    """Test batch agent creation where each agent has instance-specific thresholds."""

    class MyAgent(Agent, HasObservables):
        energy = ContinuousObservable(
            initial_value=100.0, rate_func=lambda value, elapsed, agent: -1.0
        )

        def __init__(self, model, critical_threshold):
            super().__init__(model)
            self.energy = 100.0
            self.critical_threshold = critical_threshold
            self.critical = False

            # Each agent watches their own critical threshold
            self.add_threshold("energy", critical_threshold, self.on_critical)

        def on_critical(self, signal):
            if signal.direction == "down":
                self.critical = True

    model = SimpleModel()

    # Create 10 agents with different critical thresholds
    thresholds = [90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0, 5.0]
    agents = [MyAgent(model, threshold) for threshold in thresholds]

    def check_at_45():
        # At t=45, all agents at energy=55
        for agent in agents:
            _ = agent.energy  # Trigger recalculation

        # Agents with thresholds > 55 should be critical
        for agent, threshold in zip(agents, thresholds):
            if threshold > 55:
                assert agent.critical
            else:
                assert not agent.critical

    model.simulator.schedule_event_absolute(check_at_45, 45.0)
    model.simulator.run_until(45.0)
