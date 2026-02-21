"""Test removal of agents."""

import gc
import unittest
from functools import partial
from unittest.mock import MagicMock, patch

import numpy as np

from mesa import Agent, Model
from mesa.datacollection import DataCollector
from mesa.time import EventGenerator, Schedule
from mesa.time.events import Priority


class LifeTimeModel(Model):
    """Simple model for running models with a finite life."""

    def __init__(self, agent_lifetime=1, n_agents=10, rng=None):  # noqa: D107
        super().__init__(rng=rng)

        self.agent_lifetime = agent_lifetime
        self.n_agents = n_agents

        # keep track of the the remaining life of an agent and
        # how many ticks it has seen
        self.datacollector = DataCollector(
            agent_reporters={
                "remaining_life": lambda a: a.remaining_life,
                "steps": lambda a: a.steps,
            }
        )

        self.current_ID = 0

        for _ in range(n_agents):
            FiniteLifeAgent(self.agent_lifetime, self)

        self.datacollector.collect(self)

    def step(self):
        """Add agents back to n_agents in each step."""
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)

        if len(self.agents) < self.n_agents:
            for _ in range(self.n_agents - len(self.agents)):
                FiniteLifeAgent(self.agent_lifetime, self)

    def run_model(self, step_count=100):  # noqa: D102
        for _ in range(step_count):
            self.step()


class FiniteLifeAgent(Agent):
    """An agent that is supposed to live for a finite number of ticks.

    Also has a 10% chance of dying in each tick.
    """

    def __init__(self, lifetime, model):  # noqa: D107
        super().__init__(model)
        self.remaining_life = lifetime
        self.steps = 0
        self.model = model

    def step(self):  # noqa: D102
        deactivated = self.deactivate()
        if not deactivated:
            self.steps += 1  # keep track of how many ticks are seen
            if np.random.binomial(1, 0.1) != 0:  # 10% chance of dying
                self.remove()

    def deactivate(self):  # noqa: D102
        self.remaining_life -= 1
        if self.remaining_life < 0:
            self.remove()
            return True
        return False


class TestAgentLifespan(unittest.TestCase):  # noqa: D101
    def setUp(self):  # noqa: D102
        self.model = LifeTimeModel()
        self.model.run_model()
        self.df = self.model.datacollector.get_agent_vars_dataframe()
        self.df = self.df.reset_index()

    def test_ticks_seen(self):
        """Each agent should be activated no more than one time."""
        assert self.df.steps.max() == 1

    def test_agent_lifetime(self):  # noqa: D102
        lifetimes = self.df.groupby(["AgentID"]).agg({"Step": len})
        assert lifetimes.Step.max() == 2


class TestEventGeneratorMemoryLeak(unittest.TestCase):
    """Tests EventGenerator error handling, memory behavior, and state restoration."""

    def test_error_cases_and_valid_usage(self):
        """Test all error cases + valid usage patterns."""
        model = Model()
        schedule = Schedule(interval=1.0)

        # Test 1: Non-callable → TypeError
        with self.assertRaises(TypeError):
            EventGenerator(model, 42, schedule)

        # Test 2: Non-weakly-referenceable callable → TypeError
        class NoWeakRef:
            __slots__ = ()

            def __call__(self):
                pass

        with self.assertRaises(TypeError):
            EventGenerator(model, NoWeakRef(), schedule)

        # Test 3: Inline lambda (no strong ref) → ValueError
        mock_weakref = MagicMock()
        mock_weakref.return_value = None

        with (
            patch("mesa.time.events.ref", return_value=mock_weakref),
            self.assertRaises(ValueError),
        ):
            EventGenerator(model, lambda: 10, schedule)

        # Test 4: Inline partial (no strong ref) → ValueError
        def my_func(x, y):
            return x + y

        with (
            patch("mesa.time.events.ref", return_value=mock_weakref),
            self.assertRaises(ValueError) as cm,
        ):
            EventGenerator(model, partial(my_func, 1, 2), schedule)

        self.assertIn("garbage collected", str(cm.exception).lower())

        # Test 5: Assigned function (strong ref) → works fine
        def assigned_lambda():
            return 5

        gen = EventGenerator(model, assigned_lambda, schedule)
        self.assertIsNotNone(gen._function)

        # Test 6: Assigned partial (strong ref) → works fine
        assigned_partial = partial(my_func, 1, 2)
        gen = EventGenerator(model, assigned_partial, schedule)
        self.assertIsNotNone(gen._function)

    def test_state_preparation_and_restoration(self):
        """Test __getstate__ and __setstate__ directly (no actual pickling)."""
        model = Model()
        schedule = Schedule(interval=1.0)

        # Create a simple callable
        def test_func():
            return "hello"

        # Create generator
        gen = EventGenerator(model, test_func, schedule)

        # 1. Test __getstate__
        state = gen.__getstate__()

        # Verify state contains expected keys
        self.assertIn("_fn_strong", state)
        self.assertIn("_function", state)
        self.assertIsNone(state["_function"])

        # Verify _fn_strong is the actual function
        self.assertEqual(state["_fn_strong"](), "hello")

        # 2. Test __setstate__
        new_gen = EventGenerator.__new__(EventGenerator)
        new_gen.__setstate__(state)

        # Verify weak reference was recreated correctly
        self.assertIsNotNone(new_gen._function)
        self.assertEqual(new_gen.function(), "hello")

        # Verify other state was preserved
        self.assertEqual(new_gen.schedule, schedule)
        self.assertEqual(new_gen.priority, Priority.DEFAULT)

        # 3. Cover branch where fn is None in __setstate__
        state_with_none = state.copy()
        state_with_none["_fn_strong"] = None

        none_gen = EventGenerator.__new__(EventGenerator)
        none_gen.__setstate__(state_with_none)

        self.assertIsNone(none_gen._function)

    def test_no_op_during_execution_when_weakref_dies(self):
        """Test generator stops silently when weakref dies during execution."""
        model = Model()
        schedule = Schedule(interval=1.0)

        # Track calls
        call_count = [0]

        def temp_func():
            call_count[0] += 1

        # Create and start generator
        gen = EventGenerator(model, temp_func, schedule)
        gen.start()

        # First execution
        model.run_for(1.0)
        self.assertEqual(call_count[0], 1)
        self.assertTrue(gen.is_active)

        # Remove strong reference
        del temp_func
        gc.collect()

        # Second execution - should trigger no-op and stop silently
        model.run_for(1.0)

        # Verify generator stopped (no error raised)
        self.assertFalse(gen.is_active)
        self.assertEqual(call_count[0], 1)  # No additional calls


if __name__ == "__main__":
    unittest.main()
