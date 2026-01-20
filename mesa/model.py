"""The model class for Mesa framework.

Core Objects: Model
"""

# Postpone annotation evaluation to avoid NameError from forward references (PEP 563). Remove once Python 3.14+ is required.
from __future__ import annotations

import random
import sys
import warnings
from collections.abc import Callable, Sequence

# mypy
from typing import Any

import numpy as np

from mesa.agent import Agent, AgentSet
from mesa.experimental.devs import Simulator
from mesa.experimental.scenarios import Scenario
from mesa.mesa_logging import create_module_logger, method_logger
from mesa.timeflow import Priority, RecurringEvent, RunControl, Scheduler, SimulationEvent

SeedLike = int | np.integer | Sequence[int] | np.random.SeedSequence
RNGLike = np.random.Generator | np.random.BitGenerator


_mesa_logger = create_module_logger()


# TODO: We can add `= Scenario` default type when Python 3.13+ is required
class Model[A: Agent, S: Scenario]:
    """Base class for models in the Mesa ABM library.

    This class serves as a foundational structure for creating agent-based models.
    It includes the basic attributes and methods necessary for initializing and
    running a simulation model.

    Type Parameters:
        A: The agent type used in this model
        S: The scenario type used in this model

    Attributes:
        running: A boolean indicating if the model should continue running.
        steps: the number of times `model.step()` has been called.
        time: the current simulation time. Automatically increments by 1.0
              with each step unless controlled by a discrete event simulator.
        random: a seeded python.random number generator.
        rng: a seeded numpy.random.Generator
        scenario: the scenario instance containing model parameters

    Notes:
        Model.agents returns the AgentSet containing all agents registered with the model. Changing
        the content of the AgentSet directly can result in strange behavior. If you want change the
        composition of this AgentSet, ensure you operate on a copy.

    """

    @property
    def scenario(self) -> S:
        """Return scenario instance."""
        return self._scenario

    @scenario.setter
    def scenario(self, scenario: S) -> None:
        """Set scenario instance."""
        self._scenario = scenario
        scenario.model = self

    @method_logger(__name__)
    def __init__(
        self,
        *args: Any,
        seed: float | None = None,
        rng: RNGLike | SeedLike | None = None,
        scenario: S | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a new model.

        Overload this method with the actual code to initialize the model. Always start with super().__init__()
        to initialize the model object properly.

        Args:
            args: arguments to pass onto super
            seed: the seed for the random number generator
            rng: Pseudorandom number generator state. When `rng` is None, a new `numpy.random.Generator` is created
                  using entropy from the operating system. Types other than `numpy.random.Generator` are passed to
                  `numpy.random.default_rng` to instantiate a `Generator`.
            scenario: the scenario specifying the computational experiment to run
            kwargs: keyword arguments to pass onto super

        Notes:
            you have to pass either seed or rng, but not both.

        """
        super().__init__(*args, **kwargs)
        self.running: bool = True
        self.steps: int = 0
        self.time: float = 0.0
        self.agent_id_counter: int = 1

        # Track if a simulator is controlling time
        self._simulator: Simulator | None = None

        # check if `scenario` is provided
        # and if so, whether rng is the same or not
        if scenario is not None:
            if rng is not None and (scenario.rng != rng):
                raise ValueError("rng and scenario.rng must be the same")
            else:
                rng = scenario.rng

        if (seed is not None) and (rng is not None):
            raise ValueError("you have to pass either rng or seed, not both")
        elif seed is None:
            self.rng: np.random.Generator = np.random.default_rng(rng)
            self._rng = (
                self.rng.bit_generator.state
            )  # this allows for reproducing the rng

            try:
                self.random = random.Random(rng)
            except TypeError:
                seed = int(self.rng.integers(np.iinfo(np.int32).max))
                self.random = random.Random(seed)
            self._seed = seed  # this allows for reproducing stdlib.random
        elif rng is None:
            warnings.warn(
                "The use of the `seed` keyword argument is deprecated, use `rng` instead. No functional changes.",
                FutureWarning,
                stacklevel=2,
            )

            self.random = random.Random(seed)
            self._seed = seed  # this allows for reproducing stdlib.random

            try:
                self.rng: np.random.Generator = np.random.default_rng(seed)
            except TypeError:
                rng = self.random.randint(0, sys.maxsize)
                self.rng: np.random.Generator = np.random.default_rng(rng)
            self._rng = self.rng.bit_generator.state

        # now that we have figured out the seed value for rng
        # we can set create a scenario with this if needed
        if scenario is None:
            scenario = Scenario(rng=seed)  # type: ignore[assignment]
        self.scenario = scenario

        # setup agent registration data structures
        self._agents: dict[
            A, None
        ] = {}  # the hard references to all agents in the model
        self._agents_by_type: dict[
            type[A], AgentSet[A]
        ] = {}  # a dict with an agentset for each class of agents
        self._all_agents: AgentSet[A] = AgentSet(
            [], random=self.random
        )  # an agenset with all agents

        # Add timeflow components
        self._scheduler = Scheduler(self)
        self._run_control = RunControl(self, self._scheduler)

        # Store reference to user's step method before we replace self.step
        self._user_step = self.step

        # Auto-schedule the internal step handler as the model's heartbeat
        self.step_event: RecurringEvent = self._scheduler.schedule(
            self._execute_step,
            interval=1,
            priority=Priority.DEFAULT,
        )

        # Replace step with backwards-compatible wrapper for direct calls
        self.step = self._deprecated_step_call

    def _execute_step(self) -> None:
        """Internal method that executes the user's step and updates counters."""
        self.steps += 1
        self._user_step()

    def _deprecated_step_call(self, *args, **kwargs) -> None:
        """Backwards-compatible step wrapper that advances time by 1.

        This allows old code that calls model.step() in a loop to continue working,
        while issuing a deprecation warning.
        """
        warnings.warn(
            "Calling model.step() directly to advance time is deprecated and will be "
            "removed in Mesa 4.0. Use model.run_for(1) or model.run_until() instead. "
            "See: https://mesa.readthedocs.io/latest/migration_guide.html#unified-time-api",
            PendingDeprecationWarning,
            stacklevel=2,
        )
        self.run_for(1)

    @property
    def agents(self) -> AgentSet[A]:
        """Provides an AgentSet of all agents in the model, combining agents from all types."""
        return self._all_agents

    @agents.setter
    def agents(self, agents: Any) -> None:
        raise AttributeError(
            "You are trying to set model.agents. In Mesa 3.0 and higher, this attribute is "
            "used by Mesa itself, so you cannot use it directly anymore."
            "Please adjust your code to use a different attribute name for custom agent storage."
        )

    @property
    def agent_types(self) -> list[type]:
        """Return a list of all unique agent types registered with the model."""
        return list(self._agents_by_type.keys())

    @property
    def agents_by_type(self) -> dict[type[A], AgentSet[A]]:
        """A dictionary where the keys are agent types and the values are the corresponding AgentSets."""
        return self._agents_by_type

    def register_agent(self, agent: A):
        """Register the agent with the model.

        Args:
            agent: The agent to register.

        Notes:
            This method is called automatically by ``Agent.__init__``, so there
            is no need to use this if you are subclassing Agent and calling its
            super in the ``__init__`` method.
        """
        self._agents[agent] = None
        agent.unique_id = self.agent_id_counter
        self.agent_id_counter += 1

        # because AgentSet requires model, we cannot use defaultdict
        # tricks with a function won't work because model then cannot be pickled
        try:
            self._agents_by_type[type(agent)].add(agent)
        except KeyError:
            self._agents_by_type[type(agent)] = AgentSet(
                [
                    agent,
                ],
                random=self.random,
            )

        self._all_agents.add(agent)
        _mesa_logger.debug(
            f"registered {agent.__class__.__name__} with agent_id {agent.unique_id}"
        )

    def deregister_agent(self, agent: A):
        """Deregister the agent with the model.

        Args:
            agent: The agent to deregister.

        Notes:
            This method is called automatically by ``Agent.remove``

        """
        del self._agents[agent]
        self._agents_by_type[type(agent)].remove(agent)
        self._all_agents.remove(agent)
        _mesa_logger.debug(f"deregistered agent with agent_id {agent.unique_id}")

    def run_model(self) -> None:
        """Run the model until the end condition is reached.

        Overload as needed.
        """
        while self.running:
            self.step()

    def step(self) -> None:
        """A single step. Fill in here."""

    def reset_randomizer(self, seed: int | None = None) -> None:
        """Reset the model random number generator.

        Args:
            seed: A new seed for the RNG; if None, reset using the current seed
        """
        warnings.warn(
            "The use of the `seed` keyword argument is deprecated, use `rng` instead. No functional changes.",
            FutureWarning,
            stacklevel=2,
        )

        if seed is None:
            seed = self._seed
        self.random.seed(seed)
        self._seed = seed

    def reset_rng(self, rng: RNGLike | SeedLike | None = None) -> None:
        """Reset the model random number generator.

        Args:
            rng: A new seed for the RNG; if None, reset using the current seed
        """
        if rng is None:
            # Restore from saved initial state
            bg_class = getattr(np.random, self._rng["bit_generator"])
            bg = bg_class()
            bg.state = self._rng
            self.rng = np.random.Generator(bg)
        else:
            self.rng = np.random.default_rng(rng)
            self._rng = self.rng.bit_generator.state

    def remove_all_agents(self):
        """Remove all agents from the model.

        Notes:
            This method calls agent.remove for all agents in the model. If you need to remove agents from
            e.g., a SingleGrid, you can either explicitly implement your own agent.remove method or clean this up
            near where you are calling this method.

        """
        # we need to wrap keys in a list to avoid a RunTimeError: dictionary changed size during iteration
        for agent in list(self._agents.keys()):
            agent.remove()

    def schedule(
            self,
            callback: Callable,
            start_at: int | float | None = None,
            start_after: int | float | None = None,
            interval: int | float | Callable | None = None,
            count: int | None = None,
            end_at: int | float | None = None,
            end_after: int | float | None = None,
            priority: Priority = Priority.DEFAULT,
            args: list[Any] | None = None,
            kwargs: dict[str, Any] | None = None,
    ) -> SimulationEvent | RecurringEvent:
        """Schedule an event with flexible timing options.

        See Scheduler.schedule() for full documentation.

        Returns:
            SimulationEvent for one-off events, RecurringEvent for recurring events.
        """
        return self._scheduler.schedule(
            callback,
            start_at=start_at,
            start_after=start_after,
            interval=interval,
            count=count,
            end_at=end_at,
            end_after=end_after,
            priority=priority,
            args=args,
            kwargs=kwargs,
        )

    def schedule_at(
        self,
        callback: Callable,
        time: int | float,
        priority: Priority = Priority.DEFAULT,
        args: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> SimulationEvent:
        """Schedule a one-off event at an absolute time."""
        return self._scheduler.schedule_at(callback, time, priority, args, kwargs)

    def schedule_after(
        self,
        callback: Callable,
        delay: int | float,
        priority: Priority = Priority.DEFAULT,
        args: list[Any] | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> SimulationEvent:
        """Schedule a one-off event after a delay from current time."""
        return self._scheduler.schedule_after(callback, delay, priority, args, kwargs)

    def cancel_event(self, event: SimulationEvent) -> None:
        """Cancel a scheduled event."""
        self._scheduler.cancel(event)

    def run_until(self, end_time: int | float) -> None:
        """Run the model until the specified time."""
        self._run_control.run_until(end_time)

    def run_for(self, duration: int | float) -> None:
        """Run the model for a specific duration."""
        self._run_control.run_for(duration)

    def run_while(self, condition: Callable[[Model], bool]) -> None:
        """Run the model while a condition remains true."""
        self._run_control.run_while(condition)

    def run_next_event(self) -> bool:
        """Execute the next scheduled event."""
        return self._run_control.run_next_event()
