"""The model class for Mesa framework.

Core Objects: Model
"""

# Postpone annotation evaluation to avoid NameError from forward references (PEP 563). Remove once Python 3.14+ is required.
from __future__ import annotations

import random
from collections.abc import Callable, Sequence

# mypy
from typing import TYPE_CHECKING, Any

import numpy as np

from mesa.experimental.data_collection.dataset import DataRegistry
from mesa.experimental.mesa_signals import (
    HasObservables,
    ModelSignals,
    Observable,
    emit,
)

if TYPE_CHECKING:
    from mesa.experimental.devs import Simulator

from mesa.agent import Agent
from mesa.agentset import _HardKeyAgentSet
from mesa.experimental.scenarios import Scenario
from mesa.mesa_logging import create_module_logger, method_logger
from mesa.time import (
    Event,
    EventGenerator,
    EventList,
    Priority,
    Schedule,
)

SeedLike = int | np.integer | Sequence[int] | np.random.SeedSequence
RNGLike = np.random.Generator | np.random.BitGenerator


_mesa_logger = create_module_logger()


# TODO: We can add `= Scenario` default type when Python 3.13+ is required
class Model[A: Agent, S: Scenario](HasObservables):
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

    # fixme how can we declare that "agents" is observable?
    time = (
        Observable()
    )  # we can now just subscribe to change events on the observable time

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
        rng: RNGLike | SeedLike | None = None,
        scenario: S | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a new model.

        Overload this method with the actual code to initialize the model. Always start with super().__init__()
        to initialize the model object properly.

        Args:
            args: arguments to pass onto super
            rng: Pseudorandom number generator state. When `rng` is None, a new `numpy.random.Generator` is created
                  using entropy from the operating system. Types other than `numpy.random.Generator` are passed to
                  `numpy.random.default_rng` to instantiate a `Generator`. `rng` is also used to try to seed a
                  `random.Random` instance, if this fails, a random integer will be generated using the seeded
                  numpy random number generator with which to seed `random.Random`.
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
        self.rng = None
        self._rng = None
        self.random = None
        self._seed = None

        # Track if a simulator is controlling time
        self._simulator: Simulator | None = None

        # Event list for event-based execution
        self._event_list: EventList = EventList()
        # Strong references to active EventGenerators (prevent GC)
        self._event_generators: set[EventGenerator] = set()

        # check if `scenario` is provided
        # and if so, whether rng is the same or not
        if scenario is not None:
            if rng is not None and (scenario.rng != rng):
                raise ValueError("rng and scenario.rng must be the same")
            else:
                rng = scenario.rng

        self.reset_rng(rng)

        # now that we have figured out the seed value for rng
        # we can set create a scenario with this if needed
        if scenario is None:
            scenario = Scenario(rng=rng)  # type: ignore[assignment]
        self.scenario = scenario

        # Store user's step method and create the default step schedule.
        # Uses EventGenerator to schedule _do_step every 1.0 time units.
        self._user_step = self.step
        self._default_schedule: EventGenerator = EventGenerator(
            self,
            self._do_step,
            Schedule(interval=1.0, start=1.0),
            priority=Priority.HIGH,
        ).start()
        self.step = self._wrapped_step

        # setup agent registration data structures
        self._agents_by_type: dict[
            type[A], _HardKeyAgentSet[A]
        ] = {}  # a dict with an agentset for each class of agents
        self._all_agents: _HardKeyAgentSet[A] = _HardKeyAgentSet(
            [], random=self.random
        )  # an agenset with all agents

        self.data_registry = DataRegistry()

    def _wrapped_step(self) -> None:
        """Advance time by one unit, processing any scheduled events."""
        self._advance_time(self.time + 1)

    def _advance_time(self, until: float) -> None:
        """Advance time to the given point, processing events along the way.

        Args:
            until: The time to advance to

        """
        while True:
            try:
                event = self._event_list.pop_event()
            except IndexError:
                break

            if event.time <= until:
                self.time = event.time
                event.execute()
            else:
                self._event_list.add_event(event)
                break

        self.time = until

    def _do_step(self) -> None:
        """Execute one step. Rescheduling is handled by the EventGenerator."""
        self.steps += 1
        _mesa_logger.info(f"Step {self.steps} at time {self.time}")
        self._user_step()

    @property
    def agents(self) -> _HardKeyAgentSet[A]:
        """Provides a _HardKeyAgentSet of all agents in the model, combining agents from all types.

        Returns:
            _HardKeyAgentSet: The agent set containing all agents with strong references.

        Warning:
            This returns the actual internal _HardKeyAgentSet used by Mesa for agent registration
            and tracking. It uses strong references to prevent premature garbage collection and reduce performance overhead
            caused by weak reference management.

            **Do not modify this AgentSet directly** (e.g., by adding or removing agents manually).
            Direct modifications can break the model's agent tracking system and cause unexpected
            behavior. Instead:

            - Use ``Agent()`` to create new agents (automatically registers them)
            - Use ``agent.remove()`` to remove agents (automatically deregisters them)
            - For read-only operations or transformations, work on a copy: ``model.agents.copy()``

        Notes:
            This is Mesa's core agent registration system. All agents created via ``Agent.__init__``
            are automatically registered here.
        """
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
    def agents_by_type(self) -> dict[type[A], _HardKeyAgentSet[A]]:
        """A dictionary where keys are agent types and values are the corresponding _HardKeyAgentSets.

        Returns:
            dict[type[A], _HardKeyAgentSet[A]]: Dictionary mapping agent types to their AgentSets.

        Warning:
            Each AgentSet in this dictionary is a _HardKeyAgentSet with strong references,
            forming part of Mesa's core agent registration system.

            **Do not modify these AgentSets directly**. Direct modifications can break agent
            tracking and cause unexpected behavior. Instead:

            - Use ``Agent()`` to create new agents (automatically registers them)
            - Use ``agent.remove()`` to remove agents (automatically deregisters them)
            - For read-only operations, work on copies: ``model.agents_by_type[AgentType].copy()``

        Notes:
            This is part of Mesa's core agent registration system. All agents are automatically
            registered in the appropriate type-specific AgentSet when created via ``Agent.__init__``.
        """
        return self._agents_by_type

    @emit("agents", ModelSignals.AGENT_ADDED)
    def register_agent(self, agent: A):
        """Register the agent with the model.

        Args:
            agent: The agent to register.

        Notes:
            This method is called automatically by ``Agent.__init__``, so there
            is no need to use this if you are subclassing Agent and calling its
            super in the ``__init__`` method.
        """
        # Add to main storage
        self._all_agents.add(agent)
        agent.unique_id = self.agent_id_counter
        self.agent_id_counter += 1

        # because AgentSet requires model, we cannot use defaultdict
        # tricks with a function won't work because model then cannot be pickled
        try:
            self._agents_by_type[type(agent)].add(agent)
        except KeyError:
            self._agents_by_type[type(agent)] = _HardKeyAgentSet(
                [agent],
                random=self.random,
            )

        _mesa_logger.debug(
            f"registered {agent.__class__.__name__} with agent_id {agent.unique_id}"
        )

    @emit("agents", ModelSignals.AGENT_REMOVED)
    def deregister_agent(self, agent: A):
        """Deregister the agent with the model.

        Args:
            agent: The agent to deregister.

        Notes:
            This method is called automatically by ``Agent.remove``

        """
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

    def reset_rng(self, rng: RNGLike | SeedLike | None = None) -> None:
        """Reset the model random number generator.

        Args:
            rng: A new seed for the RNG; if None, reset using the current seed
        """
        failed = True
        if rng is None:
            # Restore from saved initial state
            try:
                bg_class = getattr(np.random, self._rng["bit_generator"])
            except TypeError:
                rng = None
            else:
                failed = False
                bg = bg_class()
                bg.state = self._rng
                self.rng = np.random.Generator(bg)

        if failed:
            self.rng: np.random.Generator = np.random.default_rng(rng)

        self._rng = self.rng.bit_generator.state  # this allows for reproducing the rng

        try:
            self.random = random.Random(rng)
        except TypeError:
            seed = int(self.rng.integers(np.iinfo(np.int32).max))
            self.random = random.Random(seed)
        else:
            seed = rng
        self._seed = seed  # this allows for reproducing stdlib.random

    def remove_all_agents(self):
        """Remove all agents from the model.

        Notes:
            This method calls agent.remove for all agents in the model. If you need to remove agents from
            e.g., a SingleGrid, you can either explicitly implement your own agent.remove method or clean this up
            near where you are calling this method.

        """
        # we need to wrap keys in a list to avoid a RunTimeError: dictionary changed size during iteration
        for agent in list(self._all_agents):
            agent.remove()

        self.data_registry.close()  # this is needed to ensure GC works properly

    ### Event scheduling and time progression methods ###
    def schedule_event(
        self,
        function: Callable,
        *,
        at: float | None = None,
        after: float | None = None,
        priority: Priority = Priority.DEFAULT,
    ) -> Event:
        """Schedule a one-off event.

        Args:
            function: The callable to execute
            at: Absolute time to execute (mutually exclusive with after)
            after: Relative time from now to execute (mutually exclusive with at)
            priority: Priority level for the event

        Returns:
            The scheduled Event (can be used to cancel)

        Raises:
            ValueError: If both or neither of at/after are specified
        """
        if (at is None) == (after is None):
            raise ValueError("Specify exactly one of 'at' or 'after'")

        time = at if at is not None else self.time + after
        event = Event(time, function, priority=priority)
        self._event_list.add_event(event)
        return event

    def schedule_recurring(
        self,
        function: Callable,
        schedule: Schedule,
        priority: Priority = Priority.DEFAULT,
    ) -> EventGenerator:
        """Schedule a recurring event based on a Schedule.

        Args:
            function: The callable to execute repeatedly
            schedule: The Schedule defining when events occur
            priority: Priority level for generated events

        Returns:
            The EventGenerator (can be used to stop)
        """
        generator = EventGenerator(self, function, schedule, priority)
        generator.start()
        return generator

    def run_for(self, duration: float | int) -> None:
        """Run the model for the specified duration.

        Args:
            duration: Time units to advance
        """
        self._advance_time(self.time + duration)

    def run_until(self, end_time: float | int) -> None:
        """Run the model until the specified time.

        Args:
            end_time: Absolute time to run until
        """
        self._advance_time(end_time)
