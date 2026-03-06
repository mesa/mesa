"""Forager model: demonstrates Mesa's experimental Action API."""

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.examples.basic.forager.agents import ForagerAgent
from mesa.experimental.scenarios import Scenario


class ForagerScenario(Scenario):
    """Parameters for the Forager model.

    Attributes:
        n_agents: Number of forager agents.
        threat_prob: Per-agent, per-step probability of a predator threat (0–1).
        forage_duration: Sim-time units needed to complete one forage cycle.
        flee_duration: Sim-time units needed for an agent to flee a threat.
        food_per_forage: Maximum food awarded for one complete forage cycle.
    """

    n_agents: int = 20
    threat_prob: float = 0.1
    forage_duration: float = 5.0
    flee_duration: float = 1.0
    food_per_forage: float = 1.0


class ForagerModel(Model):
    """A model where agents forage for food and can be interrupted by threats.

    Each step every agent either:
    - Reacts to a random threat by calling request_action(flee), which
      preempts any ongoing foraging (forage is re-queued with remainder
      progress and resumes automatically after fleeing).
    - Starts foraging if currently idle.

    This model demonstrates:
    - reschedule_on_interrupt="remainder": agents don't lose forage progress.
    - request_action priority preemption: flee (priority 10) beats forage (priority 1).
    - Automatic action queue draining: forage resumes as soon as flee finishes.
    """

    def __init__(self, scenario: ForagerScenario | None = None, **kwargs):
        """Create a ForagerModel.

        Args:
            scenario: ForagerScenario containing model parameters. If None,
                a default scenario is created (kwargs are forwarded to it).
            **kwargs: Forwarded to ForagerScenario when scenario is None.
                Accepted keys: n_agents, threat_prob, forage_duration,
                flee_duration, food_per_forage.
        """
        if scenario is None:
            scenario = ForagerScenario(**kwargs)
        super().__init__(scenario=scenario)

        self.threat_prob = scenario.threat_prob

        ForagerAgent.create_agents(
            self,
            scenario.n_agents,
            food_per_forage=scenario.food_per_forage,
            forage_duration=scenario.forage_duration,
            flee_duration=scenario.flee_duration,
        )

        self.datacollector = DataCollector(
            model_reporters={
                "total_food": lambda m: round(
                    sum(a.food_collected for a in m.agents), 2
                ),
                "foraging_agents": lambda m: sum(
                    1
                    for a in m.agents
                    if a.is_busy and a.current_action.name == "forage"
                ),
                "fleeing_agents": lambda m: sum(
                    1
                    for a in m.agents
                    if a.is_busy and a.current_action.name == "flee"
                ),
                "idle_agents": lambda m: sum(
                    1 for a in m.agents if not a.is_busy
                ),
                "total_threats": lambda m: sum(
                    a.threats_faced for a in m.agents
                ),
            },
            agent_reporters={
                "food_collected": "food_collected",
                "threats_faced": "threats_faced",
                "current_action": lambda a: a.current_action.name
                if a.current_action
                else None,
            },
        )

    def step(self) -> None:
        """Advance the model by one step."""
        self.agents.do("step")
        self.datacollector.collect(self)
