import mesa
from mesa.discrete_space import OrthogonalMooreGrid, OrthogonalVonNeumannGrid  #
from mesa.examples.advanced.epstein_civil_violence.agents import (
    Citizen,
    CitizenState,
    Cop,
)
from mesa.experimental.scenarios import Scenario


class EpsteinScenario(Scenario):
    """Scenario parameters for Epstein Civil Violence model."""

    citizen_density: float = 0.7
    cop_density: float = 0.074
    citizen_vision: int = 7
    cop_vision: int = 7
    legitimacy: float = 0.8
    max_jail_term: int = 1000
    active_threshold: float = 0.1
    arrest_prob_constant: float = 2.3
    movement: bool = True
    max_iters: int = 1000
    activation_order: str = "Random"
    grid_type: str = "Von Neumann"


class EpsteinCivilViolence(mesa.Model):
    """
    Model 1 from "Modeling civil violence: An agent-based computational
    approach," by Joshua Epstein.
    http://www.pnas.org/content/99/suppl_3/7243.full

    Args:
        height: grid height
        width: grid width
        scenario: EpsteinScenario object containing model parameters.
    """

    def __init__(
        self,
        width=40,
        height=40,
        scenario=None,
        **kwargs,
    ):
        if scenario is None:
            # Extract rng/seed from kwargs to avoid conflicts with super().__init__
            rng = kwargs.get("rng", kwargs.get("seed"))
            scenario = EpsteinScenario(rng=rng)

        super().__init__(scenario=scenario, **kwargs)

        self.movement = scenario.movement
        self.max_iters = scenario.max_iters
        self.activation_order = scenario.activation_order

        match scenario.grid_type:
            case "Moore":
                self.grid = OrthogonalMooreGrid(
                    (width, height), capacity=1, torus=True, random=self.random
                )
            case "Von Neumann":
                self.grid = OrthogonalVonNeumannGrid(
                    (width, height), capacity=1, torus=True, random=self.random
                )
            case _:
                raise ValueError(f"Unknown value of grid_type: {scenario.grid_type}")

        model_reporters = {
            "active": CitizenState.ACTIVE.name,
            "quiet": CitizenState.QUIET.name,
            "arrested": CitizenState.ARRESTED.name,
        }
        agent_reporters = {
            "jail_sentence": lambda a: getattr(a, "jail_sentence", None),
            "arrest_probability": lambda a: getattr(a, "arrest_probability", None),
        }
        self.datacollector = mesa.DataCollector(
            model_reporters=model_reporters, agent_reporters=agent_reporters
        )
        if scenario.cop_density + scenario.citizen_density > 1:
            raise ValueError("Cop density + citizen density must be less than 1")

        for cell in self.grid.all_cells:
            klass = self.random.choices(
                [Citizen, Cop, None],
                cum_weights=[
                    scenario.citizen_density,
                    scenario.citizen_density + scenario.cop_density,
                    1,
                ],
            )[0]

            if klass == Cop:
                cop = Cop(self)
                cop.move_to(cell)
            elif klass == Citizen:
                citizen = Citizen(self)
                citizen.move_to(cell)

        self.running = True
        self._update_counts()
        self.datacollector.collect(self)

    def step(self):
        """
        Advance the model by one step and collect data.
        """
        match self.activation_order:
            case "Random":
                self.agents.shuffle_do("step")
            case "Sequential":
                self.agents.do("step")
            case _:
                raise ValueError(
                    f"unknown value of activation_order: {self.activation_order}"
                )

        self._update_counts()
        self.datacollector.collect(self)

        if self.steps > self.max_iters:
            self.running = False

    def _update_counts(self):
        """Helper function for counting nr. of citizens in given state."""
        counts = self.agents_by_type[Citizen].groupby("state").count()

        for state in CitizenState:
            setattr(self, state.name, counts.get(state, 0))
