import typing
from typing import Literal

import mesa
from mesa.discrete_space import OrthogonalMooreGrid
from mesa.examples.advanced.pd_grid.agents import PDAgent
from mesa.experimental.scenarios import Scenario


class PrisonersDilemmaScenario(Scenario):
    """Scenario for Prisoner's Dilemma model."""

    width: int = 50
    height: int = 50
    activation_order: Literal["Sequential", "Random", "Simultaneous"] = "Random"
    payoff: None | dict[tuple[str, str], float] = None
    torus: bool = True


class PdGrid(mesa.Model):
    """Model class for iterated, spatial prisoner's dilemma model."""

    activation_regimes: typing.ClassVar[list[str]] = [
        "Sequential",
        "Random",
        "Simultaneous",
    ]

    # This dictionary holds the payoff for this agent,
    # keyed on: (my_move, other_move)

    payoff: typing.ClassVar[dict[tuple[str, str], float]] = {
        ("C", "C"): 1,
        ("C", "D"): 0,
        ("D", "C"): 1.6,
        ("D", "D"): 0,
    }

    def __init__(
        self,
        scenario=None,
    ):
        """
        Create a new Spatial Prisoners' Dilemma Model.

        Args:
            width, height: Grid size. There will be one agent per grid cell.
            activation_order: Can be "Sequential", "Random", or "Simultaneous".
                           Determines the agent activation regime.
            payoffs: (optional) Dictionary of (move, neighbor_move) payoffs.
        """
        if scenario is None:
            scenario = PrisonersDilemmaScenario()

        super().__init__(scenario=scenario)
        self.activation_order = scenario.activation_order
        self.grid = OrthogonalMooreGrid(
            (scenario.width, scenario.height), torus=scenario.torus, random=self.random
        )

        if scenario.payoff is not None:
            self.payoff = scenario.payoff

        PDAgent.create_agents(
            self, len(self.grid.all_cells.cells), cell=self.grid.all_cells.cells
        )

        self.datacollector = mesa.DataCollector(
            {
                "Cooperating_Agents": lambda m: len(
                    [a for a in m.agents if a.move == "C"]
                )
            }
        )

        self.running = True
        self.datacollector.collect(self)

    def step(self):
        # Activate all agents, based on the activation regime
        match self.activation_order:
            case "Sequential":
                self.agents.do(lambda a: (a.step(), a.advance()))
            case "Random":
                self.agents.shuffle_do(lambda a: (a.step(), a.advance()))
            case "Simultaneous":
                self.agents.do("step").do("advance")
            case _:
                raise ValueError(f"Unknown activation order: {self.activation_order}")

        # Collect data
        self.datacollector.collect(self)
