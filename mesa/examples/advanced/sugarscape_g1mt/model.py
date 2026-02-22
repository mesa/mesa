from pathlib import Path

import numpy as np

import mesa
from mesa.discrete_space import OrthogonalVonNeumannGrid
from mesa.discrete_space.property_layer import PropertyLayer
from mesa.examples.advanced.sugarscape_g1mt.agents import Trader
from mesa.experimental.scenarios import Scenario


# Helper Functions
def flatten(list_of_lists):
    """
    helper function for model datacollector for trade price
    collapses agent price list into one list
    """
    return [item for sublist in list_of_lists for item in sublist]


def geometric_mean(list_of_prices):
    """
    find the geometric mean of a list of prices
    """
    # protects against an invalid value if no prices
    if len(list_of_prices) == 0:
        return -1
    return np.exp(np.log(list_of_prices).mean())


class SugarScapeScenario(Scenario):
    """Sugarscape scenario class."""

    initial_population: int = 200
    endowment_min: int = 25
    endowment_max: int = 50
    metabolism_min: int = 1
    metabolism_max: int = 5
    vision_min: int = 1
    vision_max: int = 5
    enable_trade: bool = True


class SugarscapeG1mt(mesa.Model):
    """
    Manager class to run Sugarscape with Traders
    """

    def __init__(self, scenario=None):
        if scenario is None:
            scenario = SugarScapeScenario()

        super().__init__(scenario=scenario)
        # Initiate width and height of sugarscape
        self.width = 50
        self.height = 50

        # Initiate population attributes
        self.enable_trade = self.scenario.enable_trade
        self.running = True

        # initiate mesa grid class
        self.grid = OrthogonalVonNeumannGrid(
            (self.width, self.height), torus=False, random=self.random
        )
        # initiate datacollector
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "#Traders": lambda m: len(m.agents),
                "Trade Volume": lambda m: sum(len(a.trade_partners) for a in m.agents),
                "Price": lambda m: geometric_mean(
                    flatten([a.prices for a in m.agents])
                ),
            },
            agent_reporters={"Trade Network": "trade_partners"},
        )

        # read in landscape file from supplementary material
        self.sugar_distribution = np.genfromtxt(Path(__file__).parent / "sugar-map.txt")
        self.spice_distribution = np.flip(self.sugar_distribution, 1)

        self.grid.add_property_layer(
            PropertyLayer.from_data("sugar", self.sugar_distribution)
        )
        self.grid.add_property_layer(
            PropertyLayer.from_data("spice", self.spice_distribution)
        )

        n = self.scenario.initial_population
        Trader.create_agents(
            self,
            self.scenario.initial_population,
            self.random.choices(self.grid.all_cells.cells, k=n),
            sugar=self.rng.integers(
                self.scenario.endowment_min,
                self.scenario.endowment_min,
                (n,),
                endpoint=True,
            ),
            spice=self.rng.integers(
                self.scenario.endowment_min,
                self.scenario.endowment_min,
                (n,),
                endpoint=True,
            ),
            metabolism_sugar=self.rng.integers(
                self.scenario.metabolism_min,
                self.scenario.metabolism_max,
                (n,),
                endpoint=True,
            ),
            metabolism_spice=self.rng.integers(
                self.scenario.metabolism_min,
                self.scenario.metabolism_max,
                (n,),
                endpoint=True,
            ),
            vision=self.rng.integers(
                self.scenario.vision_min,
                self.scenario.vision_max,
                (n,),
                endpoint=True,
            ),
        )

    def step(self):
        """
        Unique step function that does staged activation of sugar and spice
        and then randomly activates traders
        """
        # let resources regrow
        self.grid.sugar.data = np.minimum(
            self.grid.sugar.data + 1, self.sugar_distribution
        )
        self.grid.spice.data = np.minimum(
            self.grid.spice.data + 1, self.spice_distribution
        )

        # step trader agents
        self.agents.shuffle_do("step")
        if self.enable_trade:
            self.agents.shuffle_do("trade_with_neighbors")
        self.datacollector.collect(self)

    def run_model(self, step_count=1000):
        for _ in range(step_count):
            self.step()
