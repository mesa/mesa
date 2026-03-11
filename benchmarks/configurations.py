"""configurations for benchmarks."""

from mesa.examples import (
    BoidFlockers,
    BoltzmannWealth,
    Schelling,
    SugarscapeG1mt,
    WolfSheep,
)
from mesa.examples.advanced.sugarscape_g1mt.model import SugarScapeScenario
from mesa.examples.advanced.wolf_sheep.model import WolfSheepScenario
from mesa.examples.basic.boid_flockers.model import BoidsScenario
from mesa.examples.basic.boltzmann_wealth_model.model import BoltzmannScenario
from mesa.examples.basic.schelling.model import SchellingScenario

configurations = {
    # BoltzmannWealth Model Configurations
    BoltzmannWealth: {
        "small": {
            "seeds": 50,
            "replications": 5,
            "steps": 125,
            "scenario_class": BoltzmannScenario,
            "parameters": {
                "n": 100,
                "width": 10,
                "height": 10,
            },
        },
        "large": {
            "seeds": 10,
            "replications": 3,
            "steps": 10,
            "scenario_class": BoltzmannScenario,
            "parameters": {
                "n": 10000,
                "width": 100,
                "height": 100,
            },
        },
    },
    # Schelling Model Configurations
    Schelling: {
        "small": {
            "seeds": 50,
            "replications": 5,
            "steps": 20,
            "scenario_class": SchellingScenario,
            "parameters": {
                "height": 40,
                "width": 40,
                "homophily": 0.4,
                "radius": 1,
                "density": 0.625,
            },
        },
        "large": {
            "seeds": 10,
            "replications": 3,
            "steps": 10,
            "scenario_class": SchellingScenario,
            "parameters": {
                "height": 100,
                "width": 100,
                "homophily": 1,
                "radius": 2,
                "density": 0.8,
            },
        },
    },
    WolfSheep: {
        "small": {
            "seeds": 50,
            "replications": 5,
            "steps": 80,
            "scenario_class": WolfSheepScenario,
            "parameters": {
                "height": 25,
                "width": 25,
                "initial_sheep": 60,
                "initial_wolves": 40,
                "sheep_reproduce": 0.2,
                "wolf_reproduce": 0.1,
                "grass_regrowth_time": 20,
            },
        },
        "large": {
            "seeds": 10,
            "replications": 3,
            "steps": 20,
            "scenario_class": WolfSheepScenario,
            "parameters": {
                "height": 100,
                "width": 100,
                "initial_sheep": 1000,
                "initial_wolves": 500,
                "sheep_reproduce": 0.4,
                "wolf_reproduce": 0.2,
                "grass_regrowth_time": 10,
            },
        },
    },
    SugarscapeG1mt: {
        "small": {
            "seeds": 50,
            "replications": 5,
            "steps": 50,
            "scenario_class": SugarScapeScenario,
            "parameters": {
                "initial_population": 100,
                "enable_trade": False,
                "vision_min": 1,
                "vision_max": 2,
                "endowment_min": 25,
                "endowment_max": 50,
                "metabolism_min": 1,
                "metabolism_max": 5,
            },
        },
        "large": {
            "seeds": 10,
            "replications": 3,
            "steps": 50,
            "scenario_class": SugarScapeScenario,
            "parameters": {
                "initial_population": 250,
                "enable_trade": True,
                "vision_min": 4,
                "vision_max": 5,
                "endowment_min": 25,
                "endowment_max": 50,
                "metabolism_min": 1,
                "metabolism_max": 5,
            },
        },
    },
    BoidFlockers: {
        "small": {
            "seeds": 25,
            "replications": 3,
            "steps": 20,
            "scenario_class": BoidsScenario,
            "parameters": {
                "population_size": 200,
                "vision": 5.0,
            },
        },
        "large": {
            "seeds": 10,
            "replications": 3,
            "steps": 10,
            "scenario_class": BoidsScenario,
            "parameters": {
                "population_size": 400,
                "width": 150,
                "height": 150,
                "vision": 15.0,
            },
        },
    },
}
