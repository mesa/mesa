"""configurations for benchmarks."""

from mesa.examples import (
    BoidFlockers,
    BoltzmannWealth,
    Schelling,
    SugarscapeG1mt,
    WolfSheep,
)

configurations = {
    # BoltzmannWealth Model Configurations
    BoltzmannWealth: {
        "small": {
            "seeds": 50,
            "replications": 5,
            "steps": 125,
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
            "parameters": {
                "height": 40,
                "width": 40,
                "homophily": 0.4,
                "radius": 1,
                "density": 0.625,
                "minority_pc": 0.5,
            },
        },
        "large": {
            "seeds": 10,
            "replications": 3,
            "steps": 10,
            "parameters": {
                "height": 100,
                "width": 100,
                "homophily": 1,
                "radius": 2,
                "density": 0.8,
                "minority_pc": 0.5,
            },
        },
    },
    WolfSheep: {
        "small": {
            "seeds": 50,
            "replications": 5,
            "steps": 80,
            "parameters": {
                "height": 25,
                "width": 25,
                "initial_sheep": 60,
                "initial_wolves": 40,
                "sheep_reproduce": 0.2,
                "wolf_reproduce": 0.1,
                "wolf_gain_from_food": 20.0,
                "grass": True,
                "grass_regrowth_time": 20,
                "sheep_gain_from_food": 4.0,
            },
        },
        "large": {
            "seeds": 10,
            "replications": 3,
            "steps": 20,
            "parameters": {
                "height": 100,
                "width": 100,
                "initial_sheep": 1000,
                "initial_wolves": 500,
                "sheep_reproduce": 0.4,
                "wolf_reproduce": 0.2,
                "wolf_gain_from_food": 20.0,
                "grass": True,
                "grass_regrowth_time": 10,
                "sheep_gain_from_food": 4.0,
            },
        },
    },
    SugarscapeG1mt: {
        "small": {
            "seeds": 50,
            "replications": 5,
            "steps": 50,
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
            "parameters": {
                "population_size": 200,
                "width": 100,
                "height": 100,
                "speed": 1.0,
                "vision": 5.0,
                "separation": 2.0,
                "cohere": 0.03,
                "separate": 0.015,
                "match": 0.05,
            },
        },
        "large": {
            "seeds": 10,
            "replications": 3,
            "steps": 10,
            "parameters": {
                "population_size": 400,
                "width": 150,
                "height": 150,
                "speed": 1.0,
                "vision": 15.0,
                "separation": 2.0,
                "cohere": 0.03,
                "separate": 0.015,
                "match": 0.05,
            },
        },
    },
}
