# Mesa Core Examples
This repository contains a curated set of classic agent-based models implemented using Mesa. These core examples are maintained by the Mesa development team and serve as both demonstrations of Mesa's capabilities and starting points for your own models.

## Overview
The examples are categorized into two groups:
1. **Basic Examples** - Simpler models that use only stable Mesa features; ideal for beginners
2. **Advanced Examples** - More complex models that demonstrate additional concepts and may use some experimental features

> **Note:** Looking for more examples? Visit the [mesa-examples](https://github.com/mesa/mesa-examples) repository for user-contributed models and showcases.

## Basic Examples
The basic examples are relatively simple and only use stable Mesa features. They are good starting points for learning how to use Mesa.

_An online version of these examples is [available here](https://py.cafe/app/EwoutH/mesa-solara-basic-examples)._

### [Boltzmann Wealth Model](examples/basic/boltzmann_wealth_model)
Completed code to go along with the [tutorial](https://mesa.readthedocs.io/latest/tutorials/0_first_model.html) on making a simple model of how a highly-skewed wealth distribution can emerge from simple rules.

### [Boids Flockers Model](examples/basic/boid_flockers)
[Boids](https://en.wikipedia.org/wiki/Boids)-style flocking model, demonstrating the use of agents moving through a continuous space following direction vectors.

### [Conway's Game of Life](examples/basic/conways_game_of_life)
Implementation of [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life), a cellular automata where simple rules can give rise to complex patterns.

### [Schelling Segregation Model](examples/basic/schelling)
Mesa implementation of the classic [Schelling segregation](http://nifty.stanford.edu/2014/mccown-schelling-model-segregation/) model.

### [Virus on a Network Model](examples/basic/virus_on_network)
This model is based on the NetLogo [Virus on a Network](https://ccl.northwestern.edu/netlogo/models/VirusonaNetwork) model.

## Advanced Examples
The advanced examples are more complex and may use experimental Mesa features. They are good starting points for learning how to build more complex models.

### [Epstein Civil Violence Model](examples/advanced/epstein_civil_violence)
Joshua Epstein's [model](https://www.pnas.org/doi/10.1073/pnas.092080199) of how a decentralized uprising can be suppressed or reach a critical mass of support.

### [Demographic Prisoner's Dilemma on a Grid](examples/advanced/pd_grid)
Grid-based demographic prisoner's dilemma model, demonstrating how simple rules can lead to the emergence of widespread cooperation -- and how a model activation regime can change its outcome.

### [Sugarscape Model with Traders](examples/advanced/sugarscape_g1mt)
This is Epstein & Axtell's Sugarscape model with Traders, a detailed description is in Chapter four of *Growing Artificial Societies: Social Science from the Bottom Up (1996)*. The model shows how emergent price equilibrium can happen via decentralized dynamics.

### [Wolf-Sheep Predation Model](examples/advanced/wolf_sheep)
Implementation of an ecological model of predation and reproduction, based on the NetLogo [Wolf Sheep Predation](http://ccl.northwestern.edu/netlogo/models/WolfSheepPredation) model.
