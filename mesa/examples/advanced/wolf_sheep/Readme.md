# Wolf-Sheep Predation Model

## Summary

A simple ecological model, consisting of three agent types: wolves, sheep, and grass. The wolves and the sheep wander around the grid at random. Wolves and sheep both expend energy moving around, and replenish it by eating. Sheep eat grass, and wolves eat sheep if they end up on the same grid cell.

If wolves and sheep have enough energy, they reproduce, creating a new wolf or sheep (in this simplified model, only one parent is needed for reproduction). The grass on each cell regrows at a constant rate. If any wolves and sheep run out of energy, they die.

The model is tests and demonstrates several Mesa concepts and features:
 - MultiGrid
 - Multiple agent types (wolves, sheep, grass)
 - Overlay arbitrary text (wolf's energy) on agent's shapes while drawing on CanvasGrid
 - Agents inheriting a behavior (random movement) from an abstract parent
 - Writing a model composed of multiple files.
 - Dynamically adding and removing agents from the schedule

## How to Run

Install Mesa with recommended dependencies:

pip install "mesa[rec]"

Then run the example:

solara run app.py

Open the displayed local URL in your browser.

## Files

* ``wolf_sheep/random_walk.py``: This defines the ``RandomWalker`` agent, which implements the behavior of moving randomly across a grid, one cell at a time. Both the Wolf and Sheep agents will inherit from it.
* ``wolf_sheep/test_random_walk.py``: Defines a simple model and a text-only visualization intended to make sure the RandomWalk class was working as expected. This doesn't actually model anything, but serves as an ad-hoc unit test. To run it, ``cd`` into the ``wolf_sheep`` directory and run ``python test_random_walk.py``. You'll see a series of ASCII grids, one per model step, with each cell showing a count of the number of agents in it.
* ``wolf_sheep/agents.py``: Defines the Wolf, Sheep, and GrassPatch agent classes.
* ``wolf_sheep/scheduler.py``: Defines a custom variant on the RandomActivationByType scheduler, where we can define filters for the `get_type_count` function.
* ``wolf_sheep/model.py``: Defines the Wolf-Sheep Predation model itself
* ``wolf_sheep/server.py``: Sets up the interactive visualization server
* ``run.py``: Launches a model visualization server.

## Further Reading

This model is closely based on the NetLogo Wolf-Sheep Predation Model:

Wilensky, U. (1997). NetLogo Wolf Sheep Predation model. http://ccl.northwestern.edu/netlogo/models/WolfSheepPredation. Center for Connected Learning and Computer-Based Modeling, Northwestern University, Evanston, IL.

See also the [Lotka–Volterra equations
](https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations) for an example of a classic differential-equation model with similar dynamics.
