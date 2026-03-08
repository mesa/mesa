# Boids Flockers

## Summary

An implementation of Craig Reynolds's Boids flocker model. Agents (simulated birds) try to fly towards the average position of their neighbors and in the same direction as them, while maintaining a minimum distance. This produces flocking behavior.

This model tests Mesa's continuous space feature, and uses numpy arrays to represent vectors.

## How to Run

1. Install Mesa and required dependencies:

pip install -U mesa
pip install "mesa[rec]"
pip install solara

2. Run the example:

solara run app.py

3. Open your browser and go to:

http://localhost:8765



## Files

* [model.py](model.py): Contains the Boid Model
* [agents.py](agents.py): Contains the Boid agent
* [app.py](app.py): Solara based Visualization code.

## Further Reading

The following link can be visited for more information on the boid flockers model:
https://cs.stanford.edu/people/eroberts/courses/soco/projects/2008-09/modeling-natural-systems/boids.html
