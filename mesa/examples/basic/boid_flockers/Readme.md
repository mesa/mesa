# Boids Flockers

## Summary

An implementation of Craig Reynolds's Boids flocker model. Agents (simulated birds) try to fly towards the average position of their neighbors and in the same direction as them, while maintaining a minimum distance. This produces flocking behavior.

This model tests Mesa's continuous space feature, and uses numpy arrays to represent vectors.

## How to Run

Install Mesa with recommended dependencies:

pip install "mesa[rec]"

Then run the example:

solara run app.py

Open the displayed local URL in your browser.


## Files

* [model.py](https://github.com/projectmesa/mesa/blob/main/mesa/examples/basic/boid_flockers/model.py): Contains the Boid Model
* [agents.py](https://github.com/projectmesa/mesa/blob/main/mesa/examples/basic/boid_flockers/agents.py): Contains the Boid agent
* [app.py](https://github.com/projectmesa/mesa/blob/main/mesa/examples/basic/boid_flockers/app.py): Solara based Visualization code.

## Further Reading

The following link can be visited for more information on the boid flockers model:
https://cs.stanford.edu/people/eroberts/courses/soco/projects/2008-09/modeling-natural-systems/boids.html
