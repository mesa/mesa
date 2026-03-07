# Virus on a Network

## Summary

This model is based on the NetLogo model "Virus on Network". It demonstrates the spread of a virus through a network and follows the SIR model, commonly seen in epidemiology.

The SIR model is one of the simplest compartmental models, and many models are derivatives of this basic form. The model consists of three compartments:

S: The number of susceptible individuals. When a susceptible and an infectious individual come into "infectious contact", the susceptible individual contracts the disease and transitions to the infectious compartment.
I: The number of infectious individuals. These are individuals who have been infected and are capable of infecting susceptible individuals.
R for the number of removed (and immune) or deceased individuals. These are individuals who have been infected and have either recovered from the disease and entered the removed compartment, or died. It is assumed that the number of deaths is negligible with respect to the total population. This compartment may also be called "recovered" or "resistant".

For more information about this model, read the NetLogo's web page: http://ccl.northwestern.edu/netlogo/models/VirusonaNetwork.

JavaScript library used in this example to render the network: [d3.js](https://d3js.org/).

## Installation

## How to Run

To run the model interactively:

1. Install Mesa in editable mode from the repository root:

pip install -e .

2. Navigate to the example directory:

cd mesa/examples/basic/schelling

3. Start the interactive visualization:
```
    $ solara run app.py
``

## How to Run

To run the model interactively, in this directory, run the following command

```
    $ solara run app.py
```

## Files

* ``model.py``: Contains the agent class, and the overall model class.
* ``agents.py``: Contains the agent class.
* ``app.py``: Contains the code for the interactive Solara visualization.

## Further Reading

[Stonedahl, F. and Wilensky, U. (2008). NetLogo Virus on a Network model](http://ccl.northwestern.edu/netlogo/models/VirusonaNetwork).
Center for Connected Learning and Computer-Based Modeling, Northwestern University, Evanston, IL.


[Wilensky, U. (1999). NetLogo](http://ccl.northwestern.edu/netlogo/)
Center for Connected Learning and Computer-Based Modeling, Northwestern University, Evanston, IL.
