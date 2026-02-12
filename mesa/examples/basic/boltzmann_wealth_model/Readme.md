# Boltzmann Wealth Model (Tutorial)

## Summary

A simple model of agents exchanging wealth. All agents start with the same amount of money. Every step, each agent with one unit of money or more gives one unit of wealth to another random agent. Mesa's [Getting Started](https://mesa.readthedocs.io/latest/getting_started.html) section walks through the Boltzmann Wealth Model in a series of short introductory tutorials, starting with[Creating your First Model](https://mesa.readthedocs.io/latest/tutorials/0_first_model.html).

As the model runs, the distribution of wealth among agents goes from being perfectly uniform (all agents have the same starting wealth), to highly skewed -- a small number have high wealth, more have none at all.

## How to Run

To run the model interactively, in this directory, run the following command

```
    $ solara run app.py
```
## Files

* ``model.py``: Final version of the model.
* ``agents.py``: Final version of the agent.
* ``app.py``: Code for the interactive visualization.

## How the Model Is Structured

This example follows Mesa's standard separation of concerns:

- `agents.py`: Defines individual agent behavior (WealthAgent with its step method for giving money to other agents)
- `model.py`: Manages the simulation environment, instantiates agents, handles the grid/space, and collects data
- `app.py`: Sets up the visualization components to display the model in a web interface
- `st_app.py`: (Optional) Alternative Streamlit-based visualization

The visualization displays the state of the model but does not influence how agents behave or how the system evolves over time.

## Understanding the Output

The **Gini coefficient** shown in the visualization measures wealth inequality, ranging from 0 (perfect equality) to 1 (perfect inequality). It is computed in `model.py` using Mesa's `DataCollector` at each simulation step.

## Optional

An optional visualization is also provided using Streamlit, which is another popular Python library for creating interactive web applications.

To run the Streamlit app, you will need to install the `streamlit` and `altair` libraries:

```
    $ pip install streamlit altair
```

Then, you can run the Streamlit app using the following command:

```
    $ streamlit run st_app.py
```
## Further Reading

This model is drawn from econophysics and presents a statistical mechanics approach to wealth distribution. Some examples of further reading on the topic can be found at:

[Milakovic, M. A Statistical Equilibrium Model of Wealth Distribution. February, 2001.](https://editorialexpress.com/cgi-bin/conference/download.cgi?db_name=SCE2001&paper_id=214)

[Dragulescu, A and Yakovenko, V. Statistical Mechanics of Money, Income, and Wealth: A Short Survey. November, 2002](http://arxiv.org/pdf/cond-mat/0211175v1.pdf)
