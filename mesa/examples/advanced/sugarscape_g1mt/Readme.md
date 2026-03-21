# Sugarscape Constant Growback Model with Traders

## Summary

This is Epstein & Axtell's Sugarscape model with Traders, a detailed description is in Chapter four of
*Growing Artificial Societies: Social Science from the Bottom Up (1996)*. The model shows an emergent price equilibrium can happen via a decentralized dynamics.

This code generally matches the code in the Complexity Explorer Tutorial, but in `.py` instead of `.ipynb` format.

### Agents:

- **Resource**:  Resource agents grow back at one unit of sugar and spice per time step up to a specified max amount and can be harvested and traded by the trader agents.
  (if you do the interactive run, the color will be green if the resource agent has a bigger amount of sugar, or yellow if it has a bigger amount of spice)
- **Traders**: Trader agents have the following attributes: (1) metabolism for sugar, (2) metabolism for spice, (3) vision,
  (4) initial sugar endowment and (5) initial spice endowment. The traverse the landscape harvesting sugar and spice and
trading with other agents. If they run out of sugar or spice then they are removed from the model. (red circle if you do the interactive run)

The trader agents traverse the landscape according to rule **M**:
- Look out as far as vision permits in the four principal lattice directions and identify the unoccupied site(s).
- Considering only unoccupied sites find the nearest position that produces the most welfare using the Cobb-Douglas function.
- Move to the new position
- Collect all the resources (sugar and spice) at that location
(Epstein and Axtell, 1996, p. 99)

The traders trade according to rule **T**:
- Agents and potential trade partner compute their marginal rates of substitution (MRS), if they are equal *end*.
- Exchange resources, with spice flowing from the agent with the higher MRS to the agent with the lower MRS and sugar
flowing the opposite direction.
- The price (p) is calculated by taking the geometric mean of the agents' MRS.
- If p > 1 then p units of spice are traded for 1 unit of sugar; if p < 1 then 1/p units of sugar for 1 unit of spice
- The trade occurs if it will (a) make both agent better off (increases MRS) and (b) does not cause the agents' MRS to
cross over one another otherwise *end*.
- This process then repeats until an *end* condition is met.
(Epstein and Axtell, 1996, p. 105)

The model demonstrates several Mesa concepts and features:
 - OrthogonalMooreGrid
 - Multiple agent types (traders, sugar, spice)
 - Dynamically removing agents from the grid and schedule when they die
 - Data Collection at the model and agent level
 - custom solara matplotlib space visualization


## How to Run

Install Mesa with recommended dependencies:

pip install "mesa[rec]"

Then run the example:

solara run app.py

Open the displayed local URL in your browser.


## Files

* `model.py`: The Sugarscape Constant Growback with Traders model.
* `agents.py`: Defines the Trader agent class and the Resource agent class which contains an amount of sugar and spice.
* `app.py`: Runs a visualization server via Solara (`solara run app.py`).
* `sugar_map.txt`: Provides sugar and spice landscape in raster type format.
* `tests.py`: Has tests to ensure that the model reproduces the results in shown in Growing Artificial Societies.

## Further Reading

- [Growing Artificial Societies](https://mitpress.mit.edu/9780262550253/growing-artificial-societies/)
- [Complexity Explorer Sugarscape with Traders Tutorial](https://www.complexityexplorer.org/courses/172-agent-based-models-with-python-an-introduction-to-mesa)
